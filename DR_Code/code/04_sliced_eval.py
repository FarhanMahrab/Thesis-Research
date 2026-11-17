"""
04_sliced_eval.py  -  Fair full-image test scoring for every detector
=====================================================================

All models are scored on the SAME test images (retina-cropped, retina = 2048 px,
made by 03_make_tiles.py) with the SAME matching and AP code (Ultralytics' own).

  mode "full" : the whole test image is given to the model at once (resized to imgsz)
  mode "tiled": the test image is cut into overlapping imgsz x imgsz tiles, every
                tile is predicted, boxes are shifted back and merged (class-wise NMS)

Usage in Jupyter:
    %run F:/drthesis/code/04_sliced_eval.py baseline640 full 640
    %run F:/drthesis/code/04_sliced_eval.py tile1024 tiled 1024
          (arguments: run-name prefix, mode, image/tile size)

Prints AP50 per lesion for every seed, then mean +- std, model size, GFLOPs and
speed, and saves  F:/drthesis/results/<prefix>_sliced.csv
"""
import copy
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms, box_iou
_code_dir = (Path(globals()["__file__"]).resolve().parent if "__file__" in globals()
             else Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis")) / "code")
sys.path.insert(0, str(_code_dir))
try:
    import drmodules  # noqa: F401  (needed to load models that contain SimAM / CBAM layers)
except ImportError:
    pass
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "tile1024"
MODE = sys.argv[2] if len(sys.argv) > 2 else "tiled"
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
TILES_DIR = ROOT / "processed" / os.environ.get("TILES_NAME", "tiles_1024_d2048")
OVERLAP = 0.25
MERGE_IOU = 0.5          # NMS IoU when merging boxes from overlapping tiles
MAX_PER_IMAGE = 3000
CLASSES = ["MA", "HE", "EX", "SE"]
IOUV = np.linspace(0.5, 0.95, 10)
TEST_SETS = ["test_ddr", "test_idrid"]
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DUPLICATES_REMOVED = [0]  # duplicate ground-truth boxes skipped in the current test set


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def predict_image(model, img):
    """Returns boxes (N,4 xyxy), scores (N,), classes (N,) in full-image pixels."""
    if MODE == "full":
        r = model.predict(img, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=MAX_PER_IMAGE,
                          device=DEVICE, verbose=False)[0].boxes
        return r.xyxy.cpu(), r.conf.cpu(), r.cls.cpu(), 1
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, 0, max(0, IMGSZ - H), 0, max(0, IMGSZ - W),
                             cv2.BORDER_CONSTANT, value=0)
    stride = int(IMGSZ * (1 - OVERLAP))
    pos = [(y, x) for y in starts(pad.shape[0], IMGSZ, stride) for x in starts(pad.shape[1], IMGSZ, stride)]
    tiles = [pad[y:y + IMGSZ, x:x + IMGSZ] for y, x in pos]
    res = model.predict(tiles, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=1000,
                        device=DEVICE, verbose=False, batch=8)
    B, S, C = [], [], []
    for (y, x), r in zip(pos, res):
        b = r.boxes
        if len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().clone()
        xyxy[:, [0, 2]] += x
        xyxy[:, [1, 3]] += y
        B.append(xyxy); S.append(b.conf.cpu()); C.append(b.cls.cpu())
    if not B:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), len(pos)
    B, S, C = torch.cat(B), torch.cat(S), torch.cat(C)
    B[:, [0, 2]] = B[:, [0, 2]].clamp(0, W)
    B[:, [1, 3]] = B[:, [1, 3]].clamp(0, H)
    keep = batched_nms(B, S, C.long(), MERGE_IOU)[:MAX_PER_IMAGE]
    return B[keep], S[keep], C[keep], len(pos)


def read_gt(label_path, W, H):
    """Ground-truth boxes. Exact duplicate boxes (present in DDR's official XML files) are
    counted once, exactly as Ultralytics does when it loads labels for training and val."""
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    unique = list(dict.fromkeys(lines))
    DUPLICATES_REMOVED[0] += len(lines) - len(unique)
    rows = [l.split() for l in unique]
    if not rows:
        return torch.zeros((0, 4)), torch.zeros(0)
    a = torch.tensor([[float(v) for v in r] for r in rows])
    cls, cx, cy, w, h = a[:, 0], a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), cls


def match(pred_cls, gt_cls, iou):
    """Ultralytics' greedy matching. iou: (n_gt, n_pred), preds sorted by confidence."""
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return correct
    iou = (iou * (gt_cls[:, None] == pred_cls[None, :])).numpy()
    matched = np.zeros((iou.shape[0], len(IOUV)), dtype=bool)
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        avail = np.where(matched, 0, iou[:, j, None])
        k = avail.argmax(0)
        correct[j] = avail[k, range(len(IOUV))] >= IOUV
        matched[k, range(len(IOUV))] |= correct[j]
    return correct


def gflops(model):
    try:
        import thop
        m = copy.deepcopy(model.model).float().cpu().eval()
        s = int(np.ceil(IMGSZ / 32) * 32)                 # YOLO pads inputs to a multiple of 32
        x = torch.zeros(1, 3, s, s)
        return thop.profile(m, inputs=[x], verbose=False)[0] * 2 / 1e9
    except Exception as e:
        print(f"  (GFLOPs could not be measured: {e})")
        return float("nan")


def main():
    runs = sorted((ROOT / "runs").glob(f"{PREFIX}_s*/weights/best.pt"))
    if not runs:
        raise SystemExit(f"No runs found: {ROOT / 'runs'}/{PREFIX}_s*/weights/best.pt")
    print(f"{len(runs)} runs | mode={MODE} | size={IMGSZ} | test images from {TILES_DIR / 'test_full'}\n")
    rows = []
    for best in runs:
        run = best.parent.parent.name
        model = YOLO(str(best))
        n_params = sum(p.numel() for p in model.model.parameters())
        gf = gflops(model)
        for ts in TEST_SETS:
            imgs = sorted((TILES_DIR / "test_full" / ts / "images").glob("*.jpg"))
            TP, CONF, PCLS, TCLS = [], [], [], []
            t0, n_tiles = time.time(), 0
            DUPLICATES_REMOVED[0] = 0
            for p in imgs:
                img = cv2.imread(str(p))
                H, W = img.shape[:2]
                boxes, scores, cls, nt = predict_image(model, img)
                n_tiles += nt
                order = scores.argsort(descending=True)
                boxes, scores, cls = boxes[order], scores[order], cls[order]
                gt, gcls = read_gt(TILES_DIR / "test_full" / ts / "labels" / f"{p.stem}.txt", W, H)
                iou = box_iou(gt, boxes) if len(gt) and len(boxes) else torch.zeros((len(gt), len(boxes)))
                TP.append(match(cls, gcls, iou)); CONF.append(scores.numpy())
                PCLS.append(cls.numpy()); TCLS.append(gcls.numpy())
            sec = (time.time() - t0) / len(imgs)
            tp, conf = np.concatenate(TP), np.concatenate(CONF)
            pcls, tcls = np.concatenate(PCLS), np.concatenate(TCLS)
            out = ap_per_class(tp, conf, pcls, tcls)
            r, ap, ucls = out[3], out[5], out[6]
            ap50 = {c: float("nan") for c in CLASSES}
            for k, c in enumerate(ucls):
                ap50[CLASSES[int(c)]] = float(ap[k, 0])
            rows.append({"run": run, "test_set": ts, **{f"AP50_{c}": v for c, v in ap50.items()},
                         "mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()),
                         "recall": float(r.mean()), "params_M": n_params / 1e6,
                         "GFLOPs_per_pass": gf, "passes_per_image": n_tiles / len(imgs),
                         "GFLOPs_per_image": gf * n_tiles / len(imgs), "sec_per_image": sec,
                         "gt_boxes": len(tcls), "gt_duplicates_removed": DUPLICATES_REMOVED[0]})
            print(f"  {run} {ts}: mAP50 {rows[-1]['mAP50']:.3f}  ({sec:.2f} s/image, {len(tcls)} "
                  f"lesion boxes, {DUPLICATES_REMOVED[0]} duplicate boxes skipped)", flush=True)

    df = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / f"{PREFIX}_sliced.csv", index=False)
    cols = [f"AP50_{c}" for c in CLASSES] + ["mAP50", "mAP50-95", "recall"]
    for ts in TEST_SETS:
        t = df[df.test_set == ts]
        print("\n" + "=" * 70 + f"\n{PREFIX} ({MODE}, {IMGSZ})  on  {ts}\n" + "=" * 70)
        print(t[["run"] + cols].round(3).to_string(index=False))
        m, s = t[cols].mean(), t[cols].std()
        print("mean +- std: " + "  ".join(f"{c.replace('AP50_', '')} {m[c]:.3f}+-{s[c]:.3f}" for c in cols))
    f = df.iloc[0]
    print(f"\nModel: {f.params_M:.2f} M params | {f.GFLOPs_per_pass:.1f} GFLOPs per pass x "
          f"{df.passes_per_image.mean():.1f} passes = {df.GFLOPs_per_image.mean():.1f} GFLOPs per image"
          f" | {df.sec_per_image.mean():.2f} s per image")
    print(f"Saved: {out_dir / f'{PREFIX}_sliced.csv'}")


if __name__ == "__main__":
    main()
