"""
drgrading.py  -  Shared helpers for Stage 2 (lesion heat maps + DR grading)
==========================================================================
Used by 06_detect_grading.py and 07_train_grader.py. Keep it in F:/drthesis/code/.

Image geometry used everywhere in Stage 2
  1. crop the black border around the retina                (same rule as 03_make_tiles.py)
  2. resize so the longer side of the crop is TARGET_D px    (2048, as for the detector)
  3. the detector runs on overlapping TILE x TILE tiles of that image
  4. the grader image = that image padded to a centred square, resized to GRADER_SIZE (512)
Detections are stored in step-4 coordinates, normalised to 0..1, so a heat map of any size
can be drawn from them later.
"""
import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

CLASSES = ["MA", "HE", "EX", "SE"]


# ----------------------------------------------------------------------------- geometry
def retina_box(img, thr=15):
    """Bounding box (x0, y0, x1, y1) of the bright retina area (identical to 03_make_tiles.py)."""
    fov = img.max(axis=2) > thr
    rows = np.where(fov.mean(axis=1) > 0.01)[0]
    cols = np.where(fov.mean(axis=0) > 0.01)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, img.shape[1], img.shape[0]
    m = 4
    return (max(0, cols[0] - m), max(0, rows[0] - m),
            min(img.shape[1], cols[-1] + 1 + m), min(img.shape[0], rows[-1] + 1 + m))


def prepare(path, target_d=2048, grader_size=512):
    """Returns (detector image, grader image, info) or (None, None, error message)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "could not read"
    x0, y0, x1, y1 = retina_box(img)
    img = img[y0:y1, x0:x1]
    s = target_d / max(img.shape[:2])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=interp)
    H, W = img.shape[:2]
    S = max(H, W)
    py, px = (S - H) // 2, (S - W) // 2
    square = cv2.copyMakeBorder(img, py, S - H - py, px, S - W - px, cv2.BORDER_CONSTANT, value=0)
    grader = cv2.resize(square, (grader_size, grader_size), interpolation=cv2.INTER_AREA)
    info = {"W": W, "H": H, "S": S, "pad_x": px, "pad_y": py, "crop_x0": x0, "crop_y0": y0, "scale": s}
    return img, grader, info


# ----------------------------------------------------------------------------- detection
def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def tile_predict(model, img, tile=1024, overlap=0.25, conf=0.05, merge_iou=0.5, max_det=3000,
                 device=0, batch=8):
    """Same sliced inference as 04_sliced_eval.py. Returns (N, 6) array: cls, conf, x1, y1, x2, y2
    in pixels of `img`."""
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, 0, max(0, tile - H), 0, max(0, tile - W), cv2.BORDER_CONSTANT, value=0)
    stride = int(tile * (1 - overlap))
    pos = [(y, x) for y in _starts(pad.shape[0], tile, stride) for x in _starts(pad.shape[1], tile, stride)]
    tiles = [pad[y:y + tile, x:x + tile] for y, x in pos]
    res = model.predict(tiles, imgsz=tile, conf=conf, iou=0.6, max_det=1000, device=device,
                        verbose=False, batch=batch)
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
        return np.zeros((0, 6), np.float32)
    B, S, C = torch.cat(B), torch.cat(S), torch.cat(C)
    B[:, [0, 2]] = B[:, [0, 2]].clamp(0, W)
    B[:, [1, 3]] = B[:, [1, 3]].clamp(0, H)
    keep = batched_nms(B, S, C.long(), merge_iou)[:max_det]
    return torch.cat([C[keep, None], S[keep, None], B[keep]], 1).numpy().astype(np.float32)


def to_square_norm(dets, info):
    """Pixel boxes of the detector image -> 0..1 coordinates of the centred square grader frame."""
    d = dets.copy()
    d[:, [2, 4]] = (d[:, [2, 4]] + info["pad_x"]) / info["S"]
    d[:, [3, 5]] = (d[:, [3, 5]] + info["pad_y"]) / info["S"]
    return d


# ----------------------------------------------------------------------------- heat maps
def render_heatmap(dets, size=512, conf_min=0.05):
    """dets: (N, 6) cls, conf, x1, y1, x2, y2 in 0..1 square coordinates.
    Returns (size, size, 4) uint8: one channel per lesion type (MA, HE, EX, SE).
    Every detection is drawn as a Gaussian blob covering its box (box = +-2 sigma),
    with height = detection confidence; overlapping blobs keep the maximum."""
    heat = np.zeros((4, size, size), np.float32)
    if len(dets):
        dets = dets[dets[:, 1] >= conf_min]
    for c, p, x1, y1, x2, y2 in dets:
        cx, cy = (x1 + x2) / 2 * size, (y1 + y2) / 2 * size
        sx, sy = max((x2 - x1) * size / 4, 0.75), max((y2 - y1) * size / 4, 0.75)
        xa, xb = int(max(0, np.floor(cx - 3 * sx))), int(min(size, np.ceil(cx + 3 * sx) + 1))
        ya, yb = int(max(0, np.floor(cy - 3 * sy))), int(min(size, np.ceil(cy + 3 * sy) + 1))
        if xa >= xb or ya >= yb:
            continue
        gx = np.exp(-0.5 * ((np.arange(xa, xb) + 0.5 - cx) / sx) ** 2)
        gy = np.exp(-0.5 * ((np.arange(ya, yb) + 0.5 - cy) / sy) ** 2)
        blob = p * gy[:, None] * gx[None, :]
        ch = heat[int(c), ya:yb, xa:xb]
        np.maximum(ch, blob, out=ch)
    return np.round(heat.transpose(1, 2, 0) * 255).astype(np.uint8)


def lesion_features(dets, thresholds=(0.1, 0.25, 0.5)):
    """Numbers describing one image's detections (input of the G3 'lesion counts' model).
    dets: (N, 6) cls, conf, x1, y1, x2, y2 in 0..1 square coordinates."""
    f = {}
    for k, name in enumerate(CLASSES):
        d = dets[dets[:, 0] == k] if len(dets) else np.zeros((0, 6), np.float32)
        area = (d[:, 4] - d[:, 2]) * (d[:, 5] - d[:, 3]) if len(d) else np.zeros(0)
        for t in thresholds:
            f[f"{name}_n{int(t * 100):02d}"] = int((d[:, 1] >= t).sum()) if len(d) else 0
        f[f"{name}_conf_sum"] = float(d[:, 1].sum()) if len(d) else 0.0
        f[f"{name}_conf_max"] = float(d[:, 1].max()) if len(d) else 0.0
        f[f"{name}_area25"] = float(area[d[:, 1] >= 0.25].sum()) if len(d) else 0.0
    return f
