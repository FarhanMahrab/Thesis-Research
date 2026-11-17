"""
03_make_tiles.py  -  Retina crop + scale normalisation + overlapping tiles
=========================================================================

For every lesion image (DDR + IDRiD) this script:
  1. crops away the black border around the retina,
  2. resizes the image so the retina is TARGET_D pixels across
     (DDR and IDRiD lesions then have about the same size in pixels),
  3. cuts overlapping TILE x TILE tiles (OVERLAP fraction) for TRAIN and VAL,
  4. saves the TEST images whole (cropped + resized) for sliced evaluation
     in the next script (04), because a fair test must score the full image.

Box rules inside a tile: a lesion box is kept if at least MIN_VISIBLE of its
area lies inside the tile (and the visible part is >= 2 px wide and high).
Tiles without any lesion: all kept for VAL, only KEEP_EMPTY of them for TRAIN.

Output:  F:/drthesis/processed/tiles_<TILE>_d<TARGET_D>/
            images/train, images/val, labels/train, labels/val, data.yaml
            test_full/<test_ddr | test_idrid>/images + labels
            transforms.csv  (crop and scale used for every image)

Run in Jupyter:   %run F:/drthesis/code/03_make_tiles.py
Takes about 5-15 minutes. Safe to run again (it rebuilds its folder).
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
TARGET_D = int(os.environ.get("TARGET_D", 2048))   # retina diameter after resizing (px)
TILE = int(os.environ.get("TILE", 1024))           # tile size (px)
OVERLAP = 0.25                                     # tile overlap fraction
MIN_VISIBLE = 0.4                                  # keep a cut box if >= 40% is visible
KEEP_EMPTY = 0.2                                   # fraction of lesion-free TRAIN tiles kept
SEED = 0
N_THREADS = 8
JPEG_QUALITY = 95
CLASSES = ["MA", "HE", "EX", "SE"]

OUT = ROOT / "processed" / f"tiles_{TILE}_d{TARGET_D}"
TARGET = {("DDR", "train"): "train", ("IDRiD", "train"): "train",
          ("DDR", "valid"): "val",
          ("DDR", "test"): "test_ddr", ("IDRiD", "test"): "test_idrid"}


def retina_box(img, thr=15):
    """Bounding box (x0, y0, x1, y1) of the bright retina area."""
    fov = img.max(axis=2) > thr
    rows = np.where(fov.mean(axis=1) > 0.01)[0]
    cols = np.where(fov.mean(axis=0) > 0.01)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, img.shape[1], img.shape[0]
    m = 4
    return (max(0, cols[0] - m), max(0, rows[0] - m),
            min(img.shape[1], cols[-1] + 1 + m), min(img.shape[0], rows[-1] + 1 + m))


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def to_yolo(boxes, w, h):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return "\n".join(lines)


def process(args):
    (src, split, image), g, rng_seed = args
    folder = TARGET[(src, split)]
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if img is None:
        return {"image": image, "error": "could not read"}
    x0, y0, x1, y1 = retina_box(img)
    img = img[y0:y1, x0:x1]
    s = TARGET_D / max(img.shape[:2])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=interp)
    H, W = img.shape[:2]

    b = g[g.cls >= 0][["cls", "x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if len(b):
        b[:, [1, 3]] = (b[:, [1, 3]] - x0) * s
        b[:, [2, 4]] = (b[:, [2, 4]] - y0) * s
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, W)
        b[:, [2, 4]] = b[:, [2, 4]].clip(0, H)
        b = b[(b[:, 3] - b[:, 1] >= 1) & (b[:, 4] - b[:, 2] >= 1)]
    stem = f"{src.lower()}_{Path(image).stem}"
    info = {"image": image, "source": src, "split": split, "stem": stem, "crop_x0": x0,
            "crop_y0": y0, "scale": s, "W": W, "H": H, "boxes": len(b), "tiles": 0}

    # ---- test images: save whole (sliced evaluation happens later) ----
    if folder.startswith("test"):
        d = OUT / "test_full" / folder
        cv2.imwrite(str(d / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        (d / "labels" / f"{stem}.txt").write_text(to_yolo(b, W, H))
        return info

    # ---- train / val: overlapping tiles ----
    if H < TILE or W < TILE:                           # pad small images up to one tile
        img = cv2.copyMakeBorder(img, 0, max(0, TILE - H), 0, max(0, TILE - W),
                                 cv2.BORDER_CONSTANT, value=0)
    Hp, Wp = img.shape[:2]
    stride = int(TILE * (1 - OVERLAP))
    rng = np.random.default_rng(rng_seed)
    area = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2]) if len(b) else np.array([])
    for ty in starts(Hp, TILE, stride):
        for tx in starts(Wp, TILE, stride):
            kept = []
            if len(b):
                cx1 = np.clip(b[:, 1], tx, tx + TILE); cx2 = np.clip(b[:, 3], tx, tx + TILE)
                cy1 = np.clip(b[:, 2], ty, ty + TILE); cy2 = np.clip(b[:, 4], ty, ty + TILE)
                vis = (cx2 - cx1) * (cy2 - cy1) / np.maximum(area, 1e-6)
                ok = (vis >= MIN_VISIBLE) & (cx2 - cx1 >= 2) & (cy2 - cy1 >= 2)
                kept = [(b[k, 0], cx1[k] - tx, cy1[k] - ty, cx2[k] - tx, cy2[k] - ty)
                        for k in np.nonzero(ok)[0]]
            if not kept and folder == "train" and rng.random() > KEEP_EMPTY:
                continue
            name = f"{stem}_y{ty}_x{tx}"
            tile = img[ty:ty + TILE, tx:tx + TILE]
            cv2.imwrite(str(OUT / "images" / folder / f"{name}.jpg"), tile,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            (OUT / "labels" / folder / f"{name}.txt").write_text(to_yolo(kept, TILE, TILE))
            info["tiles"] += 1
    return info


def main():
    print(f"Retina diameter -> {TARGET_D} px | tiles {TILE} px | overlap {OVERLAP:.0%}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for f in ["train", "val"]:
        (OUT / "images" / f).mkdir(parents=True)
        (OUT / "labels" / f).mkdir(parents=True)
    for f in ["test_ddr", "test_idrid"]:
        (OUT / "test_full" / f / "images").mkdir(parents=True)
        (OUT / "test_full" / f / "labels").mkdir(parents=True)

    boxes = pd.read_csv(ROOT / "splits" / "lesion_boxes_all.csv")
    groups = list(boxes.groupby(["source", "split", "image"]))
    jobs = [(key, g, SEED + i) for i, (key, g) in enumerate(groups)]
    print(f"Processing {len(jobs)} images with {N_THREADS} threads ...", flush=True)
    with ThreadPoolExecutor(N_THREADS) as ex:
        infos = list(ex.map(process, jobs))

    t = pd.DataFrame(infos)
    if "error" in t and t["error"].notna().any():
        print("ERRORS:\n", t[t["error"].notna()][["image", "error"]])
    t.to_csv(OUT / "transforms.csv", index=False)

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    print("\nImages per part:")
    t["part"] = [TARGET[(a, b)] for a, b in zip(t.source, t.split)]
    print(t.groupby("part").agg(images=("image", "count"), tiles=("tiles", "sum")).to_string())

    for f in ["train", "val"]:
        labs = list((OUT / "labels" / f).glob("*.txt"))
        n_box = np.zeros(4, int)
        sizes = {c: [] for c in range(4)}
        empty = 0
        for p in labs:
            txt = p.read_text().strip()
            if not txt:
                empty += 1
                continue
            for line in txt.splitlines():
                c, _, _, w, h = line.split()
                n_box[int(c)] += 1
                sizes[int(c)].append(np.sqrt(float(w) * float(h)) * TILE)
        med = "  ".join(f"{CLASSES[c]} {np.median(sizes[c]):.1f}px" for c in range(4) if sizes[c])
        print(f"\n{f}: {len(labs)} tiles ({empty} without lesions) | boxes "
              + ", ".join(f"{CLASSES[c]} {n_box[c]}" for c in range(4)))
        print(f"   median lesion size after scaling: {med}")
    print(f"\nDone. Dataset: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
