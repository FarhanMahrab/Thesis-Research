"""
01_prepare_data.py  -  Week 1 data preparation for LA-DRNet-Lite
=================================================================

What this script does (in order):
  1. DDR grading   -> split CSVs (train / valid / test), ungradable images removed
  2. IDRiD grading -> one external-test CSV (all 516 graded images)
  3. APTOS         -> CSV of all 3,662 labelled images + near-duplicate check
  4. IDRiD overlap -> checks whether any IDRiD grading image is the same photo as
                      an IDRiD lesion-training image (those must not be used as
                      "external" grading test images)
  5. Lesion boxes  -> DDR boxes from the official XML files,
                      IDRiD boxes from the masks (connected components, PRISM-DR style)
  6. YOLO dataset  -> full-image YOLO dataset (images are hard-linked, so no extra
                      disk space), plus data.yaml.  Tiling comes in the next script.

Class order used everywhere:  0 = MA, 1 = HE, 2 = EX, 3 = SE

How to run (in Jupyter):
    %run F:/drthesis/code/01_prepare_data.py
Or in Anaconda Prompt (with the drthesis environment active):
    python F:/drthesis/code/01_prepare_data.py

It is safe to run again: it overwrites its own outputs.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

# ----------------------------------------------------------------------------
# 0. PATHS AND SETTINGS  (change ROOT if your project folder moves)
# ----------------------------------------------------------------------------
ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))

DDR = ROOT / "DDR" / "DDR-dataset" / "DDR-dataset"
IDRID = ROOT / "IDRID"
APTOS = ROOT / "APTOS" / "aptos2019-blindness-detection"

IDRID_SEG = IDRID / "A. Segmentation" / "A. Segmentation"
IDRID_GRD = IDRID / "B. Disease Grading" / "B. Disease Grading"

SPLITS = ROOT / "splits"
PROCESSED = ROOT / "processed"
YOLO_DIR = PROCESSED / "yolo_full"

CLASSES = ["MA", "HE", "EX", "SE"]                  # class id = index
IDRID_MASK_FOLDERS = {                              # lesion -> mask sub-folder
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

# Near-duplicate detection: perceptual hash with 16x16 = 256 bits.
# Two images count as duplicates if they differ in at most this many bits.
# Fundus photos all look alike (orange disc on black), so keep this strict.
HASH_SIZE = 16
DUP_MAX_BITS = 6

N_THREADS = 8


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def phash_file(path):
    """Perceptual hash of one image. JPEGs are decoded at reduced size (fast)."""
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))          # speeds up big JPEGs, no effect on PNG
        im = im.convert("L")
        return imagehash.phash(im, hash_size=HASH_SIZE).hash.flatten()


def hash_many(paths, label=""):
    print(f"  hashing {len(paths)} {label} images ...", flush=True)
    with ThreadPoolExecutor(N_THREADS) as ex:
        bits = list(ex.map(phash_file, paths))
    return np.packbits(np.array(bits, dtype=bool), axis=1)    # (N, 32) uint8


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def hamming_pairs(a, b=None, max_bits=DUP_MAX_BITS, chunk=256):
    """All pairs (i, j, dist) with Hamming distance <= max_bits.
    If b is None, compares a with itself (i < j)."""
    same = b is None
    b = a if same else b
    out = []
    for s in range(0, len(a), chunk):
        x = a[s:s + chunk]
        d = _POPCOUNT[np.bitwise_xor(x[:, None, :], b[None, :, :])].sum(-1)
        ii, jj = np.nonzero(d <= max_bits)
        for i, j in zip(ii + s, jj):
            if same and j <= i:
                continue
            out.append((int(i), int(j), int(d[i - s, j])))
    return out


def link_or_copy(src, dst):
    """Hard link (no extra disk space on the same drive); copy if that fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_one(folder, stem, exts=(".jpg", ".jpeg", ".png", ".tif", ".JPG")):
    for e in exts:
        p = folder / f"{stem}{e}"
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------------------
# 1. DDR grading splits
# ----------------------------------------------------------------------------
def prepare_ddr_grading():
    banner("1. DDR grading splits")
    grade_dir = DDR / "DR_grading"
    for split in ["train", "valid", "test"]:
        rows = []
        for line in (grade_dir / f"{split}.txt").read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name, grade = parts[0], int(parts[1])
            rows.append({"path": str(grade_dir / split / name), "label": grade})
        df = pd.DataFrame(rows)
        missing = (~df["path"].map(lambda p: Path(p).exists())).sum()
        ungradable = (df["label"] == 5).sum()
        df = df[df["label"] != 5].reset_index(drop=True)   # 5 = ungradable in DDR
        df.to_csv(SPLITS / f"ddr_grading_{split}.csv", index=False)
        dist = df["label"].value_counts().sort_index().to_dict()
        print(f"  {split:5s}: {len(df):5d} images kept | removed {ungradable} ungradable"
              f" | missing files: {missing} | grades {dist}")


# ----------------------------------------------------------------------------
# 2. IDRiD grading (external test set)
# ----------------------------------------------------------------------------
def prepare_idrid_grading():
    banner("2. IDRiD grading (external test)")
    gt_dir = IDRID_GRD / "2. Groundtruths"
    img_dirs = {"train": IDRID_GRD / "1. Original Images" / "a. Training Set",
                "test": IDRID_GRD / "1. Original Images" / "b. Testing Set"}
    csvs = {"train": next(gt_dir.glob("a.*.csv")), "test": next(gt_dir.glob("b.*.csv"))}
    rows = []
    for part, csv_path in csvs.items():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["Image name"])
        for _, r in df.iterrows():
            stem = str(r["Image name"]).strip()
            p = find_one(img_dirs[part], stem)
            rows.append({"path": str(p) if p else "", "label": int(r["Retinopathy grade"]),
                         "dme": int(r["Risk of macular edema"]), "idrid_part": part,
                         "image": stem})
    out = pd.DataFrame(rows)
    missing = (out["path"] == "").sum()
    out = out[out["path"] != ""].reset_index(drop=True)
    print(f"  {len(out)} graded images | missing files: {missing}"
          f" | grades {out['label'].value_counts().sort_index().to_dict()}")
    return out


# ----------------------------------------------------------------------------
# 3. APTOS + near-duplicate check
# ----------------------------------------------------------------------------
def prepare_aptos():
    banner("3. APTOS + near-duplicate check")
    df = pd.read_csv(APTOS / "train.csv")
    df["path"] = df["id_code"].map(lambda i: str(APTOS / "train_images" / f"{i}.png"))
    df = df.rename(columns={"diagnosis": "label"})[["id_code", "path", "label"]]
    print(f"  {len(df)} labelled images | grades "
          f"{df['label'].value_counts().sort_index().to_dict()}")

    hashes = hash_many(df["path"].tolist(), "APTOS")
    pairs = hamming_pairs(hashes)
    pd.DataFrame(
        [{"img_a": df.id_code[i], "img_b": df.id_code[j], "bits_different": d,
          "label_a": df.label[i], "label_b": df.label[j]} for i, j, d in pairs]
    ).to_csv(SPLITS / "aptos_duplicate_pairs.csv", index=False)

    # group duplicates (union-find)
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    df["group"] = [find(i) for i in range(len(df))]

    keep = []
    n_conflict_groups = 0
    for _, g in df.groupby("group"):
        if len(g) == 1:
            keep.append(g.index[0])
        elif g["label"].nunique() == 1:
            keep.append(g.index[0])          # same photo, same grade: keep one
        else:
            n_conflict_groups += 1           # same photo, different grades: drop all
    clean = df.loc[sorted(keep), ["id_code", "path", "label"]].reset_index(drop=True)
    df[["id_code", "path", "label"]].to_csv(SPLITS / "aptos_all.csv", index=False)
    clean.to_csv(SPLITS / "aptos_dedup.csv", index=False)
    n_dup_groups = (df.groupby("group").size() > 1).sum()
    print(f"  duplicate pairs found: {len(pairs)} | duplicate groups: {n_dup_groups}"
          f" | groups with conflicting grades (all dropped): {n_conflict_groups}")
    print(f"  APTOS after de-duplication: {len(clean)} images"
          f" | grades {clean['label'].value_counts().sort_index().to_dict()}")


# ----------------------------------------------------------------------------
# 4. IDRiD overlap check (grading set vs lesion set)
# ----------------------------------------------------------------------------
def idrid_seg_images():
    base = IDRID_SEG / "1. Original Images"
    return {"train": sorted((base / "a. Training Set").glob("*.jpg")),
            "test": sorted((base / "b. Testing Set").glob("*.jpg"))}


def check_idrid_overlap(idrid_grd):
    banner("4. IDRiD overlap check (lesion images vs grading images)")
    seg = idrid_seg_images()
    seg_paths = seg["train"] + seg["test"]
    seg_part = ["train"] * len(seg["train"]) + ["test"] * len(seg["test"])
    h_seg = hash_many([str(p) for p in seg_paths], "IDRiD lesion")
    h_grd = hash_many(idrid_grd["path"].tolist(), "IDRiD grading")
    pairs = hamming_pairs(h_grd, h_seg)

    idrid_grd["same_as_lesion_image"] = ""
    idrid_grd["same_as_lesion_train"] = False
    for i, j, d in pairs:
        idrid_grd.loc[i, "same_as_lesion_image"] = seg_paths[j].stem
        if seg_part[j] == "train":
            idrid_grd.loc[i, "same_as_lesion_train"] = True
    idrid_grd.to_csv(SPLITS / "idrid_grading_all.csv", index=False)

    clean = idrid_grd[~idrid_grd["same_as_lesion_train"]].reset_index(drop=True)
    clean[["path", "label", "dme", "image"]].to_csv(
        SPLITS / "idrid_grading_external.csv", index=False)
    print(f"  grading images that are the same photo as a lesion image: {len(pairs)}"
          f" (of these, {int(idrid_grd['same_as_lesion_train'].sum())} are detector-training images)")
    print(f"  IDRiD external grading test set: {len(clean)} images"
          f" (detector-training photos removed)")


# ----------------------------------------------------------------------------
# 5. Lesion boxes
# ----------------------------------------------------------------------------
def ddr_boxes():
    """Boxes from DDR's official Pascal-VOC XML files."""
    banner("5a. DDR lesion boxes (official XML)")
    rows, unknown = [], {}
    for split in ["train", "valid", "test"]:
        img_dir = DDR / "lesion_segmentation" / split / "image"
        xml_files = sorted((DDR / "lesion_detection" / split).glob("*.xml"))
        n_noimg = 0
        for xf in xml_files:
            img = find_one(img_dir, xf.stem)
            if img is None:
                n_noimg += 1
                continue
            root = ET.parse(xf).getroot()
            size = root.find("size")
            w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
            if w == 0 or h == 0:
                h, w = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE).shape[:2]
            rows.append({"source": "DDR", "split": split, "image": str(img), "w": w, "h": h,
                         "cls": -1})                    # placeholder row = image with no boxes
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().upper()
                if name not in CLASSES:
                    unknown[name] = unknown.get(name, 0) + 1
                    continue
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
                rows.append({"source": "DDR", "split": split, "image": str(img), "w": w, "h": h,
                             "cls": CLASSES.index(name), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        print(f"  {split:5s}: {len(xml_files)} XML files | without matching image: {n_noimg}")
    if unknown:
        print(f"  WARNING - unknown class names skipped: {unknown}")
    return pd.DataFrame(rows)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:                                   # some .tif files need PIL
        m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def idrid_boxes():
    """Boxes from IDRiD masks: every connected blob (8-connectivity) = one box."""
    banner("5b. IDRiD lesion boxes (from masks, connected components)")
    gt = IDRID_SEG / "2. All Segmentation Groundtruths"
    parts = {"train": "a. Training Set", "test": "b. Testing Set"}
    rows = []
    for split, imgs in idrid_seg_images().items():
        for img in imgs:
            w = h = None
            img_rows = []
            for lesion, folder in IDRID_MASK_FOLDERS.items():
                mp = gt / parts[split] / folder / f"{img.stem}_{lesion}.tif"
                if not mp.exists():
                    continue                        # this image has no such lesion
                m = read_mask(mp)
                h, w = m.shape
                n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                for k in range(1, n):
                    x, y, bw, bh, _area = stats[k]
                    img_rows.append({"cls": CLASSES.index(lesion), "x1": x, "y1": y,
                                     "x2": x + bw, "y2": y + bh})
            if w is None:
                with Image.open(img) as im:
                    w, h = im.size
            rows.append({"source": "IDRiD", "split": split, "image": str(img),
                         "w": w, "h": h, "cls": -1})
            for r in img_rows:
                rows.append({"source": "IDRiD", "split": split, "image": str(img),
                             "w": w, "h": h, **r})
        print(f"  {split:5s}: {len(imgs)} images")
    return pd.DataFrame(rows)


def box_report(df, name):
    b = df[df["cls"] >= 0].copy()
    b["bw"], b["bh"] = b.x2 - b.x1, b.y2 - b.y1
    print(f"\n  {name}: boxes per class and size (pixels, full resolution)")
    for c, lesion in enumerate(CLASSES):
        s = b[b.cls == c]
        if len(s) == 0:
            print(f"    {lesion}: 0 boxes")
            continue
        side = np.sqrt(s.bw * s.bh)
        print(f"    {lesion}: {len(s):6d} boxes | median side {side.median():6.1f} px"
              f" | 10th pct {side.quantile(.1):5.1f} px | images with it: {s.image.nunique()}")


# ----------------------------------------------------------------------------
# 6. YOLO full-image dataset
# ----------------------------------------------------------------------------
def write_yolo(boxes):
    banner("6. YOLO full-image dataset (hard links, no extra disk space)")
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    # which YOLO folder each (source, split) goes to
    target = {("DDR", "train"): "train", ("IDRiD", "train"): "train",
              ("DDR", "valid"): "val",
              ("DDR", "test"): "test_ddr", ("IDRiD", "test"): "test_idrid"}
    counts = {}
    for (src, split, image), g in boxes.groupby(["source", "split", "image"]):
        folder = target[(src, split)]
        stem = f"{src.lower()}_{Path(image).stem}"
        link_or_copy(Path(image), YOLO_DIR / "images" / folder / f"{stem}{Path(image).suffix}")
        w, h = g.w.iloc[0], g.h.iloc[0]
        lines = []
        for _, r in g[g.cls >= 0].iterrows():
            x1, y1 = max(0, r.x1), max(0, r.y1)
            x2, y2 = min(w, r.x2), min(h, r.y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{int(r.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        lab = YOLO_DIR / "labels" / folder / f"{stem}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("\n".join(lines))
        counts[folder] = counts.get(folder, 0) + 1

    yaml = (f"path: {YOLO_DIR.as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test_ddr\n"
            f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (YOLO_DIR / "data.yaml").write_text(yaml)
    for k, v in counts.items():
        print(f"  {k:11s}: {v} images")
    print(f"  data.yaml written to {YOLO_DIR / 'data.yaml'}")


# ----------------------------------------------------------------------------
def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"Project folder: {ROOT}")

    prepare_ddr_grading()
    idrid_grd = prepare_idrid_grading()
    prepare_aptos()
    check_idrid_overlap(idrid_grd)

    boxes = pd.concat([ddr_boxes(), idrid_boxes()], ignore_index=True)
    boxes.to_csv(SPLITS / "lesion_boxes_all.csv", index=False)
    box_report(boxes[boxes.source == "DDR"], "DDR")
    box_report(boxes[boxes.source == "IDRiD"], "IDRiD")
    write_yolo(boxes)

    banner("DONE - outputs")
    for p in sorted(SPLITS.glob("*.csv")):
        print(f"  {p}")
    print(f"  {YOLO_DIR}")


if __name__ == "__main__":
    main()
