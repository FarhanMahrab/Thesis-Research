"""
06_detect_grading.py  -  Stage 2, step 1: run the lesion detector on every grading image
========================================================================================

Run in the Anaconda Prompt (NOT Jupyter), from F:\\drthesis :
    python code/06_detect_grading.py

What it does
  1. Collects the grading images: DDR (train/valid/test), IDRiD external test, APTOS,
     and Messidor-2 if the folder F:/drthesis/MESSIDOR2 exists.
  2. LEAKAGE CHECK. A DDR grading photo that is the same photo as a DDR lesion image used to
     TRAIN or SELECT the detector (lesion train / valid split) is removed from every grading
     split: its heat map would be unrealistically good. DDR valid/test photos that are
     near-duplicates of a DDR grading-train photo are removed too.
     (Same photo = same file name, or perceptual hash within 6 of 256 bits, as in 01.)
     IDRiD was already cleaned in 01 (45 detector-training photos removed).
  3. Picks the detector: the p2_1024 seed with the best VALIDATION score (never the test score).
  4. For every image: crop retina -> retina 2048 px -> sliced detection (same as 04)
     -> saves the 512 x 512 grader image, the 4-channel lesion heat map, and all detections.

Output: F:/drthesis/processed/grading/
    index.csv          one row per grading image: dataset, split, label, used or not and why
    detections.csv     every detection (conf >= 0.05), boxes in 0..1 grader-image coordinates
    images512/<uid>.jpg    grader input (RGB)
    heat512/<uid>.png      lesion heat map, 4 channels = MA, HE, EX, SE (PNG with alpha)
    detector.txt       which detector made the detections

Takes roughly 1 - 1.5 hours for DDR + IDRiD + APTOS. If it stops (power cut, Ctrl+C),
run the same command again: finished images are skipped.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import drmodules  # noqa: E402,F401  (lets SimAM/CBAM models load, harmless otherwise)
from drgrading import prepare, render_heatmap, tile_predict, to_square_norm  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
DDR = ROOT / "DDR" / "DDR-dataset" / "DDR-dataset"
SPLITS = ROOT / "splits"
OUT = ROOT / "processed" / "grading"
MESSIDOR = ROOT / "MESSIDOR2"
DET_PREFIX = os.environ.get("DET_PREFIX", "p2_1024")
TARGET_D = int(os.environ.get("TARGET_D", 2048))
TILE = int(os.environ.get("TILE", 1024))
GRADER_SIZE = 512
CONF_SAVE = 0.05
HASH_SIZE, DUP_MAX_BITS = 16, 6
DEVICE = 0 if torch.cuda.is_available() else "cpu"
N_LOADERS = 6
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# ----------------------------------------------------------------------------- hashing (as in 01)
def phash_file(path):
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))
        return imagehash.phash(im.convert("L"), hash_size=HASH_SIZE).hash.flatten()


def hash_many(paths, label):
    print(f"  hashing {len(paths)} {label} images ...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        bits = list(ex.map(phash_file, paths))
    return np.packbits(np.array(bits, dtype=bool), axis=1)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def close_pairs(a, b, chunk=256):
    """(i, j) pairs with Hamming distance <= DUP_MAX_BITS between hash sets a and b."""
    out = []
    for s in range(0, len(a), chunk):
        d = _POP[np.bitwise_xor(a[s:s + chunk, None, :], b[None, :, :])].sum(-1)
        ii, jj = np.nonzero(d <= DUP_MAX_BITS)
        out += [(int(i + s), int(j)) for i, j in zip(ii, jj)]
    return out


# ----------------------------------------------------------------------------- 1. collect images
def collect():
    banner("1. Grading images")
    parts = []
    for split in ["train", "valid", "test"]:
        d = pd.read_csv(SPLITS / f"ddr_grading_{split}.csv")
        d["dataset"], d["split"] = "ddr", split
        d["uid"] = [f"ddr_{split}_{Path(p).stem}" for p in d.path]
        parts.append(d[["uid", "dataset", "split", "path", "label"]])
    d = pd.read_csv(SPLITS / "idrid_grading_external.csv")
    d["dataset"], d["split"] = "idrid", "test"
    # IDRiD numbers its training and testing images separately (both have IDRiD_001 ...)
    d["uid"] = [f"idrid_{'te' if 'Testing' in p else 'tr'}_{Path(p).stem}" for p in d.path]
    parts.append(d[["uid", "dataset", "split", "path", "label"]])
    d = pd.read_csv(SPLITS / "aptos_dedup.csv")
    d["dataset"], d["split"] = "aptos", "all"
    d["uid"] = [f"aptos_{Path(p).stem}" for p in d.path]
    parts.append(d[["uid", "dataset", "split", "path", "label"]])
    m = messidor2()
    if m is not None:
        parts.append(m)
    idx = pd.concat(parts, ignore_index=True)
    n0 = len(idx)
    idx = idx.drop_duplicates(["dataset", "split", "path"]).reset_index(drop=True)
    if len(idx) < n0:
        print(f"  note: {n0 - len(idx)} rows listed twice in a split file were counted once")
    dup = idx.uid.duplicated(keep=False)
    if dup.any():                              # same file name in two folders: keep both, rename the id
        print(f"  note: {int(dup.sum())} images share a file name with another image, e.g. "
              f"{idx.path[dup].head(2).tolist()} - given unique ids")
        idx["uid"] = idx.uid + idx.groupby("uid").cumcount().map(lambda k: "" if k == 0 else f"_{k + 1}")
    assert idx.uid.is_unique
    for (ds, sp), g in idx.groupby(["dataset", "split"], sort=False):
        print(f"  {ds:9s} {sp:5s}: {len(g):5d} images | grades {g.label.value_counts().sort_index().to_dict()}")
    return idx


def messidor2():
    if not MESSIDOR.exists():
        print("  Messidor-2: folder F:/drthesis/MESSIDOR2 not found - skipped for now")
        return None
    csvs = [c for c in MESSIDOR.rglob("*.csv")
            if "adjudicated_dr_grade" in pd.read_csv(c, nrows=0).columns]
    if not csvs:
        print("  Messidor-2: no CSV with an 'adjudicated_dr_grade' column found - skipped")
        return None
    files = {p.stem.lower(): p for p in MESSIDOR.rglob("*") if p.suffix.lower() in IMG_EXT}
    best = None
    for c in csvs:                            # use the grade file that matches the most image files
        lab = pd.read_csv(c)
        idcol = "image_id" if "image_id" in lab.columns else lab.columns[0]
        if "adjudicated_gradable" in lab.columns:
            lab = lab[lab.adjudicated_gradable == 1]
        lab = lab.dropna(subset=["adjudicated_dr_grade"]).copy()
        lab["path"] = [str(files.get(Path(str(i)).stem.lower(), "")) for i in lab[idcol]]
        if best is None or (lab.path != "").sum() > (best[1].path != "").sum():
            best = (c, lab)
    csv_used, lab = best
    missing = (lab.path == "").sum()
    lab = lab[lab.path != ""]
    print(f"  Messidor-2: grades from {csv_used.name} | images found {len(lab)} | without image file: {missing}")
    return pd.DataFrame({"uid": [f"messidor2_{Path(p).stem}" for p in lab.path], "dataset": "messidor2",
                         "split": "all", "path": lab.path.values,
                         "label": lab.adjudicated_dr_grade.astype(int).values})


# ----------------------------------------------------------------------------- 2. leakage check
def leakage_check(idx):
    banner("2. Leakage check (DDR)")
    idx["use"], idx["reason"] = True, ""
    ddr = idx[idx.dataset == "ddr"]
    lesion = {}
    for split in ["train", "valid", "test"]:
        folder = DDR / "lesion_segmentation" / split / "image"
        lesion.update({p: split for p in sorted(folder.glob("*")) if p.suffix.lower() in IMG_EXT})
    les_paths = list(lesion)
    les_split = [lesion[p] for p in les_paths]

    # (a) same file name
    stem_split = {p.stem: s for p, s in zip(les_paths, les_split)}
    match = {i: stem_split[Path(p).stem] for i, p in zip(ddr.index, ddr.path) if Path(p).stem in stem_split}
    # (b) same photo by perceptual hash
    h_les = hash_many([str(p) for p in les_paths], "DDR lesion")
    h_ddr = hash_many(ddr.path.tolist(), "DDR grading")
    for i, j in close_pairs(h_ddr, h_les):
        match.setdefault(ddr.index[i], les_split[j])
    det = {i: s for i, s in match.items() if s in ("train", "valid")}
    idx.loc[list(det), "use"] = False
    idx.loc[list(det), "reason"] = "same photo as a detector train/valid lesion image"
    print(f"  DDR grading photos that are DDR lesion photos: {len(match)} "
          f"(detector train {sum(s == 'train' for s in match.values())}, "
          f"valid {sum(s == 'valid' for s in match.values())}, test {sum(s == 'test' for s in match.values())})")
    print(f"  -> removed (detector train/valid): {len(det)} | kept (lesion test, never trained on): "
          f"{sum(s == 'test' for s in match.values())}")

    # (c) DDR valid/test photos that duplicate a DDR grading-train photo
    tr = np.flatnonzero((ddr.split == "train").values)
    ev = np.flatnonzero((ddr.split != "train").values)
    dup = sorted({ddr.index[ev[i]] for i, _ in close_pairs(h_ddr[ev], h_ddr[tr])})
    new = [i for i in dup if idx.at[i, "use"]]
    idx.loc[new, "use"] = False
    idx.loc[new, "reason"] = "duplicate of a DDR grading-train photo"
    print(f"  DDR valid/test photos that duplicate a DDR train photo: {len(dup)} -> removed {len(new)}")
    for (ds, sp), g in idx.groupby(["dataset", "split"], sort=False):
        print(f"  {ds:9s} {sp:5s}: {int(g.use.sum()):5d} used, {int((~g.use).sum())} removed")
    return idx


# ----------------------------------------------------------------------------- 3. detector
def pick_detector():
    banner("3. Detector")
    best, rows = None, []
    for res in sorted((ROOT / "runs").glob(f"{DET_PREFIX}_s*/results.csv")):
        r = pd.read_csv(res)
        r.columns = [c.strip() for c in r.columns]
        v = float(r["metrics/mAP50-95(B)"].max())
        rows.append((res.parent.name, v))
        if (res.parent / "weights" / "best.pt").exists() and (best is None or v > best[1]):
            best = (res.parent, v)
    for n, v in rows:
        print(f"  {n}: best validation mAP50-95 {v:.4f}")
    if best is None:
        raise SystemExit(f"No finished runs found: {ROOT / 'runs'}/{DET_PREFIX}_s*")
    print(f"  -> using {best[0].name} (chosen on the validation set only)")
    return best[0] / "weights" / "best.pt"


# ----------------------------------------------------------------------------- 4. detection pass
def load_job(row):
    img, grader, info = prepare(row.path, TARGET_D, GRADER_SIZE)
    if img is not None:
        cv2.imwrite(str(OUT / "images512" / f"{row.uid}.jpg"), grader, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return row, img, info


def detect_all(idx, weights):
    banner("4. Detection pass")
    (OUT / "images512").mkdir(parents=True, exist_ok=True)
    (OUT / "heat512").mkdir(parents=True, exist_ok=True)
    tag = OUT / "detector.txt"
    if tag.exists() and tag.read_text().strip() != str(weights):
        raise SystemExit(f"{OUT} was made with another detector ({tag.read_text().strip()}).\n"
                         f"Delete that folder first if you really want to redo it with {weights}.")
    tag.write_text(str(weights))

    det_csv, done_txt = OUT / "detections.csv", OUT / "done.txt"
    done = set(done_txt.read_text().split()) if done_txt.exists() else set()
    if det_csv.exists():                       # drop rows of an image that was cut off half-way
        d = pd.read_csv(det_csv)
        if (~d.uid.isin(done)).any():
            d[d.uid.isin(done)].to_csv(det_csv, index=False, float_format="%.5f")
    todo = [r for r in idx[idx.use].itertuples() if r.uid not in done]
    print(f"  {len(done)} images already done, {len(todo)} to go", flush=True)
    if not todo:
        return
    model = YOLO(str(weights))
    new_file = not det_csv.exists()
    errors = []
    with open(det_csv, "a", newline="") as fcsv, open(done_txt, "a") as fdone, \
            ThreadPoolExecutor(N_LOADERS) as ex:
        if new_file:
            fcsv.write("uid,cls,conf,x1,y1,x2,y2\n")
        t0, chunk = time.time(), 24
        nxt = [ex.submit(load_job, r) for r in todo[:chunk]]
        for start in range(0, len(todo), chunk):
            cur = nxt
            nxt = [ex.submit(load_job, r) for r in todo[start + chunk:start + 2 * chunk]]
            for fut in cur:
                row, img, info = fut.result()
                if img is None:
                    errors.append((row.uid, info))
                    continue
                dets = to_square_norm(tile_predict(model, img, TILE, conf=CONF_SAVE, device=DEVICE), info)
                heat = render_heatmap(dets, GRADER_SIZE)
                cv2.imwrite(str(OUT / "heat512" / f"{row.uid}.png"), heat)
                fcsv.writelines(f"{row.uid},{int(c)},{p:.4f},{a:.5f},{b:.5f},{e:.5f},{f:.5f}\n"
                                for c, p, a, b, e, f in dets)
                fcsv.flush()
                fdone.write(row.uid + "\n"); fdone.flush()
            n = min(start + chunk, len(todo))
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)} images | {rate:.2f} s/image | about {rate * (len(todo) - n) / 60:.0f} min left",
                  flush=True)
    if errors:
        print(f"  {len(errors)} images could not be read, e.g. {errors[:3]}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = collect()
    old = pd.read_csv(OUT / "index.csv", keep_default_na=False) if (OUT / "index.csv").exists() else None
    if old is not None and set(idx.uid[idx.dataset == "ddr"]) == set(old.uid[old.dataset == "ddr"]):
        banner("2. Leakage check (DDR) - already done, re-using index.csv")
        flags = old.set_index("uid")[["use", "reason"]]
        flags["use"] = flags["use"].astype(str).str.lower() == "true"
        idx = idx.join(flags, on="uid")
        new = idx.use.isna()                      # e.g. Messidor-2 added later: nothing to check
        idx.loc[new, "use"], idx.loc[new, "reason"] = True, ""
        idx["use"] = idx["use"].astype(bool)
        idx["reason"] = idx["reason"].fillna("")
        print(f"  removed earlier: {int((~idx.use).sum())} DDR images")
    else:
        idx = leakage_check(idx)
    idx.to_csv(OUT / "index.csv", index=False)
    weights = pick_detector()
    detect_all(idx, weights)
    done = set((OUT / "done.txt").read_text().split())
    idx["processed"] = idx.uid.isin(done)
    idx.to_csv(OUT / "index.csv", index=False)
    d = pd.read_csv(OUT / "detections.csv")
    banner("Done")
    print(f"  images processed: {int(idx.processed.sum())} | detections saved: {len(d)} "
          f"(per image: median {int(d.groupby('uid').size().median()) if len(d) else 0})")
    print(f"  output: {OUT}")


if __name__ == "__main__":
    main()
