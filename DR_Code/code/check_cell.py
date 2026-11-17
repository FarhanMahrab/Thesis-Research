import os
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
Y = ROOT / "processed" / "yolo_full"
COLORS = {0: (255, 255, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 128, 255)}   # MA HE EX SE
NAMES = ["MA", "HE", "EX", "SE"]

# ---- A. image sizes and boxes per image ----
b = pd.read_csv(ROOT / "splits" / "lesion_boxes_all.csv")
imgs = b.drop_duplicates("image")
for src in ["DDR", "IDRiD"]:
    s = imgs[imgs.source == src]
    print(f"{src}: image width  min {s.w.min()}  median {int(s.w.median())}  max {s.w.max()}")
    per = b[(b.source == src) & (b.cls >= 0)].groupby("image").size()
    print(f"{src}: boxes per image  median {int(per.median())}  max {per.max()}"
          f"  images with >300 boxes: {(per > 300).sum()}")

# ---- B. draw boxes on 3 training images ----
rng = np.random.default_rng(1)
train_imgs = sorted((Y / "images" / "train").glob("*"))
pick = [p for p in train_imgs if p.name.startswith("ddr_")]
pick = list(rng.choice(pick, 2, replace=False)) + [p for p in train_imgs if p.name.startswith("idrid_")][:1]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, p in zip(axes, pick):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    t = max(1, w // 800)
    for line in (Y / "labels" / "train" / f"{p.stem}.txt").read_text().splitlines():
        c, cx, cy, bw, bh = line.split()
        c, cx, cy, bw, bh = int(c), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        cv2.rectangle(im, (int(cx - bw / 2) - 2, int(cy - bh / 2) - 2),
                      (int(cx + bw / 2) + 2, int(cy + bh / 2) + 2), COLORS[c], t)
    ax.imshow(im); ax.set_title(p.name); ax.axis("off")
plt.suptitle("Boxes: yellow = MA, red = HE, green = EX, blue = SE")
plt.tight_layout(); plt.show()

# ---- C. three APTOS duplicate pairs (the least similar ones = most likely mistakes) ----
pairs = pd.read_csv(ROOT / "splits" / "aptos_duplicate_pairs.csv")
pairs = pairs.sort_values("bits_different", ascending=False).head(3)
ap = ROOT / "APTOS" / "aptos2019-blindness-detection" / "train_images"
fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 4 * len(pairs)), squeeze=False)
for row, (_, r) in zip(axes, pairs.iterrows()):
    for ax, k in zip(row, ["a", "b"]):
        im = cv2.cvtColor(cv2.imread(str(ap / f"{r['img_' + k]}.png")), cv2.COLOR_BGR2RGB)
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"{r['img_' + k]}  grade {r['label_' + k]}  ({r.bits_different} bits diff)")
plt.tight_layout(); plt.show()
