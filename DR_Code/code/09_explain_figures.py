"""
09_explain_figures.py  -  Explanation figures and confusion matrices for the thesis
==================================================================================

Run in the Anaconda Prompt from F:\\drthesis, after 07 (G0, G3) has finished:
    python code/09_explain_figures.py

For DDR-test, IDRiD and APTOS it picks example eyes WITHOUT cherry-picking:
  - one random correctly graded eye for every grade 0..4 (by the final model G2 = G0 + G3, seed 0)
  - two random wrongly graded eyes
and draws, for each eye, one row of four pictures:
  1. the photo
  2. the lesions the detector found (boxes, confidence >= 0.25; MA yellow, HE red, EX green, SE blue)
  3. the lesion heat map laid over the photo
  4. Grad-CAM: where the photo model (G0) looked for its answer
with the doctor's grade, G0's, G3's and G2's answers, and the lesion counts written above.

Also draws the confusion matrix of G2 on every test set (all 3 seeds pooled, rows sum to 100 %).

Output: F:/drthesis/results/figures/
    examples_<test set>.png        the rows of example eyes (for the thesis)
    eyes/<test set>_<id>.png       every example eye as its own picture (for slides)
    confusion_G2_<test set>.png
"""
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

CODE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grader", CODE / "07_train_grader.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

OUT = g.RESULTS / "figures"
SEED = 0
CONF_BOX = 0.25
COLORS = {0: (255, 220, 0), 1: (230, 30, 30), 2: (40, 200, 60), 3: (40, 120, 255)}     # RGB: MA HE EX SE
NAMES = ["MA", "HE", "EX", "SE"]
GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]


def load_g0():
    g.PRETRAINED = False                       # weights come from our own checkpoint, no download needed
    m = g.build_model("G0")
    m.load_state_dict(torch.load(g.RUNS / f"ddr_G0_s{SEED}" / "best.pt", map_location="cpu", weights_only=True))
    return m.to(g.DEVICE).eval()


def grad_cam(model, x, target):
    """Grad-CAM on the last feature layer of EfficientNet-B0 (16 x 16 for a 512 px input)."""
    store = {}
    h1 = model.bn2.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = model.bn2.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    x = x[None].to(g.DEVICE).float().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(x)[0, target].backward()
    h1.remove(); h2.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (g.SIZE, g.SIZE), interpolation=cv2.INTER_CUBIC)
    return cam / cam.max() if cam.max() > 0 else cam


def draw_boxes(img, dets):
    out = img.copy()
    for c, p, x1, y1, x2, y2 in dets:
        if p < CONF_BOX:
            continue
        a = (np.array([x1, y1, x2, y2]) * g.SIZE).round().astype(int)
        pad = 2
        cv2.rectangle(out, (a[0] - pad, a[1] - pad), (a[2] + pad, a[3] + pad), COLORS[int(c)], 1)
    return out


def heat_overlay(img, heat):
    out = img.astype(np.float32)
    for c in range(4):
        a = np.sqrt(heat[..., c].astype(np.float32) / 255)[..., None] * 0.85    # sqrt: faint blobs stay visible
        out = out * (1 - a) + np.array(COLORS[c], np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def pick(df, pred, rng):
    y = df.label.values.astype(int)
    rows = []
    for k in range(5):
        ok = np.flatnonzero((y == k) & (pred == k))
        if len(ok):
            rows.append(int(rng.choice(ok)))
    bad = np.flatnonzero(y != pred)
    if len(bad):
        rows += [int(i) for i in rng.choice(bad, min(2, len(bad)), replace=False)]
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eyes").mkdir(exist_ok=True)
    _, _, tests = g.experiment("ddr")
    d = pd.read_csv(g.G / "detections.csv")
    det = {u: x[["cls", "conf", "x1", "y1", "x2", "y2"]].to_numpy(np.float32) for u, x in d.groupby("uid")}
    model = load_g0()
    rng = np.random.default_rng(0)

    for test, df in tests.items():
        p0 = np.load(g.RUNS / f"ddr_G0_s{SEED}" / f"prob_{test}.npy")
        p3 = np.load(g.RUNS / f"ddr_G3_s{SEED}" / f"prob_{test}.npy")
        p2 = (p0 + p3) / 2
        pred = p2.argmax(1)
        idx = pick(df, pred, rng)
        ds = g.GradingSet(df, heat=False, train=False)
        fig, axes = plt.subplots(len(idx), 4, figsize=(13, 3.9 * len(idx)), squeeze=False)
        for row, i in zip(axes, idx):
            uid, y = df.uid[i], int(df.label[i])
            img = cv2.imread(str(g.G / "images512" / f"{uid}.jpg"))[:, :, ::-1]
            heat = cv2.imread(str(g.G / "heat512" / f"{uid}.png"), cv2.IMREAD_UNCHANGED)
            if img.shape[0] != g.SIZE:
                img = cv2.resize(np.ascontiguousarray(img), (g.SIZE, g.SIZE), interpolation=cv2.INTER_AREA)
                heat = cv2.resize(heat, (g.SIZE, g.SIZE), interpolation=cv2.INTER_AREA)
            img = np.ascontiguousarray(img)
            dd = det.get(uid, np.zeros((0, 6), np.float32))
            x, _ = ds[i]
            cam = grad_cam(model, x, int(p0[i].argmax()))
            camimg = (img * 0.55 + plt.cm.jet(cam)[..., :3] * 255 * 0.45).astype(np.uint8)
            counts = "  ".join(f"{n} {int(((dd[:, 0] == k) & (dd[:, 1] >= CONF_BOX)).sum())}" for k, n in enumerate(NAMES))
            ok = "correct" if pred[i] == y else "WRONG"
            title = (f"{test} | doctor: grade {y} | final G2: {pred[i]} ({p2[i].max():.0%}, {ok}) | "
                     f"photo G0: {p0[i].argmax()} | lesions G3: {p3[i].argmax()}\nlesions found (conf >= {CONF_BOX}): {counts}")
            panels = [img, draw_boxes(img, dd), heat_overlay(img, heat), camimg]
            labels = ["Photo", "Detected lesions", "Lesion heat map", "Grad-CAM (photo model G0)"]
            for ax, im, lab in zip(row, panels, labels):
                ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(lab, fontsize=9)
            row[0].set_title(title, loc="left", fontsize=9)
            single, sax = plt.subplots(1, 4, figsize=(13, 3.6))
            for ax, im, lab in zip(sax, panels, labels):
                ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(lab, fontsize=9)
            sax[0].set_title(title, loc="left", fontsize=9)
            single.tight_layout(); single.savefig(OUT / "eyes" / f"{test}_{uid}.png", dpi=150); plt.close(single)
        handles = [plt.Line2D([], [], color=np.array(COLORS[k]) / 255, lw=3, label=n) for k, n in enumerate(NAMES)]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=2.5)
        fig.savefig(OUT / f"examples_{test}.png", dpi=130); plt.close(fig)
        print(f"{test}: {len(idx)} example eyes -> {OUT / f'examples_{test}.png'}", flush=True)

        # confusion matrix of G2, 3 seeds pooled
        cm = np.zeros((5, 5))
        for s in range(3):
            a, b = g.RUNS / f"ddr_G0_s{s}" / f"prob_{test}.npy", g.RUNS / f"ddr_G3_s{s}" / f"prob_{test}.npy"
            if a.exists() and b.exists():
                pp = ((np.load(a) + np.load(b)) / 2).argmax(1)
                for t, q in zip(df.label.values.astype(int), pp):
                    cm[t, q] += 1
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        f2, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{pct[r, c]:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if pct[r, c] > 55 else "black")
        ax.set_xticks(range(5), [s.split()[0] for s in GRADES]); ax.set_yticks(range(5), GRADES, fontsize=8)
        ax.set_xlabel("Model grade (G2 = G0 + G3)"); ax.set_ylabel("Doctor's grade")
        ax.set_title(f"{test}: confusion matrix, 3 seeds pooled", fontsize=10)
        f2.tight_layout(); f2.savefig(OUT / f"confusion_G2_{test}.png", dpi=150); plt.close(f2)
    print(f"\nAll figures saved in {OUT}")


if __name__ == "__main__":
    main()
