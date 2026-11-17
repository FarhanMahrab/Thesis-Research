"""
07_train_grader.py  -  Stage 2, step 2: DR grading (G0 / G1 / G3), 3 seeds, auto-resume
======================================================================================

Run in the Anaconda Prompt (NOT Jupyter), from F:\\drthesis, after 06_detect_grading.py:

    python code/07_train_grader.py ddr G0      image only           (EfficientNet-B0, RGB)
    python code/07_train_grader.py ddr G1      image + heat maps    (EfficientNet-B0, RGB + 4 lesion channels)
    python code/07_train_grader.py ddr G3      lesion numbers only  (XGBoost on detector counts; CPU, minutes)

    first argument = experiment
      ddr      train DDR-train, pick the best epoch on DDR-valid (quadratic kappa),
               test on DDR-test, IDRiD (external) and APTOS (external)
      aptos    train APTOS (85 %, the other 15 % picks the best epoch), test on Messidor-2
               (needs Messidor-2 processed by 06)
    optional: seeds after the variant, e.g.  ... ddr G1 0 1   (default 0 1 2)

Run the SAME command again after a power cut: finished seeds are skipped, an unfinished
seed continues from its last epoch.

Results: F:/drthesis/runs_grader/<exp>_<variant>_s<seed>/  (best.pt, log.csv, test.json)
         F:/drthesis/results/grader_<exp>_<variant>.csv    (all seeds + mean / std)

G1 starts from exactly the same pretrained network as G0: the 4 extra input channels get
zero weights at the start, so any difference between G0 and G1 comes from the heat maps.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from drgrading import lesion_features  # noqa: E402

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
G = ROOT / "processed" / "grading"
RUNS = ROOT / "runs_grader"
RESULTS = ROOT / "results"
SIZE = int(os.environ.get("GRADER_SIZE", 512))
EPOCHS = {"ddr": int(os.environ.get("EPOCHS", 20)), "aptos": int(os.environ.get("EPOCHS", 25))}
BATCH = int(os.environ.get("BATCH", 16))
WORKERS = int(os.environ.get("WORKERS", 4))
LR, WD = 3e-4, 1e-4
CLIP = 2.0                             # gradient-norm clipping
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"
CHANNELS_LAST = os.environ.get("CHANNELS_LAST", "0") == "1"      # bench_grader.py: 11x slower on this PC
torch.backends.cudnn.benchmark = os.environ.get("CUDNN_BENCHMARK", "1") == "1"
FMT = torch.channels_last if CHANNELS_LAST else torch.contiguous_format
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
HEAT_SCALE = 3.0                       # heat 0..255 -> 0..3, similar range to normalised RGB


# ----------------------------------------------------------------------------- data
def experiment(exp):
    idx = pd.read_csv(G / "index.csv", keep_default_na=False)
    idx["use"] = idx["use"].astype(str).str.lower() == "true"
    idx = idx[idx.use & (idx.processed.astype(str).str.lower() == "true")]
    if exp == "ddr":
        tr, va = idx[(idx.dataset == "ddr") & (idx.split == "train")], idx[(idx.dataset == "ddr") & (idx.split == "valid")]
        tests = {"DDR-test": idx[(idx.dataset == "ddr") & (idx.split == "test")],
                 "IDRiD": idx[idx.dataset == "idrid"], "APTOS": idx[idx.dataset == "aptos"]}
    elif exp == "aptos":
        ap = idx[idx.dataset == "aptos"].sample(frac=1.0, random_state=0)
        keep = []                                          # 15 % of every grade for picking the best epoch
        for _, grp in ap.groupby("label"):
            keep += list(grp.index[:max(1, round(0.15 * len(grp)))])
        va, tr = ap.loc[keep], ap.drop(keep)
        tests = {"Messidor-2": idx[idx.dataset == "messidor2"]}
        if not len(tests["Messidor-2"]):
            raise SystemExit("Messidor-2 not processed yet: put it in F:/drthesis/MESSIDOR2 and run 06 again.")
    else:
        raise SystemExit("experiment must be 'ddr' or 'aptos'")
    return tr.reset_index(drop=True), va.reset_index(drop=True), {k: v.reset_index(drop=True) for k, v in tests.items()}


class GradingSet(torch.utils.data.Dataset):
    def __init__(self, df, heat, train):
        self.uid, self.y = df.uid.tolist(), df.label.astype(int).tolist()
        self.heat, self.train = heat, train

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, i):
        img = cv2.imread(str(G / "images512" / f"{self.uid[i]}.jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
        if img.shape[0] != SIZE:
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        hm = None
        if self.heat:
            hm = cv2.imread(str(G / "heat512" / f"{self.uid[i]}.png"), cv2.IMREAD_UNCHANGED)
            if hm.shape[0] != SIZE:
                hm = cv2.resize(hm, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if self.train:
            img, hm = augment(img, hm)
        x = (img.astype(np.float32) - MEAN) / STD
        if hm is not None:
            x = np.concatenate([x, hm.astype(np.float32) / 255 * HEAT_SCALE], 2)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), self.y[i]


def augment(img, hm):
    """Same geometric change for image and heat map; colour change for the image only."""
    rng = np.random
    ang, sc = rng.uniform(-180, 180), rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, sc)
    M[:, 2] += rng.uniform(-0.03, 0.03, 2) * SIZE
    flip = rng.rand() < 0.5
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    if hm is not None:
        hm = cv2.warpAffine(hm, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    if flip:
        img = img[:, ::-1]
        hm = hm[:, ::-1] if hm is not None else None
    a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, hm


# ----------------------------------------------------------------------------- model
def build_model(variant):
    import timm
    m = timm.create_model("efficientnet_b0", pretrained=PRETRAINED, num_classes=5)
    if variant == "G1":                                  # 3 -> 7 input channels, extra ones start at 0
        old = m.conv_stem
        new = nn.Conv2d(7, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight
        m.conv_stem = new
    return m


@torch.no_grad()
def predict(model, loader, tta=True):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True).to(memory_format=FMT)
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            p = model(x).float().softmax(1)
            if tta:                                                    # + horizontal-flip TTA (test only)
                p = (p + model(torch.flip(x, [3])).float().softmax(1)) / 2
        if not torch.isfinite(p).all():                                # half precision overflowed: redo in float32
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(x, [3])).float().softmax(1)) / 2
        P.append(p.cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def metrics(prob, y):
    pred = prob.argmax(1)
    ref_true, ref_score = (y >= 2).astype(int), prob[:, 2:].sum(1)       # referable DR = grade >= 2
    ref_pred = (ref_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ref_true, ref_pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)),
            "QWK": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_F1": float(f1_score(y, pred, average="macro", labels=list(range(5)), zero_division=0)),
            "ref_AUC": float(roc_auc_score(ref_true, ref_score)) if 0 < ref_true.sum() < len(y) else float("nan"),
            "ref_sens": float(tp / max(tp + fn, 1)), "ref_spec": float(tn / max(tn + fp, 1))}


def loader(df, heat, train, shuffle=False):
    return torch.utils.data.DataLoader(GradingSet(df, heat, train), batch_size=BATCH, shuffle=shuffle,
                                       num_workers=WORKERS, pin_memory=DEVICE == "cuda",
                                       drop_last=train, persistent_workers=WORKERS > 0)


def save(obj, path):
    """Write to a temporary file first, so a power cut never leaves a half-written checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def finite(model):
    return all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())


def train_cnn(exp, variant, seed, tr, va, tests):
    out = RUNS / f"{exp}_{variant}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test.json").exists():
        print(f"{out.name}: done")
        return json.loads((out / "test.json").read_text())
    torch.manual_seed(seed); np.random.seed(seed)
    heat = variant == "G1"
    epochs = EPOCHS[exp]
    tl, vl = loader(tr, heat, True, shuffle=True), loader(va, heat, False)
    steps = epochs * len(tl)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def fresh():
        m = build_model(variant).to(DEVICE).to(memory_format=FMT)
        o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        sc = torch.optim.lr_scheduler.LambdaLR(
            o, lambda s: min(1.0, (s + 1) / len(tl)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        return m, o, sc, torch.amp.GradScaler(enabled=DEVICE == "cuda")

    def restore(rng):
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)   # optimizer state moves to GPU itself
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if rng:
            torch.set_rng_state(torch.as_tensor(ck["rng_cpu"], dtype=torch.uint8).cpu())
            np.random.set_state(ck["rng_np"])
        return ck["epoch"] + 1, ck["best"], ck["log"]

    model, opt, sched, scaler = fresh()
    start, best, log = 0, -1.0, []
    if (out / "last.pt").exists():
        start, best, log = restore(rng=True)
        print(f"{out.name}: resuming at epoch {start + 1}/{epochs}")
    params = sum(p.numel() for p in model.parameters())
    print(f"{out.name}: {len(tr)} train / {len(va)} valid images | {params / 1e6:.2f} M parameters", flush=True)
    ep, retries = start, 0
    while ep < epochs:
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x = x.to(DEVICE, non_blocking=True).to(memory_format=FMT)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)      # stops one bad batch from blowing up the weights
            scaler.step(opt); scaler.update(); sched.step()
            if torch.isfinite(loss):
                tot += loss.item() * len(y)
            n += len(y)
            b = n // BATCH
            if b % 100 == 0:
                rate = n / (time.time() - t0)
                print(f"    epoch {ep + 1}: {b}/{len(tl)} batches | {rate:.0f} img/s | "
                      f"about {(len(tl) * BATCH - n) / rate / 60:.1f} min left in this epoch", flush=True)
        if not finite(model):                                  # numbers overflowed: redo this epoch
            retries += 1
            if retries > 2:
                raise SystemExit(f"{out.name}: training became unstable 3 times at epoch {ep + 1}. Send this to Claude.")
            back = f"the end of epoch {ep}" if (out / "last.pt").exists() else "the start"
            print(f"  epoch {ep + 1}: numbers overflowed - going back to {back} and redoing it "
                  f"(try {retries} of 2)", flush=True)
            torch.manual_seed(seed + 1000 * retries)           # different batch order this time
            model, opt, sched, scaler = fresh()
            if (out / "last.pt").exists():
                restore(rng=False)
            continue
        retries = 0
        m = metrics(*predict(model, vl, tta=False))
        log.append({"epoch": ep + 1, "train_loss": tot / n, **{f"val_{k}": v for k, v in m.items() if k != "n"}})
        improved = m["QWK"] > best
        if improved:
            best = m["QWK"]
            save(model.state_dict(), out / "best.pt")
        save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": best, "log": log,
              "rng_cpu": torch.get_rng_state(), "rng_np": np.random.get_state()}, out / "last.pt")
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        print(f"  epoch {ep + 1:2d}/{epochs} | loss {tot / n:.3f} | valid QWK {m['QWK']:.3f} acc {m['accuracy']:.3f}"
              f" | {time.time() - t0:.0f} s{'  * best' if improved else ''}", flush=True)
        ep += 1
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEVICE, weights_only=True))
    res = {"params_M": params / 1e6, "best_valid_QWK": best}
    for name, df in tests.items():
        prob, y = predict(model, loader(df, heat, False))
        res[name] = metrics(prob, y)
        np.save(out / f"prob_{name}.npy", prob)
    (out / "test.json").write_text(json.dumps(res, indent=1))
    (out / "last.pt").unlink(missing_ok=True)            # 64 MB each, not needed any more
    return res


# ----------------------------------------------------------------------------- G3: lesion numbers
def features_for(df, det):
    rows = []
    for uid in df.uid:
        d = det.get(uid)
        rows.append(lesion_features(d if d is not None else np.zeros((0, 6), np.float32)))
    return pd.DataFrame(rows)


def train_g3(exp, seed, tr, va, tests, det):
    import xgboost as xgb
    out = RUNS / f"{exp}_G3_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    Xtr, Xva = features_for(tr, det), features_for(va, det)
    clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, objective="multi:softprob", num_class=5,
                            eval_metric="mlogloss", early_stopping_rounds=50, random_state=seed, n_jobs=8)
    clf.fit(Xtr, tr.label.astype(int), eval_set=[(Xva, va.label.astype(int))], verbose=False)
    res = {"params_M": 0.0, "best_valid_QWK": metrics(clf.predict_proba(Xva), va.label.values.astype(int))["QWK"],
           "trees": int(clf.best_iteration + 1)}
    for name, df in tests.items():
        prob = clf.predict_proba(features_for(df, det))
        res[name] = metrics(prob, df.label.values.astype(int))
        np.save(out / f"prob_{name}.npy", prob)
    (out / "test.json").write_text(json.dumps(res, indent=1))
    imp = pd.Series(clf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv")
    print(f"{out.name}: {res['trees']} trees | top features: {', '.join(imp.index[:5])}")
    return res


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    exp, variant = sys.argv[1], sys.argv[2].upper()
    seeds = [int(s) for s in sys.argv[3:]] or [0, 1, 2]
    if variant not in ("G0", "G1", "G3"):
        raise SystemExit("variant must be G0, G1 or G3")
    tr, va, tests = experiment(exp)
    print(f"Experiment {exp} | {variant} | train {len(tr)} | valid {len(va)} | "
          + " | ".join(f"{k} {len(v)}" for k, v in tests.items()), flush=True)
    det = None
    if variant == "G3":
        d = pd.read_csv(G / "detections.csv")
        det = {u: g[["cls", "conf", "x1", "y1", "x2", "y2"]].to_numpy(np.float32) for u, g in d.groupby("uid")}
    rows = []
    for s in seeds:
        r = train_g3(exp, s, tr, va, tests, det) if variant == "G3" else train_cnn(exp, variant, s, tr, va, tests)
        for name in tests:
            rows.append({"seed": s, "test_set": name, **r[name], "params_M": r["params_M"]})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"grader_{exp}_{variant}.csv", index=False)
    cols = ["QWK", "accuracy", "macro_F1", "ref_AUC", "ref_sens", "ref_spec"]
    print("\n" + "=" * 78 + f"\n{exp} {variant}  ({len(seeds)} seeds, mean +- std)\n" + "=" * 78)
    for name, t in df.groupby("test_set", sort=False):
        print(f"{name:11s} n={int(t.n.iloc[0]):5d}  " +
              "  ".join(f"{c} {t[c].mean():.3f}+-{t[c].std():.3f}" for c in cols))
    print(f"Saved: {RESULTS / f'grader_{exp}_{variant}.csv'}")


if __name__ == "__main__":
    main()
