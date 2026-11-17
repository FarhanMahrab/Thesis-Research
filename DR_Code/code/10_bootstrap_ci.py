"""
10_bootstrap_ci.py  -  95 % confidence intervals and paired significance tests for the graders
==============================================================================================

Run in the Anaconda Prompt from F:\\drthesis (CPU only, a few minutes):
    python code/10_bootstrap_ci.py

For every experiment (ddr, and aptos if it has been run) and every test set it:
  1. resamples the test images with replacement 2,000 times (bootstrap),
  2. scores every model on each resample (QWK and referable-DR AUC, averaged over the 3 seeds),
  3. gives each score a 95 % confidence interval (2.5th - 97.5th percentile),
  4. tests the key differences PAIRED (both models scored on the same resampled images):
        G2 (G0+G3) vs G0  - does adding lesion evidence help the photo model?
        G3 vs G0          - lesion numbers alone vs photo alone
        G1 vs G0          - lesion maps as extra input channels vs photo alone
        G2 (G0+G3) vs G3  - does the photo add anything to the lesion numbers?
     p = two-sided bootstrap p-value; p_holm = Holm-corrected over all tests of that experiment.

Output: F:/drthesis/results/bootstrap_<exp>.csv  (+ the same table printed)
Note: images are resampled one by one. Messidor-2 has two images per patient (one per eye) and
the datasets give no patient ids, so the intervals may be slightly too narrow; state this in the thesis.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CODE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grader", CODE / "07_train_grader.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

B = 2000
SEEDS = [0, 1, 2]
MODELS = {"G0": ["G0"], "G1": ["G1"], "G3": ["G3"], "G2 (G0+G3)": ["G0", "G3"]}
TESTS = [("G2 (G0+G3)", "G0"), ("G3", "G0"), ("G1", "G0"), ("G2 (G0+G3)", "G3")]
W = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2).astype(float)


def qwk(y, p):
    cm = np.bincount(y * 5 + p, minlength=25).reshape(5, 5).astype(float)
    exp = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    den = (W * exp).sum()
    return 1.0 - (W * cm).sum() / den if den > 0 else np.nan


def auc(t, s):
    n1 = t.sum(); n0 = len(t) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(y, preds, refs, idx):
    """Mean over seeds of QWK and referable AUC on the images idx."""
    yy, tt = y[idx], (y[idx] >= 2).astype(int)
    q = np.mean([qwk(yy, p[idx]) for p in preds])
    a = np.mean([auc(tt, r[idx]) for r in refs])
    return q, a


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def run(exp):
    try:
        _, _, tests = g.experiment(exp)
    except SystemExit as e:
        print(f"[{exp}] skipped: {e}")
        return None
    rows, comps = [], []
    rng = np.random.default_rng(0)
    for test, df in tests.items():
        y = df.label.values.astype(int)
        have = {}
        for name, parts in MODELS.items():
            dirs = [[g.RUNS / f"{exp}_{p}_s{s}" / f"prob_{test}.npy" for p in parts] for s in SEEDS]
            if all(f.exists() for d in dirs for f in d):
                probs = [np.mean([np.load(f) for f in d], axis=0) for d in dirs]
                have[name] = ([pr.argmax(1) for pr in probs], [pr[:, 2:].sum(1) for pr in probs])
        if not have:
            continue
        n = len(y)
        boot = {m: np.empty((B, 2)) for m in have}
        for b in range(B):
            idx = rng.integers(0, n, n)
            for m, (pr, rf) in have.items():
                boot[m][b] = scores(y, pr, rf, idx)
        full = {m: scores(y, pr, rf, np.arange(n)) for m, (pr, rf) in have.items()}
        for m in have:
            lo, hi = np.nanpercentile(boot[m], [2.5, 97.5], axis=0)
            rows.append({"experiment": exp, "test_set": test, "n": n, "model": m,
                         "QWK": full[m][0], "QWK_lo": lo[0], "QWK_hi": hi[0],
                         "ref_AUC": full[m][1], "AUC_lo": lo[1], "AUC_hi": hi[1]})
        for a, b_ in TESTS:
            if a in have and b_ in have:
                for k, metric in enumerate(["QWK", "ref_AUC"]):
                    d = boot[a][:, k] - boot[b_][:, k]
                    d = d[~np.isnan(d)]
                    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    comps.append({"experiment": exp, "test_set": test, "comparison": f"{a} vs {b_}",
                                  "metric": metric, "difference": full[a][k] - full[b_][k],
                                  "diff_lo": lo, "diff_hi": hi, "p": p})
        print(f"[{exp}] {test}: done", flush=True)
    if not rows:
        return None
    rows, comps = pd.DataFrame(rows), pd.DataFrame(comps)
    if len(comps):
        comps["p_holm"] = holm(comps.p.values)
        comps["significant"] = np.where(comps.p_holm < 0.05, "yes", "no")
    return rows, comps


def show(rows, comps, exp):
    print("\n" + "=" * 96 + f"\n{exp}: scores with 95 % bootstrap confidence intervals (mean of 3 seeds)\n" + "=" * 96)
    for test, t in rows.groupby("test_set", sort=False):
        print(f"{test} (n = {int(t.n.iloc[0])})")
        for _, r in t.iterrows():
            print(f"  {r.model:11s} QWK {r.QWK:.3f} [{r.QWK_lo:.3f}, {r.QWK_hi:.3f}]   "
                  f"referral AUC {r.ref_AUC:.3f} [{r.AUC_lo:.3f}, {r.AUC_hi:.3f}]")
    print(f"\n{exp}: paired differences (first model minus second), Holm-corrected p")
    for test, t in comps.groupby("test_set", sort=False):
        print(f"{test}")
        for _, r in t.iterrows():
            p = "<0.001" if r.p < 0.001 else f"{r.p:.3f}"
            ph = "<0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
            print(f"  {r.comparison:18s} {r.metric:7s} {r.difference:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]"
                  f"  p {p:>6s}  p_holm {ph:>6s}  significant: {r.significant}")


def main():
    g.RESULTS.mkdir(exist_ok=True)
    for exp in ["ddr", "aptos"]:
        out = run(exp)
        if out is None:
            continue
        rows, comps = out
        show(rows, comps, exp)
        rows.assign(kind="score").to_csv(g.RESULTS / f"bootstrap_{exp}.csv", index=False)
        comps.to_csv(g.RESULTS / f"bootstrap_{exp}_comparisons.csv", index=False)
        print(f"Saved: {g.RESULTS / f'bootstrap_{exp}.csv'} and bootstrap_{exp}_comparisons.csv")


if __name__ == "__main__":
    main()
