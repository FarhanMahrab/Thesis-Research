"""
08_fusion.py  -  G2: late fusion of the CNN grader and the lesion-count model (no training)
=========================================================================================

Run in the Anaconda Prompt from F:\\drthesis, after 07 has finished G0, G1 and G3:
    python code/08_fusion.py            (experiment ddr)
    python code/08_fusion.py aptos      (APTOS -> Messidor-2, later)

For every seed k it averages the saved grade probabilities of two models, e.g.
    G2 = (G1_seed_k + G3_seed_k) / 2
on every test set, and scores the result exactly like 07 does. Equal weights are used, so
nothing is fitted and no test information is used. Prints every model and fusion side by side,
plus the per-seed numbers, and saves results/grader_<exp>_fusion.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grader", CODE / "07_train_grader.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

COMBOS = {"G0": ["G0"], "G1": ["G1"], "G3": ["G3"],
          "G2 = G0 + G3": ["G0", "G3"], "G2 = G1 + G3": ["G1", "G3"]}
COLS = ["QWK", "accuracy", "macro_F1", "ref_AUC", "ref_sens", "ref_spec"]


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "ddr"
    _, _, tests = g.experiment(exp)
    rows = []
    for name, parts in COMBOS.items():
        for seed in range(10):
            dirs = [g.RUNS / f"{exp}_{p}_s{seed}" for p in parts]
            if not all((d / "test.json").exists() for d in dirs):
                continue
            for test, df in tests.items():
                prob = np.mean([np.load(d / f"prob_{test}.npy") for d in dirs], axis=0)
                rows.append({"model": name, "seed": seed, "test_set": test,
                             **g.metrics(prob, df.label.values.astype(int))})
    if not rows:
        raise SystemExit("No finished runs found in " + str(g.RUNS))
    r = pd.DataFrame(rows)
    g.RESULTS.mkdir(exist_ok=True)
    r.to_csv(g.RESULTS / f"grader_{exp}_fusion.csv", index=False)
    pd.set_option("display.width", 220)
    for test in tests:
        t = r[r.test_set == test]
        print("\n" + "=" * 100 + f"\n{test}  (n = {int(t.n.iloc[0])}, mean +- std over seeds)\n" + "=" * 100)
        for name in COMBOS:
            s = t[t.model == name]
            if len(s):
                print(f"  {name:14s} ({len(s)} seeds)  " +
                      "  ".join(f"{c} {s[c].mean():.3f}+-{s[c].std():.3f}" for c in COLS))
    print("\nPer seed (QWK / referral AUC):")
    piv = r.pivot_table(index=["test_set", "seed"], columns="model", values=["QWK", "ref_AUC"])
    print(piv.round(3).to_string())
    print(f"\nSaved: {g.RESULTS / f'grader_{exp}_fusion.csv'}")


if __name__ == "__main__":
    main()
