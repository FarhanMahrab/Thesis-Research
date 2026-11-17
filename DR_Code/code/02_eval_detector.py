"""
02_eval_detector.py  -  Test a detector (all seeds) on DDR-test and IDRiD-test
==============================================================================

Usage in Jupyter:
    %run F:/drthesis/code/02_eval_detector.py baseline640 640
          (arguments: run-name prefix, image size used for training)

It finds every folder  F:/drthesis/runs/<prefix>_s*/weights/best.pt , tests each
one on both test sets, prints AP50 per lesion for every seed, then mean +- std,
and saves the table to  F:/drthesis/results/<prefix>_test.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "baseline640"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640
CLASSES = ["MA", "HE", "EX", "SE"]

YOLO_DIR = ROOT / "processed" / "yolo_full"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# a second yaml whose "test" points to the IDRiD test images
ddr_yaml = YOLO_DIR / "data.yaml"
idrid_yaml = YOLO_DIR / "data_idrid.yaml"
idrid_yaml.write_text(ddr_yaml.read_text().replace("test: images/test_ddr", "test: images/test_idrid"))
TEST_SETS = {"DDR-test": ddr_yaml, "IDRiD-test": idrid_yaml}

runs = sorted((ROOT / "runs").glob(f"{PREFIX}_s*/weights/best.pt"))
if not runs:
    raise SystemExit(f"No runs found matching {ROOT / 'runs'}/{PREFIX}_s*/weights/best.pt")
print(f"Found {len(runs)} runs: {[r.parent.parent.name for r in runs]}  |  imgsz={IMGSZ}\n")

rows = []
for best in runs:
    seed = best.parent.parent.name
    model = YOLO(str(best))
    n_params = sum(p.numel() for p in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, IMGSZ)          # GFLOPs for one IMGSZ x IMGSZ input
    except Exception:
        gflops = float("nan")
    for test_name, yaml in TEST_SETS.items():
        m = model.val(data=str(yaml), split="test", imgsz=IMGSZ, batch=8, max_det=1000,
                      workers=0, plots=False, verbose=False,
                      project=str(ROOT / "runs" / "eval"), name=f"{seed}_{test_name}", exist_ok=True)
        idx = list(getattr(m.box, "ap_class_index", getattr(m, "ap_class_index", [])))
        ap50 = {c: (float(m.box.ap50[idx.index(i)]) if i in idx else float("nan"))
                for i, c in enumerate(CLASSES)}
        rows.append({"run": seed, "test_set": test_name, **{f"AP50_{c}": v for c, v in ap50.items()},
                     "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                     "precision": float(m.box.mp), "recall": float(m.box.mr),
                     "params_M": n_params / 1e6, "GFLOPs": gflops})

df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"{PREFIX}_test.csv", index=False)

pd.set_option("display.width", 200, "display.precision", 3)
cols = [f"AP50_{c}" for c in CLASSES] + ["mAP50", "mAP50-95", "recall"]
for test_name in TEST_SETS:
    t = df[df.test_set == test_name]
    print("=" * 70 + f"\n{PREFIX}  on  {test_name}\n" + "=" * 70)
    print(t[["run"] + cols].to_string(index=False))
    mean, std = t[cols].mean(), t[cols].std()
    print("mean +- std: " + "  ".join(f"{c.replace('AP50_', '')} {mean[c]:.3f}+-{std[c]:.3f}"
                                      for c in cols) + "\n")
print(f"Model size: {df.params_M.iloc[0]:.2f} M parameters, {df.GFLOPs.iloc[0]:.1f} GFLOPs")
print(f"Saved: {RESULTS / f'{PREFIX}_test.csv'}")
