"""
05_train.py  -  Train one detector variant for several seeds (auto-resume after power cuts)
==========================================================================================

Run in the Anaconda Prompt (NOT in Jupyter), from F:\\drthesis :

    python code/05_train.py p2_simam_1024 yolov8n-p2-simam.yaml
    python code/05_train.py p2_cbam_1024  yolov8n-p2-cbam.yaml

    arguments: run-name prefix, model file in F:/drthesis/code/models/, optional seeds (default 0 1 2)
               e.g.  python code/05_train.py p2_simam_1024 yolov8n-p2-simam.yaml 2

Settings are exactly those of the P2 runs (tiles_1024_d2048, imgsz 1024, batch 8, 100 epochs,
patience 20, max_det 1000, pretrained yolov8n.pt), so results are directly comparable.

After a power cut or Ctrl+C, run the SAME command again:
  - a finished seed is skipped,
  - an interrupted seed continues from its last.pt (same epoch, same early-stopping counter),
  - a seed that stopped before finishing its first epoch is started again from scratch.
"""
import os
import shutil
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import drmodules  # noqa: E402  (registers SimAM and CBAM before Ultralytics builds a model)
from ultralytics import YOLO  # noqa: E402

ROOT = Path(os.environ.get("DRTHESIS_ROOT", "F:/drthesis"))
TILES = ROOT / "processed" / os.environ.get("TILES_NAME", "tiles_1024_d2048")
PRETRAINED = ROOT / "yolov8n.pt"
SETTINGS = dict(
    data=str(TILES / "data.yaml"),
    imgsz=int(os.environ.get("IMGSZ", 1024)),
    epochs=int(os.environ.get("EPOCHS", 100)),
    patience=20,
    batch=int(os.environ.get("BATCH", 8)),
    workers=int(os.environ.get("WORKERS", 4)),
    max_det=1000,
    project=str(ROOT / "runs"),
    pretrained=str(PRETRAINED) if PRETRAINED.exists() else "yolov8n.pt",
)


def state(run_dir):
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        return "new"
    try:
        ck = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as e:
        raise SystemExit(f"{last} could not be read ({e}).\n"
                         f"It was probably damaged by a power cut while saving. Tell Claude before deleting anything.")
    if ck.get("optimizer") is None or ck.get("epoch", -1) < 0:
        return "done"
    return "resume"


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    prefix, model_file = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [0, 1, 2]
    cfg = Path(model_file)
    if not cfg.exists():
        cfg = CODE / "models" / model_file
    if not cfg.exists():
        raise SystemExit(f"Model file not found: {model_file} (looked in {CODE / 'models'})")

    for seed in seeds:
        name = f"{prefix}_s{seed}"
        run_dir = ROOT / "runs" / name
        st = state(run_dir)
        print(f"\n{'=' * 70}\n{name}: {st}\n{'=' * 70}", flush=True)
        if st == "done":
            continue
        if st == "resume":
            YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True, workers=SETTINGS["workers"])
            continue
        if run_dir.exists():                    # crashed before the first epoch: nothing worth keeping
            shutil.rmtree(run_dir)
        YOLO(str(cfg)).train(**SETTINGS, seed=seed, name=name, exist_ok=False)
    print("\nAll seeds finished. Evaluate with:\n"
          f"  %run F:/drthesis/code/04_sliced_eval.py {prefix} tiled {SETTINGS['imgsz']}")


if __name__ == "__main__":      # required on Windows (data-loader workers start new processes)
    main()
