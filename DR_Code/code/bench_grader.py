"""
bench_grader.py  -  2-minute speed test for the grader (EfficientNet-B0, 512 px) on this PC
==========================================================================================
Run in the Anaconda Prompt from F:\\drthesis (nothing else should be training):
    python code/bench_grader.py

Part A times pure GPU training speed for 4 settings (memory layout x cuDNN autotuning).
Part B times how fast the real images + heat maps can be loaded (CPU side).
Send the printed table to Claude.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

SIZE, BATCH, STEPS = int(os.environ.get("GRADER_SIZE", 512)), int(os.environ.get("BATCH", 16)), 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_speed(channels_last, benchmark):
    import timm
    torch.backends.cudnn.benchmark = benchmark
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5).to(DEVICE)
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    m = m.to(memory_format=fmt)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    x = torch.randn(BATCH, 3, SIZE, SIZE, device=DEVICE).to(memory_format=fmt)
    y = torch.randint(0, 5, (BATCH,), device=DEVICE)
    lossf = nn.CrossEntropyLoss()

    def step():
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            loss = lossf(m(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    for _ in range(5):                       # warm-up (and autotuning, if on)
        step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(STEPS):
        step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return BATCH * STEPS / (time.time() - t)


def loader_speed(heat, workers):
    import importlib.util
    spec = importlib.util.spec_from_file_location("g", CODE / "07_train_grader.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    tr, _, _ = g.experiment("ddr")
    dl = torch.utils.data.DataLoader(g.GradingSet(tr.iloc[:BATCH * 40], heat, True), batch_size=BATCH,
                                     num_workers=workers, shuffle=True)
    it = iter(dl)
    next(it)                                 # start-up of the workers is not counted
    t, n = time.time(), 0
    for x, _ in it:
        n += len(x)
    return n / (time.time() - t)


def main():
    print(f"device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'} | torch {torch.__version__}"
          f" | cuDNN {torch.backends.cudnn.version()} | batch {BATCH} | {SIZE} px\n")
    print("A. GPU training speed (images per second, higher is better)")
    res = {}
    for cl in (True, False):
        for bm in (False, True):
            r = gpu_speed(cl, bm)
            res[(cl, bm)] = r
            print(f"   channels_last={str(cl):5s}  cudnn.benchmark={str(bm):5s}  ->  {r:7.1f} img/s", flush=True)
    best = max(res, key=res.get)
    print(f"   fastest: channels_last={best[0]}, cudnn.benchmark={best[1]}"
          f"  (about {5985 / res[best]:.0f} s of GPU time per training epoch)\n")
    print("B. Image loading speed with 4 workers (images per second)")
    for heat in (False, True):
        print(f"   {'image + heat map' if heat else 'image only      '}  ->  {loader_speed(heat, 4):7.1f} img/s",
              flush=True)


if __name__ == "__main__":
    main()
