"""E026: out-of-RAM datasets (CloudSEN12+): synchronous fetch vs a one-batch prefetch thread.

The bench's ``GPULoader(device_resident=False)`` reads each batch from the memory-mapped cache
(sorted indices), copies it to the GPU from pageable memory and only then runs the step. The
``prefetch`` variant reads batch b+1 into pinned memory in a background thread while batch b
trains, then copies it without blocking. Both run the same real training step (U-Net ResNet-34,
13-band S2, 512², batch 16, bf16) on disjoint random index sets. The files' page cache is
dropped (``posix_fadvise(DONTNEED)``, this file only) before each variant, so reads are cold.

    uv run python experiments/E026_perf_audit/fetch_bench.py --variant seq|prefetch|fetch_only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402

CACHE = Path(os.environ.get("MANYSEG_BENCH_DATA", "data")) \
    / "cache" / "cloudsen12plus_high"


def drop_cache(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["seq", "prefetch", "fetch_only"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = [CACHE / "train_images.s2.npy", CACHE / "train_masks.npy"]
    for f in files:
        drop_cache(f)
    imgs = np.load(files[0], mmap_mode="r")
    masks = np.load(files[1], mmap_mode="r")
    rng = np.random.default_rng(1000 + a.seed + {"seq": 0, "prefetch": 1, "fetch_only": 2}[
        a.variant] * 100)
    n = a.warmup + a.steps
    batches = rng.choice(imgs.shape[0], size=(n, a.batch), replace=False)
    dev = torch.device("cuda")
    mean = torch.full((1, 13, 1, 1), 0.2, device=dev)
    std = torch.full((1, 13, 1, 1), 0.1, device=dev)

    def read(idx, pinned: bool):
        s = np.sort(idx)
        x = torch.from_numpy(np.ascontiguousarray(imgs[s]))
        y = torch.from_numpy(np.ascontiguousarray(masks[s]))
        if pinned:
            x, y = x.pin_memory(), y.pin_memory()
        return x, y

    def to_dev(x, y, non_blocking):
        x = x.to(dev, non_blocking=non_blocking).permute(0, 3, 1, 2).float()
        return (x - mean) / std, y.to(dev, non_blocking=non_blocking).long().clamp_(max=3)  # 255 = ignore

    torch.backends.cudnn.benchmark = True
    model = models.build("unet_r34", 13, 4).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(x, y):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        F.cross_entropy(out.float(), y).backward()
        opt.step()

    fetch_s, t0 = 0.0, None
    if a.variant == "prefetch":
        pool = ThreadPoolExecutor(1)
        fut = pool.submit(read, batches[0], True)
    for i in range(n):
        if i == a.warmup:
            torch.cuda.synchronize()
            t0, fetch_s = time.perf_counter(), 0.0
        tf = time.perf_counter()
        if a.variant == "prefetch":
            x, y = fut.result()
            if i + 1 < n:
                fut = pool.submit(read, batches[i + 1], True)
            x, y = to_dev(x, y, True)
        else:
            x, y = to_dev(*read(batches[i], False), False)
        fetch_s += time.perf_counter() - tf  # time the loop waited for data
        if a.variant != "fetch_only":
            step(x, y)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mb = a.steps * a.batch * (imgs[0].nbytes + masks[0].nbytes) / 2**20
    rec = {"variant": a.variant, "batch": a.batch, "steps": a.steps,
           "ms_per_step": 1000 * dt / a.steps, "wait_for_data_ms_per_step":
           1000 * fetch_s / a.steps, "read_mb_per_s": mb / dt, "images_per_s":
           a.steps * a.batch / dt,
           "gpu_after": subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader"], capture_output=True,
                                       text=True).stdout.strip()}
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
