"""E027: host RAM of a device-resident dataset (Sen1Floods11) as the runner loads it.

Loads train + val like ``experiments/runner.py`` (``mmap`` as given), builds the GPU loaders,
and reports process RSS after loading, after building the loaders, and after dropping the
host arrays (``--drop``)."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from manyseg_bench.datasets.base import load_split
from manyseg_bench.datasets.gpu_loader import GPULoader


def rss_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--dataset", default="sen1floods11")
    a = ap.parse_args()
    cache = Path(os.environ.get("MANYSEG_BENCH_DATA",
                                "$MANYSEG_BENCH_DATA")) / "cache" / a.dataset
    torch.zeros(1, device="cuda")  # CUDA context first, so its RSS is in every reading
    rec = {"dataset": a.dataset, "mmap": a.mmap, "rss_start_mb": rss_mb()}
    tr, va = load_split(cache, "train", mmap=a.mmap), load_split(cache, "val", mmap=a.mmap)
    rec["rss_loaded_mb"] = rss_mb()
    imgs = tr.images
    c = (next(iter(imgs.values())) if isinstance(imgs, dict) else imgs).shape[-1]
    mean = {k: [0.0] * v.shape[-1] for k, v in imgs.items()} if isinstance(imgs, dict) \
        else [0.0] * c
    std = {k: [1.0] * v.shape[-1] for k, v in imgs.items()} if isinstance(imgs, dict) \
        else [1.0] * c
    dev = torch.device("cuda")
    ltr = GPULoader(tr.images, tr.masks, 16, mean, std, dev, train=True)
    lva = GPULoader(va.images, va.masks, 16, mean, std, dev, train=False)
    rec["rss_loaders_mb"] = rss_mb()
    del tr, va, imgs
    gc.collect()
    rec["rss_after_drop_mb"] = rss_mb()
    rec["batches"] = len(ltr) + len(lva)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
