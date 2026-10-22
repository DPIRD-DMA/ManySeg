"""E027: ObjectMetrics.update time (instance / point evaluation) with or without patches."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import patches2  # noqa: E402
from manyseg_bench.evaluation.objects import ObjectMetrics  # noqa: E402


def labels(b, s, k, gen, dev):
    """``k`` rectangular instances per image (ids 1..k), on ``dev``."""
    out = torch.zeros(b, s, s, dtype=torch.long)
    for i in range(b):
        for j in range(1, k + 1):
            y, x = torch.randint(0, s - 24, (2,), generator=gen).tolist()
            h, w = torch.randint(6, 24, (2,), generator=gen).tolist()
            out[i, y:y + h, x:x + w] = j
    return out.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--objects", type=int, default=40)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    applied = patches2.apply(a.patch.split(","))
    gen = torch.Generator().manual_seed(0)
    batches = [(labels(a.batch, a.size, a.objects, gen, "cuda"),
                labels(a.batch, a.size, a.objects, gen, "cuda")) for _ in range(a.reps)]
    om = ObjectMetrics()
    om.update(*batches[0])  # warm-up
    om = ObjectMetrics()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for p, g in batches:
        om.update(p, g)
    dt = time.perf_counter() - t
    print(json.dumps({"bench": "objects", "patches": applied, "batch": a.batch, "size": a.size,
                      "objects": a.objects, "ms_per_batch": 1000 * dt / a.reps,
                      "result": om.compute()}), flush=True)


if __name__ == "__main__":
    main()
