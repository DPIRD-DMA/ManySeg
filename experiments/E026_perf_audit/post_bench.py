"""E026: instance post-processing / target building.

* ``connected_components``: the current max-label propagation vs the same plus pointer
  jumping (each label also takes the label of the pixel it names; labels stay inside their
  component, so the fixed point, each component's largest pixel id, is unchanged).
* ``edt``: the torch brute-force exact EDT vs ``scipy.ndimage.distance_transform_edt``.

    uv run python experiments/E026_perf_audit/post_bench.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from manyseg.instances.postprocess import _neighbour_max, connected_components
from manyseg.instances.targets import edt


@torch.no_grad()
def cc_jump(mask, check_every: int = 8):
    b, h, w = mask.shape
    lab = torch.arange(1, h * w + 1, device=mask.device).view(1, h, w).expand(b, h, w)
    lab = torch.where(mask, lab, 0)
    it = 0
    while True:
        prev = lab
        for _ in range(check_every):
            lab = torch.where(mask, torch.maximum(lab, _neighbour_max(lab)), 0)
            flat = lab.reshape(b, -1)
            jumped = flat.gather(1, (flat - 1).clamp(min=0)).view(b, h, w)
            lab = torch.where(mask, torch.maximum(lab, jumped), 0)
        it += check_every
        if torch.equal(lab, prev) or it > h * w:
            return lab


def masks(kind: str, b: int, s: int, gen):
    if kind == "snake":  # one long serpentine component per image (worst case for propagation)
        m = torch.zeros(b, s, s, dtype=torch.bool)
        m[:, ::4, :] = True
        for r in range(0, s - 4, 4):
            c = s - 1 if (r // 4) % 2 == 0 else 0
            m[:, r:r + 5, c] = True
        return m
    if kind == "blobs":  # smoothed noise, thresholded: many irregular components
        n = torch.randn(b, 1, s, s, generator=gen)
        return (F.avg_pool2d(n, 9, 1, 4)[:, 0] > 0.05)
    if kind == "fields":  # field parcels: rectangles separated by 2-px boundaries
        m = torch.ones(b, s, s, dtype=torch.bool)
        for i in range(b):
            for cut in torch.randint(0, s, (12,), generator=gen).tolist():
                m[i, cut:cut + 2, :] = False
            for cut in torch.randint(0, s, (12,), generator=gen).tolist():
                m[i, :, cut:cut + 2] = False
        return m
    raise ValueError(kind)


def timed(fn, *args, dev="cpu", reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*args)
        if dev == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    gen = torch.Generator().manual_seed(0)
    for kind, b, s in (("snake", 2, 128), ("snake", 2, 256), ("blobs", 8, 256), ("blobs", 8, 512),
                       ("fields", 8, 256), ("fields", 8, 512)):
        m = masks(kind, b, s, gen).to(a.device)
        ref, t_ref = timed(connected_components, m, dev=a.device, reps=1 if kind == "snake" else 3)
        new, t_new = timed(cc_jump, m, dev=a.device)
        print(json.dumps({"bench": "connected_components", "device": a.device, "mask": kind,
                          "batch": b, "size": s, "current_s": t_ref, "jump_s": t_new,
                          "speedup": t_ref / t_new, "identical": bool(torch.equal(ref, new))}),
              flush=True)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        distance_transform_edt = None
    for s in (256, 512):
        f = masks("blobs", 4, s, gen)
        g = f.to(a.device)
        ref, t_ref = timed(edt, g, dev=a.device)
        rec = {"bench": "edt", "device": a.device, "batch": 4, "size": s, "torch_s": t_ref}
        if distance_transform_edt is not None:
            def sp(x):
                return np.stack([distance_transform_edt(~xi) for xi in x.numpy()])
            out, t_sp = timed(sp, f)
            rec.update(scipy_cpu_s=t_sp, speedup=t_ref / t_sp,
                       max_abs_diff=float((torch.from_numpy(out).float() - ref.cpu()).abs().max()))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
