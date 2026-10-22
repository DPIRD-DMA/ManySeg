"""E026: training-loop overheads (host syncs, gradient scaling, optimiser, loader copies).

Times a copy of ``manyseg_bench.training.trainer.train``'s inner loop on a synthetic
device-resident dataset fed through the bench's real ``GPULoader`` (with augmentation), with
the variants switched on one at a time:

* ``lossdev``: only the loss sum moves to the GPU; the ``isfinite`` skip (one sync) stays.
* ``nosync``: loss sum and non-finite count stay on the GPU (read once at the end); no
  per-micro-batch ``isfinite`` branch. (Timing only: a non-finite batch would not be skipped.)
* ``foreach``: ``torch._foreach_div_`` instead of one ``div_`` per parameter.
* ``fused``: ``AdamW(fused=True)``.
* ``loader``: the epoch order lives on the GPU (no per-batch index copy), ``theta`` is copied
  from pinned memory without blocking, normalisation is in place and a single input skips
  ``torch.cat``.

    uv run python experiments/E026_perf_audit/train_bench.py --model unet_r34 --variant base
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
from manyseg_bench.datasets.gpu_loader import GPULoader  # noqa: E402


class FastLoader(GPULoader):
    """GPULoader with the ``loader`` variant's changes (same values, same RNG stream)."""

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.gen) if self.train else torch.arange(self.n)
        order_dev = order.to(self.device) if self.device_resident else None
        for b in range(len(self)):
            sl = slice(b * self.bs, (b + 1) * self.bs)
            idx = order[sl]
            if self.device_resident:
                di = order_dev[sl]
                imgs = {k: self.images[k][di] for k in self.names}
                y = self._label(self.masks[di])
            else:
                imgs, y = self._fetch(idx)
            parts = []
            for k in self.names:
                if k in self.times:
                    v = imgs[k].permute(0, 1, 4, 2, 3).float().sub_(self.mean[k]).div_(self.std[k])
                    parts.append(v.flatten(1, 2))
                else:
                    parts.append(imgs[k].permute(0, 3, 1, 2).float().sub_(self.mean[k])
                                 .div_(self.std[k]))
            x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            x = x.nan_to_num_()
            if self.train:
                x, y = self._augment(x, y)
            if (self.multi or self.times) and not self.stack:
                widths = [self.channels[k] * self.times.get(k, 1) for k in self.names]
                x = dict(zip(self.names, torch.split(x, widths, dim=1)))
                for k, t in self.times.items():
                    x[k] = x[k].unflatten(1, (t, self.channels[k]))
            yield x, y

    def _augment(self, x, y):
        # as GPULoader._augment, with theta copied from pinned memory without blocking
        n = x.shape[0]
        g = self.gen
        area = torch.empty(n).uniform_(*self.scale, generator=g)
        logr = torch.empty(n).uniform_(math.log(self.ratio[0]), math.log(self.ratio[1]),
                                       generator=g)
        r = torch.exp(logr)
        sw = torch.sqrt(area * r).clamp(max=1.0)
        sh = torch.sqrt(area / r).clamp(max=1.0)
        tx = (torch.rand(n, generator=g) * 2 - 1) * (1 - sw)
        ty = (torch.rand(n, generator=g) * 2 - 1) * (1 - sh)
        fx = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0) if self.hflip else 1.0
        fy = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0) if self.vflip else 1.0
        theta = torch.zeros(n, 2, 3, pin_memory=True)
        theta[:, 0, 0] = sw * fx
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = sh * fy
        theta[:, 1, 2] = ty
        theta = theta.to(self.device, non_blocking=True)
        grid = F.affine_grid(theta, (n, 1, *self.out_size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        if not self.pixel_labels:
            return x, y
        stack = y.ndim == 4
        y = F.grid_sample((y if stack else y[:, None]).float(), grid, mode="nearest",
                          padding_mode="reflection", align_corners=False).round().long()
        return x, (y if stack else y[:, 0])


def gpu_state() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="base",
                    help="base, or a comma list of nosync,foreach,fused,loader")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--n", type=int, default=512, help="synthetic samples on the GPU")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    v = set(a.variant.split(",")) - {"base"}
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (a.n, a.size, a.size, a.in_ch), dtype=np.uint8)
    masks = rng.integers(0, 4, (a.n, a.size, a.size), dtype=np.uint8)
    Loader = FastLoader if "loader" in v else GPULoader
    loader = Loader(imgs, masks, a.batch, [0.45] * a.in_ch, [0.25] * a.in_ch, dev, train=True)
    del imgs, masks
    model = models.build(a.model, a.in_ch, 4).to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, fused=True if "fused" in v else None)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16: disabled, as the trainer

    loss_dev = torch.zeros((), device=dev)
    nonfinite_dev = torch.zeros((), device=dev, dtype=torch.long)
    loss_sum, nonfinite, nb = 0.0, 0, 0
    done, t0 = 0, None
    total = a.warmup + a.steps
    while done < total:
        for x, y in loader:
            if done == a.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            loss = F.cross_entropy(out.float(), y)
            n = x.shape[0]
            if "nosync" in v:
                scaler.scale(loss * n).backward()
                loss_dev += loss.detach() * n
                nonfinite_dev += (~torch.isfinite(loss)).long()
                nb += n
                group_n = n
            else:
                if not torch.isfinite(loss):  # the one remaining sync per step
                    nonfinite += 1
                    done += 1
                    continue
                scaler.scale(loss * n).backward()
                if "lossdev" in v:  # loss sum on the GPU, read at the end
                    loss_dev += loss.detach() * n
                else:
                    loss_sum += float(loss.detach()) * n
                nb += n
                group_n = n
            grads = [p.grad for p in params if p.grad is not None]
            if "foreach" in v:
                torch._foreach_div_(grads, group_n)
            else:
                for g in grads:
                    g.div_(group_n)
            scaler.step(opt)
            scaler.update()
            done += 1
            if done >= total:
                break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if "nosync" in v:
        loss_sum, nonfinite = float(loss_dev), int(nonfinite_dev)
    elif "lossdev" in v:
        loss_sum = float(loss_dev)
    rec = {"model": a.model, "variant": sorted(v) or ["base"], "size": a.size, "batch": a.batch,
           "in_ch": a.in_ch, "steps": a.steps, "ms_per_step": 1000 * dt / a.steps,
           "images_per_s": a.steps * a.batch / dt,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "mean_loss": loss_sum / max(1, nb), "nonfinite": nonfinite,
           "gpu_before": a.tag, "gpu_after": gpu_state(), "torch": torch.__version__}
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
