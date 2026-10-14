"""Device-resident batching and augmentation, identical for every provider.

Whole splits of small datasets live on the device (uint8 or float16), so data loading is never
a bottleneck that differs between models. Augmentation is seeded, per sample, and applies one
geometric transform jointly to every input and the mask.

Inputs may be a single array ``(N, H, W, C)`` or a dict of such arrays (multi-input datasets,
all on the same grid). ``stack=True`` concatenates dict inputs along channels in key order, so
single-input libraries can be benchmarked on multi-input data (early fusion).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _to_device(a: np.ndarray, device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


class GPULoader:
    def __init__(
        self,
        images,
        masks: np.ndarray,
        batch_size: int,
        mean, std,
        device: torch.device,
        train: bool,
        seed: int = 0,
        out_size: tuple[int, int] | None = None,
        scale: tuple[float, float] = (0.5, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        hflip: bool = True,
        vflip: bool = False,
        drop_last: bool | None = None,
        stack: bool = True,
    ) -> None:
        self.multi = isinstance(images, Mapping)
        imgs = dict(images) if self.multi else {"image": images}
        means = dict(mean) if self.multi else {"image": mean}
        stds = dict(std) if self.multi else {"image": std}
        self.names = list(imgs)
        self.images = {k: _to_device(v, device) for k, v in imgs.items()}  # N,H,W,C
        self.channels = {k: v.shape[-1] for k, v in self.images.items()}
        self.mean, self.std = {}, {}
        for k, v in self.images.items():
            unit = 255.0 if v.dtype == torch.uint8 else 1.0
            self.mean[k] = torch.tensor(means[k], device=device).view(1, -1, 1, 1) * unit
            self.std[k] = torch.tensor(stds[k], device=device).view(1, -1, 1, 1) * unit
        self.masks = _to_device(masks, device)
        self.n = self.masks.shape[0]
        self.bs = batch_size
        self.train = train
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.out_size = out_size or tuple(self.masks.shape[1:3])
        self.scale, self.ratio, self.hflip, self.vflip = scale, ratio, hflip, vflip
        self.drop_last = train if drop_last is None else drop_last
        self.stack = stack

    def __len__(self) -> int:
        return self.n // self.bs if self.drop_last else math.ceil(self.n / self.bs)

    def _augment(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Random resized crop + flips via one affine grid per sample."""
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
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0] = sw * fx
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = sh * fy
        theta[:, 1, 2] = ty
        theta = theta.to(self.device)
        grid = F.affine_grid(theta, (n, 1, *self.out_size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="reflection",
                          align_corners=False)[:, 0].round().long()
        return x, y

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.gen) if self.train else torch.arange(self.n)
        for b in range(len(self)):
            idx = order[b * self.bs:(b + 1) * self.bs].to(self.device)
            parts = [
                (self.images[k][idx].permute(0, 3, 1, 2).float() - self.mean[k]) / self.std[k]
                for k in self.names
            ]
            x = torch.nan_to_num(torch.cat(parts, dim=1))
            y = self.masks[idx].long()
            if self.train:
                x, y = self._augment(x, y)
            if self.multi and not self.stack:
                x = dict(zip(self.names, torch.split(x, [self.channels[k] for k in self.names],
                                                     dim=1)))
            yield x, y
