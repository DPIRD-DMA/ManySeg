"""Device-resident batching and augmentation, identical for every provider.

Whole splits of small datasets live on the device as uint8, so data loading is never the
bottleneck that differs between models. Augmentation is seeded and per-sample.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class GPULoader:
    def __init__(
        self,
        images: np.ndarray,
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
        drop_last: bool | None = None,
    ) -> None:
        self.images = torch.from_numpy(np.ascontiguousarray(images)).to(device)  # N,H,W,C uint8
        self.masks = torch.from_numpy(np.ascontiguousarray(masks)).to(device)
        self.n = self.images.shape[0]
        self.bs = batch_size
        self.train = train
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1) * 255
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1) * 255
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.out_size = out_size or tuple(self.images.shape[1:3])
        self.scale, self.ratio, self.hflip = scale, ratio, hflip
        self.drop_last = train if drop_last is None else drop_last

    def __len__(self) -> int:
        return self.n // self.bs if self.drop_last else math.ceil(self.n / self.bs)

    def _augment(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Random resized crop + horizontal flip via one affine grid per sample."""
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
        flip = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0) if self.hflip else 1.0
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0] = sw * flip
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = sh
        theta[:, 1, 2] = ty
        theta = theta.to(self.device)
        size = (n, 1, *self.out_size)
        grid = F.affine_grid(theta, size, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="reflection",
                          align_corners=False)[:, 0].round().long()
        return x, y

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.gen) if self.train else torch.arange(self.n)
        for b in range(len(self)):
            idx = order[b * self.bs:(b + 1) * self.bs].to(self.device)
            x = self.images[idx].permute(0, 3, 1, 2).float()
            y = self.masks[idx].long()
            if self.train:
                x, y = self._augment(x, y)
            yield (x - self.mean) / self.std, y
