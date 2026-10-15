"""Device-resident batching and augmentation, identical for every provider.

Whole splits of small datasets live on the device (uint8 or float16), so data loading is never
a bottleneck that differs between models. Augmentation is seeded, per sample, and applies one
geometric transform jointly to every input and the mask.

Inputs may be a single array ``(N, H, W, C)`` or a dict of such arrays (multi-input datasets,
all on the same grid). ``stack=True`` concatenates dict inputs along channels in key order, so
single-input libraries can be benchmarked on multi-input data (early fusion).

Labels are per-pixel masks ``(N, H, W)`` (augmented with the images) or image-level labels
``(N,)`` / ``(N, K)`` (class ids, multi-hot or regression targets; never augmented). Float
labels stay float32; integer labels are returned as int64.

``vectors`` (``{name: (N, columns)}`` float arrays, NaN = missing) are per-sample metadata. In
dict mode they are yielded as extra ``(B, columns)`` entries of the input dict, unaugmented
(a random-resized crop moves a chip centre by at most half a chip, which we ignore). In
``stack`` mode they are dropped: stacking is the library-neutral early-fusion baseline for
rasters, and a provider that takes one tensor has no place for a vector. Dropping keeps such
providers runnable as the no-metadata control on identical data; "metadata as extra channels"
is ManySeg's ``channels`` conditioning mode, which encodes fields (cyclic dates, unit-sphere
coordinates) before broadcasting them instead of stacking raw numbers.
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
        device_resident: bool = True,
        vectors: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.multi = isinstance(images, Mapping)
        imgs = dict(images) if self.multi else {"image": images}
        means = dict(mean) if self.multi else {"image": mean}
        stds = dict(std) if self.multi else {"image": std}
        self.names = list(imgs)
        self.device_resident = device_resident
        if device_resident:
            self.images = {k: _to_device(v, device) for k, v in imgs.items()}  # N,H,W,C
        else:  # keep (possibly memory-mapped) arrays on the host; move batches on demand
            self.images = dict(imgs)
        self.channels = {k: v.shape[-1] for k, v in self.images.items()}
        self.mean, self.std = {}, {}
        for k, v in self.images.items():
            unit = 255.0 if v.dtype in (torch.uint8, np.uint8) else 1.0
            self.mean[k] = torch.tensor(means[k], device=device).view(1, -1, 1, 1) * unit
            self.std[k] = torch.tensor(stds[k], device=device).view(1, -1, 1, 1) * unit
        self.masks = _to_device(masks, device) if device_resident else masks
        self.pixel_labels = masks.ndim == 3
        self.float_labels = np.issubdtype(np.asarray(masks[:0]).dtype, np.floating)
        self.n = self.masks.shape[0]
        self.bs = batch_size
        self.train = train
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.out_size = out_size or tuple(next(iter(self.images.values())).shape[1:3])
        self.scale, self.ratio, self.hflip, self.vflip = scale, ratio, hflip, vflip
        self.drop_last = train if drop_last is None else drop_last
        self.stack = stack
        self.vectors = {} if stack or not vectors else {
            k: _to_device(np.asarray(v, dtype=np.float32), device) for k, v in vectors.items()}
        for k, v in self.vectors.items():
            if v.shape[0] != self.n:
                raise ValueError(f"vector input {k!r} has {v.shape[0]} rows for {self.n} samples")

    def _fetch(self, idx: Tensor):
        if self.device_resident:
            idx = idx.to(self.device)
            return {k: self.images[k][idx] for k in self.names}, self._label(self.masks[idx])
        # Sorted reads are faster on memory-mapped files; restore the requested order after.
        sidx, inv = torch.sort(idx)
        npi = sidx.numpy()
        imgs = {k: _to_device(np.asarray(self.images[k][npi]), self.device)[inv]
                for k in self.names}
        y = _to_device(np.asarray(self.masks[npi]), self.device)[inv]
        return imgs, self._label(y)

    def _label(self, y: Tensor) -> Tensor:
        return y.float() if self.float_labels else y.long()

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
        if not self.pixel_labels:  # image-level labels do not move with the pixels
            return x, y
        y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="reflection",
                          align_corners=False)[:, 0].round().long()
        return x, y

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.gen) if self.train else torch.arange(self.n)
        for b in range(len(self)):
            idx = order[b * self.bs:(b + 1) * self.bs]
            imgs, y = self._fetch(idx)
            parts = [
                (imgs[k].permute(0, 3, 1, 2).float() - self.mean[k]) / self.std[k]
                for k in self.names
            ]
            x = torch.nan_to_num(torch.cat(parts, dim=1))
            if self.train:
                x, y = self._augment(x, y)
            if self.multi and not self.stack:
                x = dict(zip(self.names, torch.split(x, [self.channels[k] for k in self.names],
                                                     dim=1)))
            if self.vectors:
                x = x if isinstance(x, dict) else {self.names[0]: x}
                x.update({k: v[idx.to(v.device)] for k, v in self.vectors.items()})
            yield x, y
