"""Device-resident batching and augmentation, identical for every provider.

Whole splits of small datasets live on the device (uint8 or float16), so data loading is never
a bottleneck that differs between models. Augmentation is seeded, per sample, and applies one
geometric transform jointly to every input and the mask.

Inputs may be a single array ``(N, H, W, C)`` or a dict of such arrays (multi-input datasets,
all on the same grid). ``stack=True`` concatenates dict inputs along channels in key order, so
single-input libraries can be benchmarked on multi-input data (early fusion).

Time series are ``(N, T, H, W, C)`` arrays: every date gets the same geometric transform. In
dict mode they are yielded as ``(B, T, C, H, W)``; in stack mode their dates are flattened into
``T * C`` channels (date-major), the "dates as channels" baseline. A series' optional mask /
dates come in through ``vectors`` (e.g. ``"s2_dates"``, ``(N, T)``), like metadata.

Labels are per-pixel masks ``(N, H, W)`` (augmented with the images) or image-level labels
``(N,)`` / ``(N, K)`` (class ids, multi-hot or regression targets; never augmented). Float
labels stay float32; integer labels are returned as int64. A dict of per-pixel label arrays
``{name: (N, H, W) or (N, K, H, W)}`` (targets encoded from geometry, e.g. a corner heatmap and
a direction field) is yielded as a dict, every entry moved with the image (bool entries stay
bool).

``dihedral=True`` replaces the random resized crop and flips by one of the 8 exact dihedral
transforms per sample (flips, quarter turns, transposes; uniform; square tiles), which move
whole pixels: targets whose widths and sigmas are in pixels stay what the transformed geometry
would encode. ``directions`` names label entries holding ``(cos 2t, sin 2t)`` channel pairs
(an orientation ``t`` mod 180 degrees); their values are transformed too (a quarter turn
negates both, a horizontal or vertical flip negates ``sin 2t``, a transpose ``cos 2t``; see
``manyseg.vectors.augment``). Directions need ``dihedral``: a resized crop with a changed
aspect ratio would bend them and rescale the encoded widths. They must be stored as floats
(float16 or float32): an integer entry (a quantised cache) raises, as the loss would see it
unscaled.

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


def _dict_label(y: Tensor) -> Tensor:
    """A dict label entry as yielded: float32, bool or int64."""
    if y.dtype == torch.bool:
        return y
    return y.float() if y.is_floating_point() else y.long()


def _warp_label(y: Tensor, grid: Tensor) -> Tensor:
    """Nearest-neighbour resampling of a label entry ``(B, H, W)`` / ``(B, K, H, W)``, its dtype
    kept (float values are not rounded)."""
    stack = y.ndim == 4
    out = F.grid_sample((y if stack else y[:, None]).float(), grid, mode="nearest",
                        padding_mode="reflection", align_corners=False)
    out = out if stack else out[:, 0]
    if y.is_floating_point():
        return out
    return out > 0.5 if y.dtype == torch.bool else out.round().to(y.dtype)


def _dihedral(t: Tensor, d: int, direction: bool = False) -> Tensor:
    """Transform ``d`` of ``(B, ..., H, W)``; ``direction``: ``(B, 2k, H, W)`` (cos 2t, sin 2t)
    pairs whose values change with it (same rule as ``manyseg.vectors.dihedral``)."""
    k, flip = d % 4, d >= 4
    out = torch.rot90(t, k, dims=(-2, -1)) if k else t
    out = torch.flip(out, dims=(-1,)) if flip else out
    if direction:
        sign = torch.tensor([-1.0 if k % 2 else 1.0, 1.0 if (k % 2) == flip else -1.0],
                            device=t.device, dtype=t.dtype)
        out = out * sign.repeat(t.shape[-3] // 2).view(-1, 1, 1)
    return out


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
        dihedral: bool = False,
        directions: tuple[str, ...] | list[str] = (),
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
        self.times = {k: v.shape[1] for k, v in self.images.items() if v.ndim == 5}
        self.mean, self.std = {}, {}
        for k, v in self.images.items():
            unit = 255.0 if v.dtype in (torch.uint8, np.uint8) else 1.0
            shape = (1, 1, -1, 1, 1) if k in self.times else (1, -1, 1, 1)
            self.mean[k] = torch.tensor(means[k], device=device).view(shape) * unit
            self.std[k] = torch.tensor(stds[k], device=device).view(shape) * unit
        self.dict_labels = isinstance(masks, Mapping)
        if self.dict_labels:
            self.masks = {k: _to_device(v, device) if device_resident else v
                          for k, v in masks.items()}
            ns = {v.shape[0] for v in masks.values()}
            if len(ns) != 1 or any(v.ndim not in (3, 4) for v in masks.values()):
                raise ValueError("dict labels must be (N, H, W) / (N, K, H, W) arrays of one N")
            self.pixel_labels, self.float_labels = True, False
            self.n = ns.pop()
        else:
            self.masks = _to_device(masks, device) if device_resident else masks
            self.pixel_labels = masks.ndim >= 3  # (N, H, W) or a label stack (N, K, H, W)
            self.float_labels = np.issubdtype(np.asarray(masks[:0]).dtype, np.floating)
            self.n = self.masks.shape[0]
        self.dihedral, self.directions = dihedral, tuple(directions)
        unknown = set(self.directions) - (set(self.masks) if self.dict_labels else set())
        if unknown:
            raise ValueError(f"direction targets {sorted(unknown)} are not label entries")
        for k in self.directions:  # (cos 2t, sin 2t) in [-1, 1]: integer storage is a scaled
            dt = masks[k].dtype    # cache (e.g. int8 x 127) the loss would see unscaled
            is_float = (dt.is_floating_point if isinstance(dt, torch.dtype)
                        else np.issubdtype(dt, np.floating))
            if not is_float:
                raise ValueError(
                    f"direction target {k!r} has dtype {dt}; direction fields must be float "
                    "(cos 2t, sin 2t) in [-1, 1]. Dequantise an integer cache (e.g. int8 / 127) "
                    "before passing it")
        if self.directions and train and not dihedral:
            raise ValueError("direction targets need dihedral=True augmentation (a resized "
                             "crop would bend the directions and rescale the encoded widths)")
        self.bs = batch_size
        self.train = train
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.out_size = out_size or tuple(next(iter(self.images.values())).shape[-3:-1])
        if dihedral and train:
            size = tuple(next(iter(self.images.values())).shape[-3:-1])
            if size[0] != size[1] or tuple(self.out_size) != size:
                raise ValueError(f"dihedral augmentation needs square tiles at their own size "
                                 f"(tiles {size}, out_size {tuple(self.out_size)})")
        self.scale, self.ratio, self.hflip, self.vflip = scale, ratio, hflip, vflip
        self.drop_last = train if drop_last is None else drop_last
        if train and len(self) == 0:  # a run would complete without a single training step
            raise ValueError(
                f"training loader is empty: {self.n} samples, batch size {self.bs}"
                + (" with drop_last" if self.drop_last else "")
                + "; use more samples (e.g. a larger --subset) or a smaller batch size")
        self.stack = stack
        self.vectors = {} if stack or not vectors else {
            k: _to_device(np.asarray(v, dtype=np.float32), device) for k, v in vectors.items()}
        for k, v in self.vectors.items():
            if v.shape[0] != self.n:
                raise ValueError(f"vector input {k!r} has {v.shape[0]} rows for {self.n} samples")

    def _fetch(self, idx: Tensor):
        if self.device_resident:  # idx is already on the device (see __iter__)
            if self.dict_labels:
                return ({k: self.images[k][idx] for k in self.names},
                        {k: _dict_label(v[idx]) for k, v in self.masks.items()})
            return {k: self.images[k][idx] for k in self.names}, self._label(self.masks[idx])
        # Sorted reads are faster on memory-mapped files; restore the requested order after.
        sidx, inv = torch.sort(idx)
        npi = sidx.numpy()
        imgs = {k: _to_device(np.asarray(self.images[k][npi]), self.device)[inv]
                for k in self.names}
        if self.dict_labels:
            return imgs, {k: _dict_label(_to_device(np.asarray(v[npi]), self.device)[inv])
                          for k, v in self.masks.items()}
        y = _to_device(np.asarray(self.masks[npi]), self.device)[inv]
        return imgs, self._label(y)

    def _label(self, y: Tensor) -> Tensor:
        return y.float() if self.float_labels else y.long()

    def _augment_dihedral(self, x: Tensor, y):
        """One of the 8 dihedral transforms per sample (``manyseg.vectors.augment`` numbering:
        ``d % 4`` quarter turns counter-clockwise as displayed, then a horizontal flip for
        ``d >= 4``), applied exactly to the images, every label and the direction values."""
        d = torch.randint(8, (x.shape[0],), generator=self.gen)
        single = not isinstance(y, dict)
        # fresh tensors from _fetch, transformed in place; image-level labels do not move
        labels = {} if not self.pixel_labels else {None: y} if single else dict(y)
        for t in d.unique().tolist():
            if t == 0:
                continue
            sel = (d == t).nonzero()[:, 0].to(x.device)
            x[sel] = _dihedral(x[sel], t)
            for k, v in labels.items():
                v[sel] = _dihedral(v[sel], t, direction=k in self.directions)
        if not self.pixel_labels:  # image-level labels do not move with the pixels
            return x, y
        return x, (labels[None] if single else labels)

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
        theta = torch.zeros(n, 2, 3, pin_memory=self.device.type == "cuda")
        theta[:, 0, 0] = sw * fx
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = sh * fy
        theta[:, 1, 2] = ty
        theta = theta.to(self.device, non_blocking=True)  # pinned: no wait for the GPU
        grid = F.affine_grid(theta, (n, 1, *self.out_size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        if not self.pixel_labels:  # image-level labels do not move with the pixels
            return x, y
        if isinstance(y, dict):
            return x, {k: _warp_label(v, grid) for k, v in y.items()}
        return x, _warp_label(y, grid)  # float labels stay float (not rounded)

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.gen) if self.train else torch.arange(self.n)
        # One copy of the epoch's order to the device: a per-batch copy of a CPU index tensor
        # makes the host wait for the GPU every batch (E026).
        dev_order = order.to(self.device) if self.device_resident or self.vectors else None
        for b in range(len(self)):
            sl = slice(b * self.bs, (b + 1) * self.bs)
            idx = dev_order[sl] if self.device_resident else order[sl]
            imgs, y = self._fetch(idx)
            parts = []
            for k in self.names:  # the fetched batch is a fresh tensor: normalise in place
                if k in self.times:  # (B, T, H, W, C) -> (B, T * C, H, W), date-major
                    v = imgs[k].permute(0, 1, 4, 2, 3).float().sub_(self.mean[k]).div_(self.std[k])
                    parts.append(v.flatten(1, 2))
                else:
                    parts.append(imgs[k].permute(0, 3, 1, 2).float().sub_(self.mean[k])
                                 .div_(self.std[k]))
            x = (parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)).nan_to_num_()
            if self.train:
                x, y = self._augment_dihedral(x, y) if self.dihedral else self._augment(x, y)
            if (self.multi or self.times) and not self.stack:
                widths = [self.channels[k] * self.times.get(k, 1) for k in self.names]
                x = dict(zip(self.names, torch.split(x, widths, dim=1)))
                for k, t in self.times.items():
                    x[k] = x[k].unflatten(1, (t, self.channels[k]))
            if self.vectors:
                x = x if isinstance(x, dict) else {self.names[0]: x}
                vidx = dev_order[sl]
                x.update({k: v[vidx] for k, v in self.vectors.items()})
            yield x, y
