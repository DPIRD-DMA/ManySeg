"""Reduced-resolution outputs: head scales, target downsampling, prediction upsampling and
coordinate mapping.

A pixel head with ``scale = 1/f`` (``f`` in 1, 2, 4, 8) predicts on a grid of
``ceil(H / f) x ceil(W / f)`` for an ``H x W`` input. Output pixel ``(i, j)`` covers input
pixels ``[i*f, (i+1)*f) x [j*f, (j+1)*f)``; the last row/column covers a partial block when
``f`` does not divide the size (ceil, so every input pixel belongs to exactly one output pixel).
With pixel centres at integer coordinates, output pixel ``i`` is centred at input coordinate
``(i + 0.5) * f - 0.5`` (:func:`to_input_coords`).

The downsampling helpers make targets for a loss at head scale:

* :func:`downsample_labels`: class ids, by majority vote per block (``mode="mode"``, ignored
  pixels do not vote, an all-ignored block stays ignored, ties go to the lowest id) or by the
  block-centre pixel (``mode="nearest"``).
* :func:`downsample_mean`: continuous targets, the mean over the in-image (and optionally
  valid) pixels of each block.
* :func:`downsample_sum`: density targets, the block sum, so the integral (count) is kept.

:func:`upsample_output` maps head outputs back to the input grid (e.g. to evaluate at full
resolution). These run outside the model graph (targets, metrics, post-processing).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .spatial import crop_to

SCALES = (1, 0.5, 0.25, 0.125)


def scale_factor(scale) -> int:
    """``1 / scale`` as an int, validating ``scale`` in {1, 1/2, 1/4, 1/8}."""
    ok = isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale > 0
    f = round(1 / scale) if ok else 0
    if not ok or f not in (1, 2, 4, 8) or abs(scale * f - 1) > 1e-9:
        raise ValueError(f"head scale must be one of 1, 1/2, 1/4, 1/8, got {scale!r}")
    return f


def scaled_size(h, w, scale) -> tuple:
    """``(ceil(h * scale), ceil(w * scale))`` via floor division (export-friendly)."""
    f = scale_factor(scale)
    return (h + f - 1) // f, (w + f - 1) // f


def _blocks(x: Tensor, f: int) -> Tensor:
    """``(..., H, W) -> (..., Hs, Ws, f*f)`` after padding to multiples of ``f`` (value 0)."""
    *lead, h, w = x.shape
    hs, ws = (h + f - 1) // f, (w + f - 1) // f
    x = F.pad(x, (0, ws * f - w, 0, hs * f - h))
    x = x.reshape(*lead, hs, f, ws, f).movedim(-3, -2)
    return x.reshape(*lead, hs, ws, f * f)


@torch.no_grad()
def downsample_labels(labels: Tensor, scale, ignore_index: int | None = None,
                      mode: str = "mode", num_classes: int | None = None) -> Tensor:
    """Class-id targets ``(..., H, W)`` on a head grid of ``scale``.

    ``mode="mode"``: the most frequent non-ignored id in each block (ties -> lowest id; an
    all-ignored block or a partial block's padding never votes). ``mode="nearest"``: the pixel
    at the block centre ``(i*f + f//2, j*f + f//2)``, clamped into the image.

    ``num_classes`` (ids in ``[0, num_classes)``) avoids reading the id range from the data,
    i.e. a device sync per call in a training loop; other non-ignored ids then do not vote.
    """
    f = scale_factor(scale)
    if f == 1:
        return labels
    h, w = labels.shape[-2:]
    if mode == "nearest":
        ys = (torch.arange((h + f - 1) // f, device=labels.device) * f + f // 2).clamp(max=h - 1)
        xs = (torch.arange((w + f - 1) // f, device=labels.device) * f + f // 2).clamp(max=w - 1)
        return labels[..., ys[:, None], xs[None, :]]
    if mode != "mode":
        raise ValueError(f"mode must be 'mode' or 'nearest', got {mode!r}")
    lab = labels.long()
    valid = torch.ones_like(lab, dtype=torch.bool) if ignore_index is None else lab != ignore_index
    if num_classes is not None:
        lo, k = 0, num_classes
        valid = valid & (lab >= 0) & (lab < k)
    elif bool(valid.any()):
        lo = int(lab[valid].min())
        k = int(lab[valid].max()) - lo + 1
    else:
        lo, k = 0, 1
    blk = _blocks(lab - lo, f)                       # (..., Hs, Ws, f*f)
    votes = _blocks(valid.to(torch.long), f)         # 0 for ignored pixels and the padding
    shape = blk.shape[:-1]
    n = shape.numel()
    cell = torch.arange(n, device=lab.device).view(shape)[..., None]
    counts = torch.zeros(n * k, dtype=torch.long, device=lab.device)
    counts.index_add_(0, (cell * k + blk.clamp(0, k - 1)).reshape(-1), votes.reshape(-1))
    counts = counts.view(*shape, k)
    out = counts.argmax(-1) + lo
    if ignore_index is not None:
        out = torch.where(counts.sum(-1) > 0, out, torch.full_like(out, ignore_index))
    return out.to(labels.dtype)


@torch.no_grad()
def downsample_mean(x: Tensor, scale, valid: Tensor | None = None) -> Tensor:
    """Area average ``(..., H, W)`` over each block's in-image pixels (and ``valid`` pixels if
    given); a block without any such pixel is 0."""
    f = scale_factor(scale)
    if f == 1 and valid is None:
        return x
    w8 = torch.ones_like(x) if valid is None else valid.to(x.dtype).expand_as(x)
    s = _blocks(x * w8, f).sum(-1)
    n = _blocks(w8, f).sum(-1)
    return torch.where(n > 0, s / n.clamp(min=1), torch.zeros_like(s))


@torch.no_grad()
def downsample_sum(x: Tensor, scale) -> Tensor:
    """Block sums ``(..., H, W)``: the total (e.g. a density map's count) is preserved."""
    f = scale_factor(scale)
    return x if f == 1 else _blocks(x, f).sum(-1)


def upsample_output(y: Tensor, size, scale, mode: str = "bilinear") -> Tensor:
    """Map a head output ``(..., ceil(H*s), ceil(W*s))`` back to the input grid ``size = (H, W)``.

    ``bilinear`` (logits, continuous values; ``align_corners=False``, i.e. consistent with
    output pixel ``i`` covering input ``[i*f, (i+1)*f)``) or ``nearest`` (labels, any dtype:
    each output pixel is repeated over its block). The ``f``-times larger map is cropped.
    """
    f = scale_factor(scale)
    if mode == "nearest":
        out = y.repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1) if f > 1 else y
        return out[..., : size[0], : size[1]]
    if mode != "bilinear":
        raise ValueError(f"mode must be 'bilinear' or 'nearest', got {mode!r}")
    if f > 1:
        h, w = y.shape[-2], y.shape[-1]
        y = F.interpolate(y, size=(h * f, w * f), mode="bilinear", align_corners=False)
    return crop_to(y, size[0], size[1])


def to_input_coords(points: Tensor, scale) -> Tensor:
    """Head-grid ``(y, x)`` coordinates -> input-pixel coordinates: ``(p + 0.5) * f - 0.5``."""
    f = scale_factor(scale)
    return points if f == 1 else (points.float() + 0.5) * f - 0.5


def to_output_coords(points: Tensor, scale) -> Tensor:
    """Input-pixel ``(y, x)`` coordinates -> head-grid coordinates: ``(p + 0.5) / f - 0.5``."""
    f = scale_factor(scale)
    return points if f == 1 else (points.float() + 0.5) / f - 0.5


__all__ = ["SCALES", "downsample_labels", "downsample_mean", "downsample_sum", "scale_factor",
           "scaled_size", "to_input_coords", "to_output_coords", "upsample_output"]
