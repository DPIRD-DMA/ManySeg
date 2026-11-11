"""Losses for density heads (pure torch)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor

from .postprocess import PER_POINT_SIGMA, _squeeze
from .targets import _per_image, density_map


def density_loss(pred: Tensor, points: Sequence[Tensor], sigma=3.0, *,
                 value_scale: float | None = None, count_weight: float = 0.0,
                 mask: Tensor | None = None, kind: str = "mse") -> Tensor:
    """Pixel loss of a density head against a mass-preserving target, plus an optional count
    term.

    ``pred`` ``(B, 1, H, W)``: the head's output. ``points``: per image ``(N_i, 2)`` ``(y, x)``
    on ``pred``'s grid (for a head with ``scale`` < 1, ``(p + 0.5) * scale - 0.5``).
    ``sigma`` as in :func:`density_map` (a float, or per-image per-point tensors).

    The target is :func:`density_map` (it sums to the number of points inside the image;
    points outside add nothing) times ``value_scale``.
    The default, ``2 pi sigma^2``, makes an isolated object's peak about 1. Unscaled, a peak
    is ``1 / (2 pi sigma^2)`` (0.001 for sigma 12), so the pixel loss is tiny and the model
    can get away with a near-constant map. Predicted counts are then
    ``count_from_density(pred, sigma=sigma)`` (or ``value_scale=`` the same value if given
    here), and ``DensityBlockDecoder`` / ``block_decoder("density")`` take the same
    ``sigma=`` / ``value_scale=``. With per-point sigmas, give ``value_scale``.

    ``mask`` ``(B, [1,] H, W)``, > 0 for valid pixels: the pixel loss and both counts use only
    those. ``kind``: ``"mse"`` or ``"l1"``, the mean over valid pixels. ``count_weight`` > 0
    adds ``count_weight * mean |predicted count - true count|`` per image (in objects). The
    count term sees the whole crop: a bias learnt on small training crops is summed over a
    larger area at inference, so check counts on full-size validation images.
    """
    if kind not in ("mse", "l1"):
        raise ValueError(f"kind must be 'mse' or 'l1', got {kind!r}")
    if pred.dim() != 4 or pred.shape[1] != 1:
        raise ValueError(f"pred must be (B, 1, H, W), got {tuple(pred.shape)}")
    if len(points) != pred.shape[0]:
        raise ValueError(f"{len(points)} point lists for a batch of {pred.shape[0]}")
    if value_scale is None:
        sig = _per_image(sigma, len(points))[0]
        if isinstance(sig, Tensor) and sig.dim() > 0:
            raise ValueError(PER_POINT_SIGMA)
        value_scale = 2 * math.pi * float(sig) ** 2
    p = pred.float()
    target = density_map([q.to(p.device) for q in points], tuple(p.shape[-2:]), sigma)
    target = target * value_scale
    m = None if mask is None else (_squeeze(mask, "mask", like=p)[:, None].to(p.device) > 0).float()
    err = F.mse_loss(p, target, reduction="none") if kind == "mse" else (p - target).abs()
    loss = err.mean() if m is None else (err * m).sum() / m.sum().clamp(min=1)
    if count_weight:
        count_pred = (p if m is None else p * m).flatten(1).sum(1) / value_scale
        count_true = (target if m is None else target * m).flatten(1).sum(1) / value_scale
        loss = loss + count_weight * (count_pred - count_true).abs().mean()
    return loss
