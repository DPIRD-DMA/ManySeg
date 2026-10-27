"""Losses for dense tasks (kept out of the model library by design)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def soft_dice(logits: Tensor, target: Tensor, mask: Tensor | None = None,
              eps: float = 1.0) -> Tensor:
    """1 - Dice on sigmoid probabilities, per sample then averaged. Shapes (B, 1, H, W)."""
    p = logits.float().sigmoid()
    t = target.float()
    if mask is not None:
        p, t = p * mask, t * mask
    inter = (p * t).flatten(1).sum(1)
    denom = p.flatten(1).sum(1) + t.flatten(1).sum(1)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def dice_bce(logits: Tensor, target: Tensor, pos_weight: float | None = None,
             mask: Tensor | None = None, dice_weight: float = 1.0,
             bce_weight: float = 1.0) -> Tensor:
    """Binary Dice + BCE on logits. ``pos_weight`` up-weights positive pixels in the BCE term
    (useful for thin boundary classes); ``mask`` restricts both terms to valid pixels."""
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), pos_weight=pw,
                                             reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1) if mask is not None else bce.mean()
    return bce_weight * bce + dice_weight * soft_dice(logits, target, mask)


def masked_regression(pred: Tensor, target: Tensor, mask: Tensor | None = None,
                      kind: str = "l1") -> Tensor:
    """L1 / MSE averaged over ``mask`` pixels (all pixels if None; 0 if the mask is empty)."""
    err = (pred.float() - target.float()).abs() if kind == "l1" else (pred.float() - target) ** 2
    if mask is None:
        return err.mean()
    m = mask.expand_as(err).float()
    return (err * m).sum() / m.sum().clamp(min=1)


def heatmap_focal(logits: Tensor, target: Tensor, alpha: float = 2.0, beta: float = 4.0) -> Tensor:
    """CenterNet penalty-reduced focal loss for Gaussian centre heatmaps (logits in)."""
    p = logits.float().sigmoid().clamp(1e-6, 1 - 1e-6)
    pos = target >= 0.999
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (~pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def distance_map(geometry, max_distance: float = 32.0):
    """Distance (px, float32) from each pixel to the nearest ``True`` pixel of a 2-D bool
    ``geometry`` (numpy), capped at ``max_distance`` (also when there is no geometry). Linear
    time (SciPy); meant for data loaders, since the distances depend only on the target."""
    import numpy as np
    from scipy import ndimage

    g = np.asarray(geometry, bool)
    if not g.any():
        return np.full(g.shape, max_distance, np.float32)
    return np.minimum(ndimage.distance_transform_edt(~g), max_distance).astype(np.float32)


def _as_bhw(t: Tensor, name: str) -> Tensor:
    if t.dim() == 4 and t.shape[1] == 1:
        return t[:, 0]
    if t.dim() != 3:
        raise ValueError(f"{name} must be (B, H, W) or (B, 1, H, W), got {tuple(t.shape)}")
    return t


def distance_weighted_bce(logits: Tensor, target: Tensor, distance: Tensor, *,
                          tolerance: float = 2.0, far_weight: float = 1.0,
                          max_distance: float = 32.0) -> Tensor:
    """BCE for sparse targets (points, vertices, lines, boundaries) whose false positives cost
    more the further they are from the true geometry.

    Plain BCE charges a false positive the same whether it is 1 px or 100 px from a true
    element. Here, with ``D`` the distance (px) of each pixel to the nearest true element (from
    :func:`distance_map`, computed on the *exact* geometry: point / vertex pixels, line pixels;
    never a threshold of a soft target):

    * inside the target (``y > 0``, e.g. a Gaussian blob) plain BCE, so a soft target is learned
      as given;
    * outside it (``y == 0``) the negative term is weighted ``w(D) = 1 - exp(-D^2 /
      (2 tolerance^2))``: a false positive just next to the truth is cheap, one far away costs
      full BCE (CenterNet's penalty reduction, generalised to any geometry by the distance map);
    * plus ``far_weight * p * D / max_distance`` outside the target: an extra penalty that grows
      with distance (capped at ``max_distance``). At ``D = max_distance`` it adds ``far_weight``
      times the BCE gradient of a confident false positive; the default 1 doubles it there.

    Both terms are averaged over pixels, like BCE, so the scale does not depend on image size or
    on how much geometry there is, and an empty image costs what BCE would charge for its
    predictions plus the capped penalty. This *suppresses* wrong predictions in proportion to
    their distance; it does not move a prediction sideways towards the truth. For that, train an
    offset field (vectors to the nearest true element) and snap predictions with it.

    Args:
        logits, target, distance: ``(B, H, W)`` or ``(B, 1, H, W)``; ``target`` in [0, 1] (binary
            or a soft blob around each element), ``distance`` >= 0 in pixels.
    """
    if tolerance <= 0 or max_distance <= 0 or far_weight < 0:
        raise ValueError("need tolerance > 0, max_distance > 0, far_weight >= 0")
    x = _as_bhw(logits, "logits").float()
    y = _as_bhw(target, "target").float()
    d = _as_bhw(distance, "distance").float()
    if not (x.shape == y.shape == d.shape):
        raise ValueError(f"shape mismatch: logits {tuple(x.shape)}, target {tuple(y.shape)}, "
                         f"distance {tuple(d.shape)}")
    d = d.clamp(0, max_distance)
    outside = y == 0
    w_neg = torch.where(outside, 1 - torch.exp(-d ** 2 / (2 * tolerance ** 2)), torch.ones_like(d))
    loss = -(y * F.logsigmoid(x) + (1 - y) * w_neg * F.logsigmoid(-x)).mean()
    if far_weight > 0:
        loss = loss + far_weight * (x.sigmoid() * (d / max_distance) * outside).mean()
    return loss
