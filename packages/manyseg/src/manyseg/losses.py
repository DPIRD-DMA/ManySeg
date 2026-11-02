"""Losses for dense heads, as plain functions on logits and targets (no state, no modules).

* Segmentation: :func:`soft_dice` and :func:`dice_bce` (binary or one channel per label, on
  logits), :func:`multiclass_dice` (softmax over classes, integer targets).
* Regression: :func:`masked_regression` (L1 / MSE over valid pixels).
* Heatmaps for points, corners and keypoints: :func:`heatmap_focal` (CenterNet focal loss)
  with :func:`heatmap_peaks` for its positives, and :func:`distance_weighted_bce`, whose false
  positives cost more the further they are from the true geometry (:func:`distance_map`).
* Density heads: :func:`~manyseg.points.density_loss` (re-exported here).
* Several heads: :func:`multi_head_loss` sums a loss per head of a model's output dict.

The training code that chose these, with the weights between heads used in the experiments,
is in the ``manyseg-bench`` package, which re-exports this module. ``distance_map`` needs SciPy
(``manyseg[instances]``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .points.losses import density_loss


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


def heatmap_focal(logits: Tensor, target: Tensor, alpha: float = 2.0, beta: float = 4.0,
                  pos: Tensor | None = None, min_pos: int = 1) -> Tensor:
    """CenterNet penalty-reduced focal loss for Gaussian centre heatmaps (logits in). ``pos``
    (bool, like ``target``) marks the centre pixels; default ``target >= 0.999``, which misses
    centres that fall between pixels (pass e.g. the target's local maxima then). The summed loss
    is divided by the number of positives, at least ``min_pos``."""
    p = logits.float().sigmoid().clamp(1e-6, 1 - 1e-6)
    pos = target >= 0.999 if pos is None else pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (~pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=min_pos)


def heatmap_peaks(target: Tensor, threshold: float = 0.5) -> Tensor:
    """Positives of a Gaussian heatmap target ``(B, C, H, W)`` for :func:`heatmap_focal`: its
    3 x 3 local maxima above ``threshold`` (a point on a pixel edge never reaches 1 at a pixel
    centre, so ``target >= 0.999`` would miss it)."""
    t = target.float()
    return (t > threshold) & (t == F.max_pool2d(t, 3, 1, 1))


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


def multiclass_dice(logits: Tensor, target: Tensor, *, ignore_index: int | None = None,
                    eps: float = 1.0) -> Tensor:
    """1 - Dice on softmax probabilities, per class over the whole batch, averaged over
    classes. ``logits`` ``(B, C, H, W)``; ``target`` ``(B, H, W)`` integer class ids (or
    ``(B, 1, H, W)``). Pixels equal to ``ignore_index`` count for no class. A class absent from
    both target and prediction scores about 1 (smoothing ``eps``), so it neither helps nor
    hurts. Add it to cross-entropy (``F.cross_entropy(logits, target) + multiclass_dice(...)``)
    for classes that cover few pixels."""
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B, C, H, W), got {tuple(logits.shape)}")
    target = _as_bhw(target, "target").long()
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(f"target {tuple(target.shape)} does not match logits "
                         f"{tuple(logits.shape)}")
    classes = logits.shape[1]
    valid = None if ignore_index is None else target != ignore_index
    labels = target if valid is None else target[valid]
    if labels.numel() and not (labels.min() >= 0 and labels.max() < classes):
        raise ValueError(f"target has class ids outside [0, {classes}) that are not ignore_index")
    # Per-class sums by label rather than through a one-hot, which takes C x H x W longs. Summed
    # in float64 where available: float32 counts stop growing at 2^24 pixels.
    p = logits.float().softmax(1)
    wide = torch.float32 if p.device.type == "mps" else torch.float64
    index = (target if valid is None else target.where(valid, 0))[:, None]
    weight = None if valid is None else valid[:, None].to(wide)
    at_label = p.gather(1, index).to(wide)
    at_label = at_label if weight is None else at_label * weight
    count = torch.ones_like(at_label) if weight is None else weight
    inter = at_label.new_zeros(classes).scatter_add(0, index.flatten(), at_label.flatten())
    labelled = at_label.new_zeros(classes).scatter_add(0, index.flatten(), count.flatten())
    total = p if valid is None else p * valid[:, None]
    denom = total.sum((0, 2, 3)) + labelled.to(p.dtype)
    return (1 - (2 * inter.to(p.dtype) + eps) / (denom + eps)).mean()


def multi_head_loss(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor],
                    losses: Mapping[str, Callable[[Tensor, Tensor], Tensor]],
                    weights: Mapping[str, float] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    """The weighted sum of a loss per head: ``sum(weights[h] * losses[h](outputs[h],
    targets[h]))`` over the heads in ``losses``, with weights defaulting to 1. Returns the total
    and each head's unweighted loss, detached, for logging. Heads of ``outputs`` without a loss
    are ignored (e.g. an auxiliary head you only monitor); a head in ``losses`` that is missing
    from ``outputs`` or ``targets`` raises, as do weights for heads without a loss.

    For a loss that needs more than the head's output and target (a mask, positives), close
    over them: ``{"corner": lambda o, t: heatmap_focal(o, t, pos=heatmap_peaks(t))}``.
    """
    weights = dict(weights or {})
    unknown = sorted(set(weights) - set(losses))
    if unknown:
        raise ValueError(f"weights for heads without a loss: {unknown}")
    if not losses:
        raise ValueError("losses is empty")
    total, parts = None, {}
    for head, fn in losses.items():
        missing = [n for n, d in (("outputs", outputs), ("targets", targets)) if head not in d]
        if missing:
            raise KeyError(f"head {head!r} has a loss but is missing from {' and '.join(missing)}")
        value = fn(outputs[head], targets[head])
        parts[head] = value.detach()
        term = weights.get(head, 1.0) * value
        total = term if total is None else total + term
    return total, parts


__all__ = ["density_loss", "dice_bce", "distance_map", "distance_weighted_bce", "heatmap_focal",
           "heatmap_peaks", "masked_regression", "multi_head_loss", "multiclass_dice",
           "soft_dice"]
