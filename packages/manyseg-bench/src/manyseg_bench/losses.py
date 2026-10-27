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


def distance_to(geometry: Tensor, max_distance: float) -> Tensor:
    """Euclidean distance (px) from each pixel to the nearest ``True`` pixel of ``geometry``
    ``(B, H, W)`` bool, capped at ``max_distance`` (also for images with no geometry). Uses the
    library's exact GPU distance transform."""
    from manyseg.instances import edt

    return edt(geometry.bool()).clamp(max=max_distance)


def distance_aware_bce(logits: Tensor, target: Tensor, geometry: Tensor | None = None, *,
                       tolerance: float = 2.0, pull: float = 0.1, max_distance: float = 32.0,
                       distance: Tensor | None = None) -> Tensor:
    """BCE for sparse targets (points, vertices, lines, boundaries) that knows how far a wrong
    prediction is from the true geometry.

    Plain BCE charges a false positive the same whether it is 1 px or 100 px from a true point,
    so a slightly shifted prediction is punished like a missing one plus a far-away false one.
    Here, with ``D`` the distance (px) of each pixel to the nearest true element:

    * false positives are weighted ``w(D) = 1 - exp(-D^2 / (2 tolerance^2))``: almost free next to
      the truth, full cost far from it (CenterNet's penalty reduction, generalised from Gaussian
      points to any geometry through the distance map);
    * a pull term ``pull * sum(p * D') / max(sum(y), 1)`` per image, with ``D' = D / max_distance``:
      predicted mass costs in proportion to its (capped) distance from the truth, so the further
      away, the larger the loss and its gradient; it is zero on the truth and never rewards a
      wrong pixel (normalising by the predicted mass instead would reward pixels closer than
      average, i.e. near misses);
    * true pixels are trained as in BCE (``-y log p``), which raises the correct location while
      the two terms above lower the wrong ones.

    Args:
        logits: ``(B, 1, H, W)`` or ``(B, H, W)`` raw outputs.
        target: same shape, in [0, 1] (binary or a soft blob such as a Gaussian heatmap).
        geometry: bool ``(B, H, W)`` true elements (point / vertex pixels, line or boundary pixels);
            default ``target >= 0.5``.
        distance: precomputed ``D`` ``(B, H, W)`` (e.g. from the data loader) instead of
            ``geometry``.
    """
    x = logits.float().reshape(logits.shape[0], *logits.shape[-2:])
    y = target.float().reshape(x.shape)
    if distance is None:
        geometry = (y >= 0.5) if geometry is None else geometry.reshape(x.shape)
        distance = distance_to(geometry, max_distance)
    d = distance.float().clamp(max=max_distance)
    w_neg = 1 - torch.exp(-d ** 2 / (2 * tolerance ** 2))
    # BCE written out: -y log p - (1 - y) w log(1 - p), stable via logsigmoid
    loss = -(y * F.logsigmoid(x) + (1 - y) * w_neg * F.logsigmoid(-x)).mean()
    if pull > 0:
        p = x.sigmoid()
        per = (p * d / max_distance).flatten(1).sum(1) / y.flatten(1).sum(1).clamp(min=1)
        loss = loss + pull * per.mean()
    return loss
