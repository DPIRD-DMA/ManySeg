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
