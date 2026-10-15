"""Confusion-matrix segmentation metrics (computed on-device, accumulated in int64)."""

from __future__ import annotations

import torch
from torch import Tensor


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int | None = None, device=None) -> None:
        self.k = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor) -> None:
        """pred, target: integer label maps of equal shape. Rows = target, cols = pred."""
        pred, target = pred.reshape(-1), target.reshape(-1)
        keep = (target >= 0) & (target < self.k)
        if self.ignore_index is not None:
            keep &= target != self.ignore_index
        idx = target[keep].long() * self.k + pred[keep].long()
        self.mat += torch.bincount(idx, minlength=self.k * self.k).reshape(self.k, self.k).to(
            self.mat.device
        )

    def compute(self) -> dict:
        m = self.mat.double()
        tp = m.diag()
        gt = m.sum(1)
        pr = m.sum(0)
        union = gt + pr - tp
        valid = gt > 0
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.nan)
        dice = torch.where(gt + pr > 0, 2 * tp / (gt + pr).clamp(min=1), torch.nan)
        acc = torch.where(valid, tp / gt.clamp(min=1), torch.nan)
        return {
            "miou": float(iou[valid].mean()),
            "mdice": float(dice[valid].mean()),
            "pixel_acc": float(tp.sum() / m.sum().clamp(min=1)),
            "mean_class_acc": float(acc[valid].mean()),
            "iou": [float(v) for v in iou],
            "dice": [float(v) for v in dice],
            "class_acc": [float(v) for v in acc],
        }


@torch.no_grad()
def boundary_f1(pred: Tensor, target: Tensor, tolerance: int = 2, ignore_index=None) -> Tensor:
    """Per-image boundary F1 for label maps (N, H, W) -> (N,), tolerance in pixels.

    Boundaries are pixels whose 4-neighbourhood contains another label; matching uses
    max-pool dilation by ``tolerance``.
    """
    import torch.nn.functional as F

    def edges(lbl: Tensor, valid: Tensor | None = None) -> Tensor:
        # A boundary pixel sees two different labels in its 3x3 neighbourhood. With ``valid``,
        # only valid neighbours count, so the border of an ignored region is not an edge.
        lbl = lbl.float().unsqueeze(1)
        hi = lbl if valid is None else torch.where(valid, lbl, torch.full_like(lbl, -1e9))
        lo = lbl if valid is None else torch.where(valid, lbl, torch.full_like(lbl, 1e9))
        mx = F.max_pool2d(hi, 3, 1, 1)
        mn = -F.max_pool2d(-lo, 3, 1, 1)
        return (mx != mn).float()

    if ignore_index is not None:
        valid = (target != ignore_index).unsqueeze(1)
        pe = edges(pred, valid) * valid.float()  # judged on valid pixels and neighbours only
        te = edges(target, valid) * valid.float()
    else:
        pe, te = edges(pred), edges(target)
    k = 2 * tolerance + 1
    pd = F.max_pool2d(pe, k, 1, tolerance)
    td = F.max_pool2d(te, k, 1, tolerance)
    prec = (pe * td).sum((1, 2, 3)) / pe.sum((1, 2, 3)).clamp(min=1)
    rec = (te * pd).sum((1, 2, 3)) / te.sum((1, 2, 3)).clamp(min=1)
    return 2 * prec * rec / (prec + rec).clamp(min=1e-12)
