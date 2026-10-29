"""Object-level metrics for instance and point outputs (accumulated over images).

Conventions:
* Instance matching: a predicted and a ground-truth object match when IoU > ``iou_threshold``
  (0.5 by default, which makes matches unique); TP/FP/FN, precision, recall, F1.
* PQ = SQ x RQ with SQ = mean IoU of matches and RQ = TP / (TP + FP/2 + FN/2) (Kirillov et al.).
* Over-/under-segmentation (per ground-truth object, partner = predicted object with the largest
  overlap): ``over_seg = 1 - |G ∩ P| / |G|`` and ``under_seg = 1 - |G ∩ P| / |P|``, averaged over
  ground-truth objects (field-delineation convention, e.g. Persello et al. 2019).
* Points: one-to-one matching within ``radius`` pixels (optimal via scipy if installed,
  otherwise greedy by distance).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _overlaps(pred: Tensor, gt: Tensor):
    pred, gt = pred.reshape(-1).long(), gt.reshape(-1).long()
    pu, pi = torch.unique(pred, return_inverse=True)
    gu, gi = torch.unique(gt, return_inverse=True)
    inter = torch.bincount(gi * len(pu) + pi, minlength=len(gu) * len(pu)).view(len(gu), len(pu))
    gsize, psize = inter.sum(1), inter.sum(0)
    gkeep, pkeep = gu > 0, pu > 0
    inter = inter[gkeep][:, pkeep].double()
    gsize, psize = gsize[gkeep].double(), psize[pkeep].double()
    union = gsize[:, None] + psize[None, :] - inter
    return inter, gsize, psize, inter / union.clamp(min=1)


def _one_to_one(iou: Tensor, thr: float) -> Tensor:
    """Boolean (G, P) match matrix. For thr >= 0.5, IoU > thr matches are unique already;
    below that, use a maximum-IoU one-to-one assignment (scipy if available, else greedy)."""
    cand = iou > thr
    if thr >= 0.5 or not bool(cand.any()):
        return cand
    out = torch.zeros_like(cand)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-torch.where(cand, iou, -1.0).numpy())
        for i, j in zip(r, c):
            out[i, j] = bool(cand[i, j])
    except ImportError:
        used_g, used_p = set(), set()
        for flat in torch.argsort(iou.reshape(-1), descending=True).tolist():
            i, j = divmod(flat, iou.shape[1])
            if not cand[i, j]:
                break
            if i not in used_g and j not in used_p:
                out[i, j] = True
                used_g.add(i)
                used_p.add(j)
    return out


def panoptic_quality(pred: Tensor, gt: Tensor, iou_threshold: float = 0.5) -> dict:
    om = ObjectMetrics(iou_threshold)
    om.update(pred, gt)
    r = om.compute()
    return {k: r[k] for k in ("pq", "sq", "rq")}


class ObjectMetrics:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.thr = iou_threshold
        self.tp = self.fp = self.fn = 0
        self.iou_sum = 0.0
        self.os_sum = self.us_sum = 0.0
        self.n_gt = 0

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for p, g in zip(pred.reshape(-1, *pred.shape[-2:]), gt.reshape(-1, *gt.shape[-2:])):
            inter, gsize, psize, iou = _overlaps(p.cpu(), g.cpu())
            n_g, n_p = iou.shape
            match = _one_to_one(iou, self.thr)
            tp = int(match.sum())
            self.tp += tp
            self.fp += n_p - int(match.any(0).sum())
            self.fn += n_g - int(match.any(1).sum())
            self.iou_sum += float(iou[match].sum())
            self.n_gt += n_g
            if n_g and n_p:
                best = inter.argmax(1)
                i = inter[torch.arange(n_g), best]
                self.os_sum += float((1 - i / gsize).sum())
                self.us_sum += float((1 - i / psize[best]).sum())
            elif n_g:
                self.os_sum += n_g
                self.us_sum += n_g

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        sq = self.iou_sum / tp if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
        n = max(1, self.n_gt)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
                "sq": sq, "rq": rq, "pq": sq * rq, "over_seg": self.os_sum / n,
                "under_seg": self.us_sum / n}


def match_points(pred: Tensor, gt: Tensor, radius: float) -> tuple[int, int, int]:
    """One-to-one matching of ``(N, 2)`` point sets within ``radius``: (tp, fp, fn)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)
    d = torch.cdist(pred.double().cpu(), gt.double().cpu())
    try:
        from scipy.optimize import linear_sum_assignment

        cost = torch.where(d <= radius, d, torch.full_like(d, 1e9)).numpy()
        r, c = linear_sum_assignment(cost)
        tp = int(sum(cost[i, j] < 1e9 for i, j in zip(r, c)))
    except ImportError:
        tp, used_p, used_g = 0, set(), set()
        for flat in torch.argsort(d.reshape(-1)).tolist():
            i, j = divmod(flat, d.shape[1])
            if d[i, j] > radius:
                break
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                tp += 1
    return tp, len(pred) - tp, len(gt) - tp


class PointMetrics:
    def __init__(self, radius: float = 3.0) -> None:
        self.radius = radius
        self.tp = self.fp = self.fn = 0

    def update(self, pred: Tensor, gt: Tensor) -> None:
        tp, fp, fn = match_points(pred, gt, self.radius)
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


class CountMetrics:
    def __init__(self) -> None:
        self.abs = self.sq = self.signed = 0.0
        self.n = 0

    def update(self, pred: Tensor, gt: Tensor) -> None:
        e = pred.double().cpu() - gt.double().cpu()
        self.abs += float(e.abs().sum())
        self.signed += float(e.sum())
        self.sq += float((e ** 2).sum())
        self.n += e.numel()

    def compute(self) -> dict:
        n = max(1, self.n)
        # bias: mean signed error (predicted - true count), e.g. a softplus background floor
        return {"mae": self.abs / n, "rmse": math.sqrt(self.sq / n), "bias": self.signed / n,
                "n": self.n}
