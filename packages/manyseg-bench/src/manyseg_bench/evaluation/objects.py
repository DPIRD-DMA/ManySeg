"""Object-level metrics for instance and point outputs (accumulated over images).

Conventions:
* Instance matching: a predicted and a ground-truth object match when IoU > ``iou_threshold``
  (0.5 by default, which makes matches unique); TP/FP/FN, precision, recall, F1. Below 0.5 the
  one-to-one matches are the assignment with the largest IoU sum (exact, no scipy needed, so
  results do not depend on what is installed; per image this maximises PQ, whose denominator
  ``TP + FP/2 + FN/2 = (G + P) / 2`` does not depend on the matching).
* PQ = SQ x RQ with SQ = mean IoU of matches and RQ = TP / (TP + FP/2 + FN/2) (Kirillov et al.).
* Over-/under-segmentation (per ground-truth object, partner = predicted object with the largest
  overlap): ``over_seg = 1 - |G ∩ P| / |G|`` and ``under_seg = 1 - |G ∩ P| / |P|``, averaged over
  ground-truth objects (field-delineation convention, e.g. Persello et al. 2019).
* Points: one-to-one matching within ``radius`` pixels; TP = size of a maximum matching
  (exact, no scipy needed, so results do not depend on what is installed).
"""

from __future__ import annotations

import math

import numpy as np
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


def _hungarian_max(w: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight assignment of an ``(n, m)`` weight matrix with ``n <= m``: one column
    per row (shortest augmenting paths with potentials, O(n^2 m)); deterministic for a given
    matrix."""
    n, m = w.shape
    cost = -w
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j]: row (1-based) assigned to column j, 0 = none
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j]]


def _max_iou_assignment(iou: np.ndarray, cand: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one candidate pairs with the largest IoU sum, solved per connected component of
    the candidate graph (most are single pairs)."""
    n_g, n_p = cand.shape
    parent = list(range(n_g + n_p))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = np.argwhere(cand)
    for i, j in edges:
        a, b = find(int(i)), find(n_g + int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    comps: dict[int, tuple[list, list]] = {}
    for i in sorted({int(i) for i in edges[:, 0]}):
        comps.setdefault(find(i), ([], []))[0].append(i)
    for j in sorted({int(j) for j in edges[:, 1]}):
        comps.setdefault(find(n_g + j), ([], []))[1].append(j)
    pairs = []
    for rows, cols in comps.values():
        w = np.where(cand[np.ix_(rows, cols)], iou[np.ix_(rows, cols)], 0.0)
        if len(rows) <= len(cols):
            sol = _hungarian_max(w)
        else:
            sol = [(r, c) for c, r in _hungarian_max(w.T)]
        pairs += [(rows[r], cols[c]) for r, c in sol if cand[rows[r], cols[c]]]
    return pairs


def _one_to_one(iou: Tensor, thr: float) -> Tensor:
    """Boolean (G, P) match matrix. For thr >= 0.5, IoU > thr matches are unique already;
    below that, the one-to-one assignment of IoU > thr pairs with the largest IoU sum
    (exact and scipy-free, so the result does not depend on the environment)."""
    cand = iou > thr
    if thr >= 0.5 or not bool(cand.any()):
        return cand
    out = torch.zeros_like(cand)
    for i, j in _max_iou_assignment(iou.double().numpy(), cand.numpy()):
        out[i, j] = True
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


def _max_matching(adj: list[list[int]], n_right: int) -> int:
    """Size of a maximum-cardinality matching of a bipartite graph (Hopcroft-Karp). ``adj[u]``
    lists the right vertices of left vertex ``u``. Iterative: long chains of points cannot hit
    the recursion limit."""
    from collections import deque

    n_left, inf = len(adj), len(adj) + n_right + 1
    match_l, match_r = [-1] * n_left, [-1] * n_right
    while True:
        dist = [inf] * n_left
        q = deque(u for u in range(n_left) if match_l[u] == -1)
        for u in q:
            dist[u] = 0
        found = False
        while q:  # BFS layers from the free left vertices
            u = q.popleft()
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == inf:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if not found:
            return sum(m != -1 for m in match_l)
        nxt = [0] * n_left
        for s in range(n_left):  # DFS along the layers, with an explicit stack
            if match_l[s] != -1:
                continue
            stack = [s]
            while stack:
                u = stack[-1]
                if nxt[u] == len(adj[u]):
                    dist[u] = inf  # dead end for this phase
                    stack.pop()
                    continue
                v = adj[u][nxt[u]]
                nxt[u] += 1
                w = match_r[v]
                if w == -1:  # augmenting path: flip it (each u used edge adj[u][nxt[u] - 1])
                    for x in stack:
                        y = adj[x][nxt[x] - 1]
                        match_l[x], match_r[y] = y, x
                    break
                if dist[w] == dist[u] + 1:
                    stack.append(w)


def match_points(pred: Tensor, gt: Tensor, radius: float) -> tuple[int, int, int]:
    """One-to-one matching of ``(N, 2)`` point sets within ``radius``: (tp, fp, fn).

    ``tp`` is the size of a maximum one-to-one matching of pairs at distance <= ``radius`` (the
    count a minimum-distance assignment such as scipy's ``linear_sum_assignment`` with
    out-of-radius pairs forbidden gives), computed exactly without scipy. Which pairs match
    when several maximum matchings exist does not change the counts.
    """
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)
    d = torch.cdist(pred.double().cpu(), gt.double().cpu())
    adj: list[list[int]] = [[] for _ in range(len(pred))]
    for i, j in (d <= radius).nonzero().tolist():
        adj[i].append(j)
    tp = _max_matching(adj, len(gt))
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
