"""Post-processing for point/density heads (runs outside the exported model graph)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def find_peaks(heatmap: Tensor, threshold: float = 0.5, kernel: int = 3,
               max_points: int | None = None) -> list[tuple[Tensor, Tensor]]:
    """Local maxima via max-pool NMS. Returns per image ``(points (K, 2) long (y, x), scores)``,
    sorted by score descending. Pass probabilities (apply ``sigmoid`` to logits first)."""
    if kernel % 2 != 1:
        raise ValueError(f"kernel must be odd, got {kernel}")
    hm = heatmap[:, 0] if heatmap.dim() == 4 else heatmap
    pooled = F.max_pool2d(hm[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    keep = (hm == pooled) & (hm > threshold)
    out = []
    for i in range(hm.shape[0]):
        pts = torch.nonzero(keep[i])
        scores = hm[i][keep[i]]
        order = torch.argsort(scores, descending=True)
        if max_points is not None:
            order = order[:max_points]
        out.append((pts[order], scores[order]))
    return out


def count_from_density(density: Tensor) -> Tensor:
    """Predicted count per image: the integral of the density map."""
    return density.flatten(1).sum(1)


@torch.no_grad()
def group_by_offsets(centers: Sequence[Tensor], offsets: Tensor, foreground: Tensor,
                     chunk: int = 65536) -> Tensor:
    """Panoptic-DeepLab grouping: each foreground pixel votes ``pixel + offset`` and joins the
    nearest centre. Returns instance labels ``(B, H, W)`` (centre ``k`` -> label ``k + 1``)."""
    b, _, h, w = offsets.shape
    ys = torch.arange(h, device=offsets.device, dtype=torch.float32)
    xs = torch.arange(w, device=offsets.device, dtype=torch.float32)
    vy = (offsets[:, 0] + ys[:, None]).reshape(b, -1)
    vx = (offsets[:, 1] + xs[None, :]).reshape(b, -1)
    labels = torch.zeros(b, h * w, dtype=torch.long, device=offsets.device)
    for i, c in enumerate(centers):
        if len(c) == 0:
            continue
        c = c.float().to(offsets.device)
        for s in range(0, h * w, chunk):
            d = (vy[i, s:s + chunk, None] - c[None, :, 0]) ** 2 + \
                (vx[i, s:s + chunk, None] - c[None, :, 1]) ** 2
            labels[i, s:s + chunk] = d.argmin(1) + 1
    return labels.view(b, h, w) * foreground.long()


def panoptic_labels(semantic: Tensor, instances: Tensor, thing_classes: Sequence[int],
                    label_divisor: int = 1000) -> Tensor:
    """``class * label_divisor + instance_id`` for thing classes, ``class * label_divisor`` for
    stuff (Cityscapes/COCO panoptic convention); background class 0 stays 0."""
    things = torch.isin(semantic, torch.as_tensor(list(thing_classes), device=semantic.device))
    return semantic * label_divisor + torch.where(things, instances, 0)
