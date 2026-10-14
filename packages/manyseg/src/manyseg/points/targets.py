"""Targets for point (centre heatmap + offset) and density heads, in pure torch.

Coordinates are ``(y, x)`` in pixels, pixel centres at integer positions.

* :func:`gaussian_heatmap`: per image, the max over points of ``exp(-r^2 / 2 sigma^2)`` (peak 1 at
  a point lying on a pixel centre). ``sigma`` is a float or a per-point tensor.
* :func:`heatmap_from_instances`: CenterNet-style: centres are instance centroids **rounded to the
  nearest pixel** (a half-pixel centroid would otherwise give a two-pixel plateau and duplicate
  peaks); ``sigma`` fixed or ``sigma_scale * sqrt(area)``.
* :func:`offset_targets`: for every instance pixel, ``centroid - pixel`` (exact, unrounded) as
  ``(dy, dx)``; zero on background (Panoptic-DeepLab).
* :func:`density_map`: one Gaussian per point, each normalised to sum to exactly 1 inside the
  image (truncated at borders), so ``density.sum() == number of points``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

_CHUNK = 64  # points per chunk when rendering Gaussians


def _grid(h: int, w: int, device) -> tuple[Tensor, Tensor]:
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    return ys, xs


def _gaussians(pts: Tensor, sigma: Tensor, h: int, w: int) -> Tensor:
    """(N, H, W) unnormalised Gaussians (separable)."""
    ys, xs = _grid(h, w, pts.device)
    gy = torch.exp(-((ys[None] - pts[:, :1]) ** 2) / (2 * sigma[:, None] ** 2))  # (N, H)
    gx = torch.exp(-((xs[None] - pts[:, 1:]) ** 2) / (2 * sigma[:, None] ** 2))  # (N, W)
    return gy[:, :, None] * gx[:, None, :]


def _sigmas(sigma, n: int, device) -> Tensor:
    s = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    return s.expand(n).clone() if s.dim() == 0 else s


@torch.no_grad()
def gaussian_heatmap(points: Sequence[Tensor], size: tuple[int, int], sigma=2.0) -> Tensor:
    """Centre heatmap ``(B, 1, H, W)`` from per-image point lists ``(N_i, 2)``."""
    h, w = size
    out = []
    for pts in points:
        pts = pts.float()
        hm = torch.zeros(h, w, device=pts.device)
        s = _sigmas(sigma, len(pts), pts.device)
        for i in range(0, len(pts), _CHUNK):
            hm = torch.maximum(hm, _gaussians(pts[i:i + _CHUNK], s[i:i + _CHUNK], h, w).amax(0))
        out.append(hm)
    return torch.stack(out)[:, None]


@torch.no_grad()
def density_map(points: Sequence[Tensor], size: tuple[int, int], sigma=3.0) -> Tensor:
    """Density ``(B, 1, H, W)`` whose integral equals the number of points (mass preserved)."""
    h, w = size
    out = []
    for pts in points:
        pts = pts.float()
        d = torch.zeros(h, w, device=pts.device)
        s = _sigmas(sigma, len(pts), pts.device)
        for i in range(0, len(pts), _CHUNK):
            g = _gaussians(pts[i:i + _CHUNK], s[i:i + _CHUNK], h, w)
            d = d + (g / g.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)).sum(0)
        out.append(d)
    return torch.stack(out)[:, None]


@torch.no_grad()
def centers_from_instances(instance_mask: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Per image: (centroids ``(K, 2)`` float ``(y, x)``, instance ids ``(K,)``), ids ascending."""
    m = instance_mask.long()
    if m.dim() == 2:
        m = m[None]
    h, w = m.shape[-2:]
    ys, xs = _grid(h, w, m.device)
    res = []
    for mi in m:
        ids = torch.unique(mi)
        ids = ids[ids > 0]
        onehot = (mi[None] == ids[:, None, None]).float()  # (K, H, W)
        area = onehot.sum((1, 2)).clamp(min=1)
        cy = (onehot * ys[None, :, None]).sum((1, 2)) / area
        cx = (onehot * xs[None, None, :]).sum((1, 2)) / area
        res.append((torch.stack([cy, cx], 1), ids))
    return res


@torch.no_grad()
def heatmap_from_instances(instance_mask: Tensor, sigma: float | None = 2.0,
                           sigma_scale: float | None = None, min_sigma: float = 1.0) -> Tensor:
    """Centre heatmap from an instance mask (centres = centroids rounded to pixels).

    Use ``sigma`` for a fixed width or ``sigma_scale`` for ``max(min_sigma, sigma_scale *
    sqrt(area))``.
    """
    m = instance_mask.long()
    if m.dim() == 2:
        m = m[None]
    pts, sig = [], []
    for (c, ids), mi in zip(centers_from_instances(m), m):
        pts.append(torch.floor(c + 0.5))
        if sigma_scale is not None:
            area = (mi[None] == ids[:, None, None]).sum((1, 2)).float()
            sig.append((sigma_scale * area.sqrt()).clamp(min=min_sigma))
        else:
            sig.append(torch.full((len(ids),), float(sigma), device=m.device))
    h, w = m.shape[-2:]
    return torch.stack([gaussian_heatmap([p], (h, w), s)[0] for p, s in zip(pts, sig)])


@torch.no_grad()
def offset_targets(instance_mask: Tensor) -> Tensor:
    """``(B, 2, H, W)``: ``centroid - pixel`` (dy, dx) on instance pixels, 0 on background."""
    m = instance_mask.long()
    if m.dim() == 2:
        m = m[None]
    b, h, w = m.shape
    ys, xs = _grid(h, w, m.device)
    out = torch.zeros(b, 2, h, w, device=m.device)
    for i, (c, ids) in enumerate(centers_from_instances(m)):
        if len(ids) == 0:
            continue
        lut = torch.zeros(int(m[i].max()) + 1, 2, device=m.device)
        lut[ids] = c
        cen = lut[m[i]]  # (H, W, 2)
        fg = (m[i] > 0).float()
        out[i, 0] = (cen[..., 0] - ys[:, None]) * fg
        out[i, 1] = (cen[..., 1] - xs[None, :]) * fg
    return out
