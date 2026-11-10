"""Targets for point (centre heatmap + offset) and density heads, in pure torch.

Coordinates are ``(y, x)`` in pixels, pixel centres at integer positions.

* :func:`gaussian_heatmap`: per image, the max over points of ``exp(-r^2 / 2 sigma^2)`` (peak 1 at
  a point lying on a pixel centre). ``sigma`` is a float, a per-point tensor (one image) or a
  list of per-image per-point tensors (as :func:`points_from_boxes` returns).
* :func:`heatmap_from_instances`: CenterNet-style: centres are instance centroids **rounded to the
  nearest pixel** (a half-pixel centroid would otherwise give a two-pixel plateau and duplicate
  peaks); ``sigma`` fixed or ``sigma_scale * sqrt(area)``.
* :func:`offset_targets`: for every instance pixel, ``centroid - pixel`` (exact, unrounded) as
  ``(dy, dx)``; zero on background (Panoptic-DeepLab).
* :func:`density_map`: one Gaussian per point, each normalised to sum to exactly 1 inside the
  image (truncated at borders), so ``density.sum() == number of points``.
* :func:`points_from_boxes`: box centres (YOLO, ``xyxy`` or COCO ``xywh`` boxes) as ``(y, x)``
  points, optionally with a per-point sigma from the box size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

_CHUNK = 64  # points per chunk when rendering Gaussians
_AXIS_CHUNK = 1024  # the same for density_map, which keeps only each Gaussian's two axes
_ZERO_BEYOND = 15.0  # sigmas: float32 exp(-r^2 / 2 sigma^2) is exactly 0 further out
_WINDOW_COST = 30  # gaussian_heatmap renders windows if the image has this many times the pixels
_WINDOW_MIN_GPU_PIXELS = 10 ** 8  # ... and on a GPU, if points x image pixels reaches this


def _grid(h: int, w: int, device) -> tuple[Tensor, Tensor]:
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    return ys, xs


def _gaussian_axes(pts: Tensor, sigma: Tensor, h: int, w: int) -> tuple[Tensor, Tensor]:
    """The factors (N, H) and (N, W) of N unnormalised, separable Gaussians."""
    ys, xs = _grid(h, w, pts.device)
    gy = torch.exp(-((ys[None] - pts[:, :1]) ** 2) / (2 * sigma[:, None] ** 2))
    gx = torch.exp(-((xs[None] - pts[:, 1:]) ** 2) / (2 * sigma[:, None] ** 2))
    return gy, gx


def _gaussians(pts: Tensor, sigma: Tensor, h: int, w: int) -> Tensor:
    """(N, H, W) unnormalised Gaussians."""
    gy, gx = _gaussian_axes(pts, sigma, h, w)
    return gy[:, :, None] * gx[:, None, :]


def _stamp_gaussians(hm: Tensor, pts: Tensor, sigma: Tensor, radius: int) -> None:
    """``hm = max(hm, Gaussians)`` in place, computing only ``radius`` pixels around each point
    (at least ``_ZERO_BEYOND * sigma``: everything further out is exactly 0)."""
    h, w = hm.shape
    offset = torch.arange(-radius, radius + 1, device=pts.device)
    iy = pts[:, :1].floor().long() + offset  # (N, K)
    ix = pts[:, 1:].floor().long() + offset
    gy = torch.exp(-((iy.float() - pts[:, :1]) ** 2) / (2 * sigma[:, None] ** 2))
    gx = torch.exp(-((ix.float() - pts[:, 1:]) ** 2) / (2 * sigma[:, None] ** 2))
    inside = (((iy >= 0) & (iy < h))[:, :, None] & ((ix >= 0) & (ix < w))[:, None, :])
    pixel = (iy[:, :, None] * w + ix[:, None, :])[inside]
    hm.view(-1).scatter_reduce_(0, pixel, (gy[:, :, None] * gx[:, None, :])[inside], "amax")


def _max_gaussians(pts: Tensor, sigma: Tensor, h: int, w: int) -> Tensor:
    """(H, W) max over N Gaussians, by windows around the points where those are cheaper than
    whole images: a windowed pixel costs some 12 (CPU) to 25 (GPU) times a whole-image pixel,
    and setting up windows on a GPU costs about what 10^8 of its whole-image pixels do."""
    hm = torch.zeros(h, w, device=pts.device)
    min_pixels = 0 if pts.device.type == "cpu" else _WINDOW_MIN_GPU_PIXELS

    def fits(sig: float) -> bool:
        return (2 * math.ceil(_ZERO_BEYOND * sig) + 1) ** 2 * _WINDOW_COST <= h * w

    # NaN points or sigmas keep propagating through the whole-image path; infinite sigmas stay
    # there too. MPS has no tested scatter_reduce.
    windowed = (len(pts) > 0 and pts.device.type != "mps" and len(pts) * h * w >= min_pixels
                and bool(pts.isfinite().all() and sigma.isfinite().all())
                and fits(float(sigma.min())))
    if not windowed:
        for i in range(0, len(pts), _CHUNK):
            hm = torch.maximum(hm, _gaussians(pts[i:i + _CHUNK], sigma[i:i + _CHUNK], h, w).amax(0))
        return hm
    sigma, order = sigma.sort(descending=True)  # similar sigmas share a window size
    pts, sig = pts[order], sigma.tolist()
    i = 0
    while i < len(pts):
        if fits(sig[i]):
            radius = math.ceil(_ZERO_BEYOND * sig[i])
            n = max(1, _CHUNK * h * w // (8 * (2 * radius + 1) ** 2))  # 8: indices and mask too
            _stamp_gaussians(hm, pts[i:i + n], sigma[i:i + n], radius)
        else:
            n = _CHUNK
            hm = torch.maximum(hm, _gaussians(pts[i:i + n], sigma[i:i + n], h, w).amax(0))
        i += n
    return hm


def _sigmas(sigma, n: int, device) -> Tensor:
    s = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    if not (s > 0).all():
        raise ValueError("sigma must be > 0 (a zero-size box gives sigma 0: use "
                         "points_from_boxes(min_sigma=...))")
    if s.dim() == 0:
        return s.expand(n).clone()
    if s.shape != (n,):
        raise ValueError(f"sigma has {tuple(s.shape)} values for {n} points; give one float, "
                         "one value per point, or a list of per-image tensors")
    return s


def _per_image(sigma, n_images: int) -> list:
    """``sigma`` per image: a list of per-image tensors (e.g. from :func:`points_from_boxes`)
    is used as is; a float or one per-point tensor applies to every image."""
    if isinstance(sigma, (list, tuple)) and sigma and all(isinstance(t, Tensor) for t in sigma):
        if len(sigma) != n_images:
            raise ValueError(f"{len(sigma)} per-image sigma tensors for {n_images} images")
        return list(sigma)
    return [sigma] * n_images


@torch.no_grad()
def gaussian_heatmap(points: Sequence[Tensor], size: tuple[int, int], sigma=2.0) -> Tensor:
    """Centre heatmap ``(B, 1, H, W)`` from per-image point lists ``(N_i, 2)``."""
    h, w = size
    out = []
    for pts, sig in zip(points, _per_image(sigma, len(points))):
        pts = pts.float()
        out.append(_max_gaussians(pts, _sigmas(sig, len(pts), pts.device), h, w))
    return torch.stack(out)[:, None]


@torch.no_grad()
def density_map(points: Sequence[Tensor], size: tuple[int, int], sigma=3.0) -> Tensor:
    """Density ``(B, 1, H, W)`` whose integral equals the number of points (mass preserved)."""
    h, w = size
    out = []
    for pts, sig in zip(points, _per_image(sigma, len(points))):
        pts = pts.float()
        d = torch.zeros(h, w, device=pts.device)
        s = _sigmas(sig, len(pts), pts.device)
        for i in range(0, len(pts), _AXIS_CHUNK):
            gy, gx = _gaussian_axes(pts[i:i + _AXIS_CHUNK], s[i:i + _AXIS_CHUNK], h, w)
            mass = (gy.sum(1) * gx.sum(1)).clamp(min=1e-12)
            d = d + (gy / mass[:, None]).T @ gx  # the summed Gaussians, without an (N, H, W) stack
        out.append(d)
    return torch.stack(out)[:, None]


def _instance_stats(mi: Tensor):
    """(ids, area, centroid (K, 2)) via bincount over compacted ids (no K x H x W tensors)."""
    h, w = mi.shape
    ids, inv = torch.unique(mi.reshape(-1), return_inverse=True)
    ys = torch.arange(h, device=mi.device, dtype=torch.float64).repeat_interleave(w)
    xs = torch.arange(w, device=mi.device, dtype=torch.float64).repeat(h)
    k = len(ids)
    area = torch.bincount(inv, minlength=k).double()
    cy = torch.bincount(inv, weights=ys, minlength=k) / area.clamp(min=1)
    cx = torch.bincount(inv, weights=xs, minlength=k) / area.clamp(min=1)
    keep = ids > 0
    return ids[keep], area[keep].float(), torch.stack([cy, cx], 1)[keep].float()


@torch.no_grad()
def centers_from_instances(instance_mask: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Per image: (centroids ``(K, 2)`` float ``(y, x)``, instance ids ``(K,)``), ids ascending."""
    m = instance_mask.long()
    if m.dim() == 2:
        m = m[None]
    res = []
    for mi in m:
        ids, _, c = _instance_stats(mi)
        res.append((c, ids))
    return res


@torch.no_grad()
def heatmap_from_instances(instance_mask: Tensor, sigma: float | None = 2.0,
                           sigma_scale: float | None = None, min_sigma: float = 1.0) -> Tensor:
    """Centre heatmap from an instance mask (centres = centroids rounded to pixels).

    Use ``sigma`` for a fixed width or ``sigma_scale`` for ``max(min_sigma, sigma_scale *
    sqrt(area))`` (``sigma_scale`` takes precedence); one of them is required.
    """
    if sigma is None and sigma_scale is None:
        raise ValueError("heatmap_from_instances needs a fixed sigma or a sigma_scale")
    m = instance_mask.long()
    if m.dim() == 2:
        m = m[None]
    pts, sig = [], []
    for mi in m:
        ids, area, c = _instance_stats(mi)
        pts.append(torch.floor(c + 0.5))
        if sigma_scale is not None:
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
        u, inv = torch.unique(m[i], return_inverse=True)  # compact ids, not a max-id table
        lut = torch.zeros(len(u), 2, device=m.device)
        lut[torch.searchsorted(u, ids)] = c
        cen = lut[inv]  # (H, W, 2)
        fg = (m[i] > 0).float()
        out[i, 0] = (cen[..., 0] - ys[:, None]) * fg
        out[i, 1] = (cen[..., 1] - xs[None, :]) * fg
    return out


BOX_FORMATS = ("yolo", "xyxy", "xywh")


def points_from_boxes(boxes: Sequence[Tensor], size: tuple[int, int] | None = None,
                      fmt: str = "yolo", sigma_from_size: float | None = None,
                      min_sigma: float = 1.0):
    """Box centres as point targets: per image ``(N, 2)`` float ``(y, x)``, **y first** (box
    formats are x first).

    ``boxes``: per image ``(N, 4)``, in ``fmt``:

    * ``"yolo"``: ``(cx, cy, w, h)`` normalised to [0, 1] (drop the class column of a YOLO
      label file first); needs ``size=(H, W)``;
    * ``"xyxy"``: ``(x1, y1, x2, y2)`` in pixels;
    * ``"xywh"``: COCO ``(x, y, w, h)``, top-left corner and size in pixels.

    Box coordinates have pixel edges at integers (a box over pixels 0-9 spans 0 to 10); points
    have pixel centres at integers, like every target here, so a centre ``c`` becomes
    ``c - 0.5``. ``sigma_from_size``: also return a per-point sigma ``max(min_sigma, k *
    sqrt(w * h))`` (pixels; the floor keeps zero-size boxes valid), as ``(points, sigmas)``,
    ready for :func:`gaussian_heatmap` / :func:`density_map` ``sigma=``. Points are not
    rounded: round them (``torch.floor(p + 0.5)``) for a heatmap whose peaks must be exactly 1
    on a pixel.
    """
    if fmt not in BOX_FORMATS:
        raise ValueError(f"fmt must be one of {BOX_FORMATS}, got {fmt!r}")
    if fmt == "yolo" and size is None:
        raise ValueError("fmt='yolo' boxes are normalised: give size=(H, W)")
    pts, sigmas = [], []
    for b in boxes:
        b = torch.as_tensor(b, dtype=torch.float32)
        b = b.reshape(0, 4) if b.numel() == 0 else b
        if b.dim() != 2 or b.shape[1] != 4:
            raise ValueError(f"boxes must be (N, 4), got {tuple(b.shape)}"
                             + (" (drop the YOLO class column)" if b.shape[-1] == 5 else ""))
        if fmt == "yolo":
            h, w = size
            cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
        elif fmt == "xyxy":
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        else:
            cx, cy, bw, bh = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2, b[:, 2], b[:, 3]
        if (bw < 0).any() or (bh < 0).any():
            raise ValueError(f"boxes with negative width or height: are they {fmt!r}?")
        pts.append(torch.stack([cy - 0.5, cx - 0.5], 1))
        if sigma_from_size is not None:
            sigmas.append((sigma_from_size * (bw * bh).sqrt()).clamp(min=min_sigma))
    return (pts, sigmas) if sigma_from_size is not None else pts
