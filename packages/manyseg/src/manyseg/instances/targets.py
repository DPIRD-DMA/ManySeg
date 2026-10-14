"""Targets for boundary + distance instance heads (ResUNet-a / FracTAL style), in pure torch.

Everything runs on whatever device the instance mask lives on, so targets can be generated
on the GPU inside a data loader.

Definitions, for an integer instance-ID mask ``m`` (0 = background):

* **edge pixels**: instance pixels with a 4-neighbour of a different id (background or another
  instance). Pixels outside the tile are treated as continuing the tile (replicate padding),
  so an instance cut by the tile border has no boundary along the border.
* **distance** ``d(p)`` for an instance pixel ``p``: ``1 + `` Euclidean distance to the nearest
  edge pixel (an edge pixel has ``d = 1``). ``"normalized"`` divides by the instance's
  maximum, giving ``(0, 1]`` per instance. Background is 0. A tile filled by a single instance
  (no edge pixels anywhere) uses ``1 +`` distance to the tile border instead.
* **boundary** with ``width`` ``w``:
  * ``"inner"`` (default): instance pixels with ``d <= w`` (a band inside every instance; where
    two instances touch, the band has total width ``2w``, ``w`` on each side);
  * ``"outer"``: background pixels within distance ``w`` of an instance;
  * ``"both"``: the union.
* **extent**: ``m > 0`` as float ``(B, 1, H, W)``, or the semantic class map ``(B, H, W)`` if
  ``semantic`` is given.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_MAX_ELEMS = 2**26  # cap on the (rows x rows x cols) intermediate per chunk


def _min_plus_1d(f: Tensor) -> Tensor:
    """Exact 1D squared distance transform along dim -2: ``g[i, j] = min_k (i-k)^2 + f[k, j]``.

    Brute force over ``k`` (O(H^2 W) time), tiled over query rows ``i``, source rows ``k`` and
    columns ``j`` so no intermediate exceeds ``_MAX_ELEMS`` elements; vectorised on device.
    """
    b, h, w = f.shape
    idx = torch.arange(h, device=f.device, dtype=f.dtype)
    out = torch.full_like(f, float("inf"))
    # tile sizes: B * rq * rk * c <= _MAX_ELEMS
    c = max(1, min(w, _MAX_ELEMS // max(1, b * h * h)))
    rq = rk = h
    while b * rq * rk * c > _MAX_ELEMS and (rq > 1 or rk > 1):
        if rq >= rk:
            rq = max(1, rq // 2)
        else:
            rk = max(1, rk // 2)
    for j in range(0, w, c):
        fj = f[:, :, j:j + c]
        for i0 in range(0, h, rq):
            qi = idx[i0:i0 + rq]
            best = out[:, i0:i0 + rq, j:j + c]
            for k0 in range(0, h, rk):
                d2 = (qi[:, None] - idx[None, k0:k0 + rk]) ** 2  # (rq, rk)
                cand = (d2[None, :, :, None] + fj[:, None, k0:k0 + rk, :]).amin(dim=2)
                best = torch.minimum(best, cand)
            out[:, i0:i0 + rq, j:j + c] = best
    return out


def edt(features: Tensor) -> Tensor:
    """Exact Euclidean distance from each pixel to the nearest ``True`` pixel.

    ``features``: bool ``(B, H, W)``. Returns float32 ``(B, H, W)``; ``inf`` where an image has no
    features. Separable (rows then columns) brute-force lower envelope: exact, GPU-friendly,
    O(H^2 W + W^2 H) per image.
    """
    if features.dim() != 3:
        raise ValueError("features must be (B, H, W)")
    f = torch.where(features, 0.0, float("inf")).float()
    g = _min_plus_1d(f)                                   # along H
    g = _min_plus_1d(g.transpose(1, 2)).transpose(1, 2)   # along W
    return g.sqrt()


def edge_pixels(m: Tensor) -> Tensor:
    """Instance pixels with a 4-neighbour of a different id (replicate padding at tile edges)."""
    p = F.pad(m[:, None].float(), (1, 1, 1, 1), mode="replicate")[:, 0].long()
    c = p[:, 1:-1, 1:-1]
    diff = ((p[:, :-2, 1:-1] != c) | (p[:, 2:, 1:-1] != c) | (p[:, 1:-1, :-2] != c)
            | (p[:, 1:-1, 2:] != c))
    return diff & (c > 0)


def _per_instance_max(values: Tensor, m: Tensor) -> Tensor:
    """For each pixel, the max of ``values`` over its instance (batched, via scatter_reduce)."""
    b = m.shape[0]
    n = int(m.max()) + 1 if m.numel() else 1
    keys = (m + torch.arange(b, device=m.device)[:, None, None] * n).reshape(-1)
    mx = torch.zeros(b * n, device=values.device, dtype=values.dtype)
    mx = mx.scatter_reduce(0, keys, values.reshape(-1), reduce="amax", include_self=True)
    return mx[keys].reshape(m.shape)


def _border_distance(m: Tensor) -> Tensor:
    """1 + distance to the tile border (used for instances with no edge pixels)."""
    _, h, w = m.shape
    ys = torch.arange(h, device=m.device, dtype=torch.float32)
    xs = torch.arange(w, device=m.device, dtype=torch.float32)
    dy = torch.minimum(ys, h - 1 - ys)[:, None]
    dx = torch.minimum(xs, w - 1 - xs)[None, :]
    return (torch.minimum(dy, dx) + 1.0).expand(m.shape)


@torch.no_grad()
def instance_distance(m: Tensor, normalized: bool = True) -> Tensor:
    """Distance target ``(B, H, W)`` (see module docstring).

    An instance with no edge pixels (it fills the whole tile) has no defined edge distance; it
    uses the distance to the tile border instead, so it still gets a peaked distance map.
    """
    edges = edge_pixels(m)
    d = edt(edges) + 1.0
    d = torch.where(torch.isinf(d), _border_distance(m), d)
    d = torch.where(m > 0, d, torch.zeros((), device=m.device))
    if normalized:
        d = d / _per_instance_max(d, m).clamp(min=1e-6)
    return d


@torch.no_grad()
def instance_targets(
    instance_mask: Tensor,
    boundary_width: float = 1,
    boundary_mode: str = "inner",
    distance: str = "normalized",
    semantic: Tensor | None = None,
) -> dict[str, Tensor]:
    """Extent, boundary and distance targets from an instance-ID mask ``(H, W)`` or ``(B, H, W)``.

    Returns ``{"extent", "boundary", "distance"}``: ``boundary`` and ``distance`` are float
    ``(B, 1, H, W)``; ``extent`` is float ``(B, 1, H, W)`` (binary) or long ``(B, H, W)`` class ids
    when ``semantic`` is given. See the module docstring for exact definitions.
    """
    if boundary_mode not in ("inner", "outer", "both"):
        raise ValueError(f"boundary_mode must be inner|outer|both, got {boundary_mode!r}")
    if distance not in ("normalized", "absolute"):
        raise ValueError(f"distance must be 'normalized' or 'absolute', got {distance!r}")
    m = instance_mask.long()
    if m.dim() == 2:
        m = m[None]
    if semantic is not None and semantic.dim() == 2:
        semantic = semantic[None]
    d_abs = instance_distance(m, normalized=False)
    no_edges = ~edge_pixels(m).flatten(1).any(1)[:, None, None]  # whole-tile instance
    fg = m > 0
    boundary = torch.zeros_like(fg)
    if boundary_mode in ("inner", "both"):
        boundary |= fg & (d_abs <= boundary_width) & ~no_edges
    if boundary_mode in ("outer", "both"):
        boundary |= ~fg & (edt(fg) <= boundary_width)
    d = d_abs / _per_instance_max(d_abs, m).clamp(min=1e-6) if distance == "normalized" else d_abs
    extent = semantic.long() if semantic is not None else fg.float()[:, None]
    return {"extent": extent, "boundary": boundary.float()[:, None], "distance": d[:, None]}
