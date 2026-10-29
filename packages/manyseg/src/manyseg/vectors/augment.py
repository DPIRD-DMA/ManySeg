"""Dihedral augmentation (flips, quarter turns, transposes) for rasters, double-angle direction
fields and the ``(x, y)`` geometry they were encoded from.

The 8 transforms are numbered ``d = 0..7``: ``d % 4`` quarter turns counter-clockwise as
displayed (``np.rot90`` on the last two axes), then, for ``d >= 4``, a horizontal flip (columns
reversed). Every flip, rotation and transpose is one of them (:data:`HFLIP`, :data:`VFLIP`,
:data:`TRANSPOSE`). They move whole pixels, so heatmaps, line widths and masks stay exact: a
transformed target equals the target encoded from the transformed geometry.

A direction field stores an orientation ``t`` (mod 180°) as ``(cos 2t, sin 2t)`` channel pairs.
Moving pixels is not enough for it: a quarter turn adds 90° to ``t`` (both components negate),
a horizontal or vertical flip maps ``t`` to ``-t`` (``sin 2t`` negates), a transpose maps ``t``
to ``90° - t`` (``cos 2t`` negates). :func:`dihedral` applies that with ``direction=True``.
"""

from __future__ import annotations

import numpy as np

#: Transform numbers of a horizontal flip, a vertical flip and a transpose (``x <-> y``).
HFLIP, VFLIP, TRANSPOSE = 4, 6, 7


def _parts(d: int) -> tuple[int, bool]:
    if not 0 <= int(d) < 8:
        raise ValueError(f"dihedral transform must be 0..7, got {d!r}")
    return int(d) % 4, int(d) >= 4


def dihedral(a, d: int, *, direction: bool = False):
    """Transform ``d`` of an array or tensor ``(..., H, W)`` (numpy or torch, same type back).

    ``direction=True``: ``a`` is a double-angle field ``(..., 2k, H, W)`` of ``k``
    ``(cos 2t, sin 2t)`` pairs, whose values are transformed as well as moved.
    """
    k, flip = _parts(d)
    if hasattr(a, "detach"):  # torch
        import torch

        out = torch.rot90(a, k, dims=(-2, -1)) if k else a
        out = torch.flip(out, dims=(-1,)) if flip else out
        if direction:
            out = out * _signs(k, flip, a.shape[-3]).to(device=a.device, dtype=a.dtype)
        return out.contiguous()
    out = np.rot90(np.asarray(a), k, axes=(-2, -1))
    out = out[..., ::-1] if flip else out
    if direction:
        out = out * _signs(k, flip, out.shape[-3]).numpy().astype(out.dtype)
    return np.ascontiguousarray(out)


def _signs(k: int, flip: bool, channels: int):
    """Per-channel signs ``(channels, 1, 1)`` of transform ``(k, flip)`` on (cos, sin) pairs."""
    import torch

    if channels % 2:
        raise ValueError(f"a direction field has (cos 2t, sin 2t) pairs, got {channels} channels")
    s = torch.tensor([-1.0 if k % 2 else 1.0] * 2)
    if flip:
        s[1] = -s[1]
    return s.repeat(channels // 2).view(-1, 1, 1)


def dihedral_xy(xy, d: int, shape: tuple[int, int]) -> np.ndarray:
    """Transform ``d`` of ``(x, y)`` coordinates ``(..., 2)`` in the pixel frame of a raster of
    ``shape`` ``(H, W)`` (GIS convention, pixel corners at integers), matching
    :func:`dihedral` on that raster."""
    k, flip = _parts(d)
    p = np.array(xy, np.float64)
    h, w = shape
    for _ in range(k):  # np.rot90: new[r, c] = old[c, w - 1 - r], so (x, y) -> (y, w - x)
        p = np.stack([p[..., 1], w - p[..., 0]], -1)
        h, w = w, h
    if flip:
        p[..., 0] = w - p[..., 0]
    return p


def dihedral_geometry(geom, d: int, shape: tuple[int, int]):
    """Transform ``d`` of a shapely geometry (pixel coordinates) in a raster of ``shape``."""
    import shapely

    return shapely.transform(geom, lambda c: dihedral_xy(c, d, shape))


def dihedral_shape(shape: tuple[int, int], d: int) -> tuple[int, int]:
    """``(H, W)`` after transform ``d`` (odd quarter turns swap them)."""
    k, _ = _parts(d)
    return (shape[1], shape[0]) if k % 2 else (shape[0], shape[1])
