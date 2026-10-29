"""Heatmap peak finding shared by the point and vector decoders (private).

Three steps, each used by more than one caller:

* :func:`maxima_mask`: pixels equal to the maximum of their ``kernel`` x ``kernel``
  neighbourhood and strictly above ``threshold`` (``>``, so a peak exactly at the threshold is
  dropped). numpy ``(H, W)`` goes through ``scipy.ndimage.maximum_filter(mode="nearest")``,
  torch ``(B, H, W)`` through ``max_pool2d`` (padding ``kernel // 2``, odd kernels only).
  Replicated borders and ``-inf`` padding give the same neighbourhood maximum, so for finite
  values and odd kernels the two backends agree; with an even kernel only numpy works (its
  window is off centre, as scipy's). Every pixel of a plateau is kept (ties are not broken).
* :func:`suppress`: greedy suppression of maxima within ``(kernel // 2) * 1.5`` px (Euclidean,
  on pixel indices), strongest first, ties in the order given (stable sort). This is what
  turns a plateau into one peak (one per radius if it is wider).
* :func:`refine`: sub-pixel ``(x, y)`` from a per-axis parabola through the log heatmap.

How the callers use them (each keeps its own behaviour; the experiments' decoder settings
were tuned against it):

* :func:`manyseg.points.find_peaks`: torch :func:`maxima_mask` only, after its optional blur
  and mask. Plateaus give several tied points; points stay integer ``(y, x)``; sorting by
  score (``torch.argsort(descending=True)``, ties in an unspecified order) and ``max_points``
  stay in ``find_peaks``.
* :mod:`manyseg.vectors` (corner peaks in ``polygons_from_corners``, keypoint peaks in
  ``graph_from_road_heads`` with kernel 5, vertex peaks in ``graph_from_tensor``):
  :func:`peak_pixels` (numpy mask + suppression; ``(row, col)`` strongest first) then
  :func:`refine` to ``(x, y)`` pixel-centre coordinates.
* ``RoadBlockDecoder``: :func:`maxima_mask` and :func:`refine` per window (``origin`` and
  ``shape`` make the border rule the scene's), :func:`suppress` once on the merged maxima in
  scene raster order, which equals :func:`peak_pixels` on the whole scene.
"""

from __future__ import annotations

import numpy as np


def maxima_mask(hm, threshold: float, kernel: int):
    """Bool mask of local maxima above ``threshold``: ``hm == max over the kernel window`` and
    ``hm > threshold``. numpy ``(H, W)`` (scipy max filter, ``mode="nearest"``) or torch
    ``(B, H, W)`` (``max_pool2d``, odd ``kernel``); returns the same type."""
    if isinstance(hm, np.ndarray):
        from scipy import ndimage

        return (hm == ndimage.maximum_filter(hm, size=kernel, mode="nearest")) & (hm > threshold)
    import torch.nn.functional as F

    if kernel % 2 != 1:
        raise ValueError(f"kernel must be odd, got {kernel}")
    pooled = F.max_pool2d(hm[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    return (hm == pooled) & (hm > threshold)


def suppress(pts: np.ndarray, vals: np.ndarray, kernel: int) -> np.ndarray:
    """Indices into ``pts`` ``(N, 2)`` of the peaks kept by greedy suppression: strongest
    ``vals`` first (stable, so ties keep the order of ``pts``), each kept peak removes later
    ones within ``(kernel // 2) * 1.5`` px. Returned strongest first."""
    if len(pts) < 2:
        return np.arange(len(pts))
    from scipy.spatial import cKDTree

    order = np.argsort(-vals, kind="stable")
    q = pts[order]
    tree = cKDTree(q)
    alive = np.ones(len(q), bool)
    r = (kernel // 2) * 1.5
    for i in range(len(q)):
        if alive[i]:
            for j in tree.query_ball_point(q[i], r):
                if j > i:
                    alive[j] = False
    return order[alive]


def peak_pixels(hm: np.ndarray, threshold: float, kernel: int) -> np.ndarray:
    """``(row, col)`` peaks of a numpy ``(H, W)`` heatmap: :func:`maxima_mask` in raster
    order, then :func:`suppress`; strongest first. A plateau within the suppression radius
    gives one peak, its first pixel in raster order (a wider one, one per radius)."""
    keep = maxima_mask(hm, threshold, kernel)
    pts = np.argwhere(keep)
    return pts[suppress(pts, hm[keep], kernel)]


def refine(hm: np.ndarray, pix: np.ndarray, origin=(0, 0), shape=None) -> np.ndarray:
    """Sub-pixel ``(x, y)`` of ``(row, col)`` pixels ``pix`` of ``hm``: per-axis parabola
    through the log heatmap (clipped at 1e-4; exact for a Gaussian), offset clipped to +-1 px,
    no offset on an axis whose curvature is not negative or whose neighbour is outside the
    image. Pixel centres are at ``+0.5``.

    ``hm`` may be a window of a larger image: ``origin`` is the window's top-left ``(row, col)``
    in it and ``shape`` its ``(H, W)`` (default ``hm.shape``). The border rule then uses the
    image, results are in image coordinates, and pixels must not be on the window's inner edge
    (their neighbours would be clipped to the window)."""
    if len(pix) == 0:
        return np.zeros((0, 2))
    H, W = hm.shape if shape is None else shape
    h, w = hm.shape
    lg = np.log(np.clip(hm, 1e-4, None))
    r, c = pix[:, 0], pix[:, 1]
    R, C = r + origin[0], c + origin[1]

    def offset(lo, mid, hi):
        den = lo - 2 * mid + hi
        ok = den < -1e-9
        return np.clip(np.where(ok, 0.5 * (lo - hi) / np.where(ok, den, -1), 0), -1, 1)

    inx, iny = (C > 0) & (C < W - 1), (R > 0) & (R < H - 1)
    dx = np.where(
        inx, offset(lg[r, np.maximum(c - 1, 0)], lg[r, c], lg[r, np.minimum(c + 1, w - 1)]), 0
    )
    dy = np.where(
        iny, offset(lg[np.maximum(r - 1, 0), c], lg[r, c], lg[np.minimum(r + 1, h - 1), c]), 0
    )
    return np.stack([C + 0.5 + dx, R + 0.5 + dy], 1)
