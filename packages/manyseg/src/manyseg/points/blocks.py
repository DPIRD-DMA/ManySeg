"""Block-wise point and count decoding of large scenes (stitched rasters, core + halo blocks).

Follows the block contract of :mod:`manyseg.vectors.blocks` (as ManyPred's block
post-processing calls it): a decoder is called once per block as
``decoder(fields, block, scene)``, with ``fields`` name -> numpy ``(C, h, w)`` cropped to
``block.window``, and ``decoder.combine(results, scene)`` joins the per-block results in grid
order. ``block`` and ``scene`` are duck-typed: ``block.core`` / ``block.window`` are ``(top,
bottom, left, right)``, half-open, in scene pixels (cores partition the scene); ``scene.height``,
``scene.width`` and optionally ``scene.transform`` / ``scene.crs``. Both decoders are plain
dataclasses, so they and their results pickle (process pools).

Fields are the stitched head outputs on the scene grid (full-resolution heads; a reduced-
resolution head's raster has its own grid and transform).

:class:`PointBlockDecoder` equals a whole-scene :func:`find_peaks` exactly (same peaks, same
scores, ties and plateaus included) when the halo is at least :attr:`PointBlockDecoder.min_halo`:
a peak at pixel ``p`` depends only on the (blurred) heatmap within ``kernel // 2`` of ``p``, and
a blurred value only on the heatmap within the blur radius. Each block keeps the peaks whose
pixel lies in its core, so no peak is kept twice or lost at a seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .geo import points_to_xy
from .postprocess import blur_radius, density_value_scale, find_peaks

_UNSET: Any = object()  # "argument not given" (``None`` means pixel coordinates)


@dataclass(eq=False)
class ScenePoints:
    """Points of a scene (or of one block's core).

    * ``yx``: ``(N, 2)`` float64 ``(y, x)`` in scene pixels, :mod:`manyseg.points`
      convention (pixel centres at integers): the peak pixel, plus the predicted ``(dy, dx)``
      offset there if the decoder reads an offset head.
    * ``scores``: ``(N,)`` float32 heatmap values at the peaks (blurred if ``smooth_sigma``).
    * ``pixels``: ``(N, 2)`` int64 ``(row, col)`` of the peak pixel (which core kept it).
    * ``transform``, ``crs``: the scene's, when known.
    * :attr:`xy`: ``(N, 2)`` pixel ``(x, y)`` with corners at integers, for GIS.

    Sorted by score, descending; ties in raster order of ``pixels`` (row, then column).
    """

    yx: np.ndarray
    scores: np.ndarray
    pixels: np.ndarray
    transform: Any = None
    crs: Any = None

    def __len__(self) -> int:
        return len(self.yx)

    @property
    def xy(self) -> np.ndarray:
        """``(N, 2)`` float64 pixel ``(x, y)`` with pixel corners at integers
        (:mod:`manyseg.vectors` convention: ``yx[:, ::-1] + 0.5``)."""
        return points_to_xy(self.yx)

    def to_geojson(self, transform=_UNSET, properties: dict | None = None) -> dict:
        """GeoJSON ``Point`` features through ``transform``: not given, the scene's (pixel
        coordinates when the scene has none); ``None``, pixel ``(x, y)`` coordinates; else an
        affine in rasterio order. Each feature has a ``score`` property plus any columns in
        ``properties`` (``score`` is reserved: a ``properties["score"]`` raises). The scene's
        ``crs`` is written as a ``name`` member when the coordinates are in world units, as
        :func:`~manyseg.vectors.records_to_geojson` does. Needs the ``vectors`` extra."""
        from ..vectors.polygons import to_geojson

        if properties and "score" in properties:
            raise ValueError("'score' is a reserved property (the peak score); rename yours")
        t = self.transform if transform is _UNSET else transform
        fc = to_geojson(self.xy, t, {"score": self.scores, **(properties or {})})
        if self.crs is not None and t is not None:  # pixel coordinates are in no CRS
            fc["crs"] = {"type": "name", "properties": {"name": str(self.crs)}}
        return fc

    def write(self, path, *, layer: str | None = None):
        """Write the points to ``path`` (``.geojson``, or e.g. ``.gpkg`` with geopandas) as
        ``Point`` features with a ``score`` property, through the scene's ``transform`` and
        naming its ``crs`` (pixel coordinates and no CRS without a transform), as
        :meth:`to_geojson` (:func:`~manyseg.vectors.write_records`). Needs the ``vectors``
        extra. Returns the path."""
        from ..vectors.blocks import write_records

        return write_records(self, path, layer=layer)


def _bounds(b) -> tuple[int, int, int, int]:
    """``(top, bottom, left, right)`` from the contract's 4-tuple or a ``(row slice, col slice)``
    pair (the older ``manyseg.vectors.blocks.Block``)."""
    if len(b) == 2 and all(isinstance(v, slice) for v in b):
        b = (b[0].start, b[0].stop, b[1].start, b[1].stop)
    t, bo, le, r = (int(v) for v in b)
    if not (t <= bo and le <= r):
        raise ValueError(f"expected (top, bottom, left, right), got {tuple(b)}")
    return t, bo, le, r


def _layout(block, scene, reach: int):
    """(core, window, core in window coordinates), after checking that the halo covers
    ``reach`` (the decoder's ``min_halo``) on every side that is not the scene edge."""
    core, win = _bounds(block.core), _bounds(block.window)
    if not (win[0] <= core[0] and core[1] <= win[1] and win[2] <= core[2] and core[3] <= win[3]):
        raise ValueError(f"block core {core} is not inside its window {win}")
    h, w = getattr(scene, "height", None), getattr(scene, "width", None)
    if h is not None and w is not None:
        edges = (win[0] == 0, win[1] == h, win[2] == 0, win[3] == w)
    else:
        edges = tuple(bool(e) for e in block.scene_edges)
    margins = (core[0] - win[0], win[1] - core[1], core[2] - win[2], win[3] - core[3])
    for side, m, e in zip(("top", "bottom", "left", "right"), margins, edges):
        if not e and m < reach:
            raise ValueError(f"halo too small: block {getattr(block, 'index', '?')} has {m} px "
                             f"of halo on its {side} side, the decoder needs >= {reach} "
                             "(kernel // 2 + blur radius)")
    inner = (core[0] - win[0], core[1] - win[0], core[2] - win[2], core[3] - win[2])
    return core, win, inner


def _field(fields, name: str, channels: int, win) -> np.ndarray:
    if name not in fields:
        raise KeyError(f"field {name!r} missing (have {sorted(fields)})")
    a = np.asarray(fields[name])
    a = a[None] if a.ndim == 2 else a
    shape = (win[1] - win[0], win[3] - win[2])
    if a.ndim != 3 or a.shape[0] != channels or a.shape[1:] != shape:
        raise ValueError(f"field {name!r}: expected ({channels}, {shape[0]}, {shape[1]}) for "
                         f"window {win}, got {a.shape}")
    return a


@dataclass
class PointBlockDecoder:
    """Peaks of a point heatmap, block by block: :func:`find_peaks` on each window, keeping the
    peaks whose pixel lies in the block's core.

    Args:
        field: the heatmap field (:func:`point_heads` names it ``"heatmap"``), 1 channel,
            probabilities (or logits with ``logits=True``).
        threshold: peak threshold, as :func:`find_peaks`.
        kernel: odd non-maximum suppression window (heatmap px), as :func:`find_peaks`.
        smooth_sigma: Gaussian blur of the heatmap before peak finding (0: none), as
            :func:`find_peaks`.
        max_points: keep the best ``max_points`` of the whole scene (applied in
            :meth:`combine`, after sorting).
        offsets: the offset field (``"offset"`` in :func:`point_heads` ``offsets=True``), 2
            channels ``(dy, dx)`` in pixels: added to each peak for sub-pixel points.
            ``None``: points are the peak pixels.
        mask: a field (1 channel) whose value must exceed ``mask_threshold`` at a peak (e.g. a
            foreground probability), as :func:`find_peaks` ``mask=``.
        logits: the heatmap and the ``mask`` field are logits: a sigmoid is applied to both
            first (``mask_threshold`` stays a probability; ``offsets`` are values).

    The block grid's halo must be at least :attr:`min_halo`; a block with less halo on a side that
    is not the scene edge raises. Equals ``find_peaks(heatmap, ...)`` on the whole scene, float32
    (fields are converted to float32 first, so a float16 raster's threshold comparison is done
    in float32). Result: :class:`ScenePoints`.
    """

    field: str = "heatmap"
    threshold: float = 0.5
    kernel: int = 3
    smooth_sigma: float = 0.0
    max_points: int | None = None
    offsets: str | None = None
    mask: str | None = None
    mask_threshold: float = 0.5
    logits: bool = False

    def __post_init__(self):
        if self.kernel % 2 != 1 or self.kernel < 1:
            raise ValueError(f"kernel must be odd and positive, got {self.kernel}")
        if self.smooth_sigma < 0:
            raise ValueError(f"smooth_sigma must be >= 0, got {self.smooth_sigma}")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError(f"max_points must be >= 0, got {self.max_points}")

    @property
    def min_halo(self) -> int:
        """The smallest halo (px) that gives whole-scene results: ``kernel // 2`` plus the blur
        radius ``max(1, ceil(3 * smooth_sigma))`` when ``smooth_sigma`` > 0."""
        return self.kernel // 2 + blur_radius(self.smooth_sigma)

    def __call__(self, fields, block, scene) -> ScenePoints:
        """Peaks of one block's window whose pixel lies in the block's core."""
        core, win, (it, ib, il, ir) = _layout(block, scene, self.min_halo)
        hm = torch.from_numpy(np.array(_field(fields, self.field, 1, win), np.float32))
        if self.logits:
            hm = torch.sigmoid(hm)
        m = None
        if self.mask is not None:
            m = np.asarray(_field(fields, self.mask, 1, win))
            if self.logits:
                m = torch.sigmoid(torch.from_numpy(np.array(m, np.float32))).numpy()
            m = torch.from_numpy(m > self.mask_threshold)
        ((pts, scores),) = find_peaks(hm[None], self.threshold, kernel=self.kernel,
                                      smooth_sigma=self.smooth_sigma, mask=m)
        pts, scores = pts.numpy(), scores.numpy()
        keep = (pts[:, 0] >= it) & (pts[:, 0] < ib) & (pts[:, 1] >= il) & (pts[:, 1] < ir)
        pts, scores = pts[keep], scores[keep]
        points = pts.astype(np.float64)
        if self.offsets is not None:
            off = _field(fields, self.offsets, 2, win)
            points += off[:, pts[:, 0], pts[:, 1]].T.astype(np.float64)
        origin = np.array([win[0], win[2]])
        return ScenePoints(points + origin, scores.astype(np.float32),
                           (pts + origin).astype(np.int64))

    def combine(self, results: list, scene) -> ScenePoints:
        """All blocks' points in one :class:`ScenePoints`: score descending, ties in raster
        order, cut to ``max_points``, with the scene's ``transform`` / ``crs``."""
        res = list(results)
        yx = np.concatenate([r.yx for r in res] or [np.zeros((0, 2))]).reshape(-1, 2)
        scores = np.concatenate([r.scores for r in res] or [np.zeros(0, np.float32)])
        pixels = np.concatenate([r.pixels for r in res] or [np.zeros((0, 2), np.int64)])
        pixels = pixels.reshape(-1, 2).astype(np.int64)
        if len(np.unique(pixels, axis=0)) != len(pixels):
            raise ValueError("a peak pixel was kept by two blocks: the block cores overlap")
        order = np.lexsort((pixels[:, 1], pixels[:, 0], -scores.astype(np.float64)))
        if self.max_points is not None:
            order = order[:self.max_points]
        return ScenePoints(yx[order], scores[order].astype(np.float32), pixels[order],
                           getattr(scene, "transform", None), getattr(scene, "crs", None))


@dataclass(eq=False)
class SceneCount:
    """Object count of a scene from a density head: ``total`` and, per block in grid order,
    its ``cores`` ``(top, bottom, left, right)`` and the count inside each (``counts``)."""

    total: float
    counts: np.ndarray
    cores: list


@dataclass
class DensityBlockDecoder:
    """:func:`count_from_density` block by block: the density summed over each block's core
    (float64), divided by the target's scale. Needs no halo (a sum has no reach).

    Give exactly one of ``sigma`` (the :func:`density_loss` ``sigma``: its default target is
    scaled by ``2 pi sigma^2``) or ``value_scale`` (the loss's ``value_scale``; 1.0 for a plain
    :func:`density_map` target), as :func:`count_from_density`.

    ``mask``: a field (1 channel); only pixels where it exceeds ``mask_threshold`` count, as
    ``count_from_density(density, mask=field > mask_threshold)``. Result: :class:`SceneCount`
    (equal to the whole-scene count up to float rounding; the sum is in float64).

    NaN pixels count as 0, so a scene with nodata (NaN in ManyPred's stitched outputs) still has
    a total. A model that returns NaN where there is data is not detected here."""

    field: str = "density"
    sigma: float | None = None
    value_scale: float | None = None
    mask: str | None = None
    mask_threshold: float = 0.5

    min_halo = 0

    def __post_init__(self):
        density_value_scale(self.sigma, self.value_scale)  # exactly one, valid

    @property
    def count_scale(self) -> float:
        """The density is divided by this: ``value_scale``, or ``2 pi sigma^2``."""
        return density_value_scale(self.sigma, self.value_scale)

    def __call__(self, fields, block, scene) -> tuple[tuple[int, int, int, int], float]:
        """The block's core and the sum of the density field over it."""
        core, win, (it, ib, il, ir) = _layout(block, scene, 0)
        a = np.asarray(fields[self.field])
        a = a[None] if a.ndim == 2 else a
        d = _field(fields, self.field, a.shape[0], win)[:, it:ib, il:ir].astype(np.float64)
        if self.mask is not None:  # NaN in the mask compares False
            d = np.where(np.asarray(_field(fields, self.mask, 1, win))[:, it:ib, il:ir]
                         > self.mask_threshold, d, 0.0)
        return core, float(np.nansum(d)) / self.count_scale

    def combine(self, results: list, scene) -> SceneCount:
        """The scene total and per-block counts: :class:`SceneCount`."""
        res = list(results)
        counts = np.array([c for _, c in res], np.float64)
        return SceneCount(float(counts.sum()), counts, [core for core, _ in res])
