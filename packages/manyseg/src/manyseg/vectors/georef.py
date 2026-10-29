"""Vector data in a CRS -> pixel geometry of one tile, the input of the target encoders.

The inverse of :func:`manyseg.vectors.to_geojson`'s transform convention: pixel coordinates are
``(x, y)`` with pixel corners at integers (pixel ``(row, col)`` covers ``[col, col + 1) x
[row, row + 1)``, as in :mod:`manyseg.vectors.graph`), and an affine geotransform in rasterio /
``affine.Affine`` order ``(a, b, c, d, e, f)`` maps them to world coordinates::

    X = a * x + b * y + c
    Y = d * x + e * y + f

For a north-up raster ``a`` is the pixel width, ``e`` minus the pixel height and ``(c, f)`` the
world coordinates of the top-left corner of the top-left pixel (``rasterio``'s
``dataset.transform``). A GDAL geotransform ``(c, a, b, f, d, e)`` (``GetGeoTransform()``)
converts with :func:`from_gdal`; a plain 6-sequence is always read in rasterio order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .graph import _deps


def from_gdal(geotransform) -> tuple[float, ...]:
    """A GDAL geotransform ``(c, a, b, f, d, e)`` in rasterio / ``affine.Affine`` order
    ``(a, b, c, d, e, f)``."""
    c, a, b, f, d, e = (float(v) for v in geotransform)
    return (a, b, c, d, e, f)


def _coeffs(transform) -> tuple[float, ...]:
    if all(hasattr(transform, k) for k in "abcdef"):  # affine.Affine (rasterio's transform)
        return tuple(float(getattr(transform, k)) for k in "abcdef")
    t = [float(v) for v in transform]
    if len(t) not in (6, 9):
        raise ValueError(f"transform: expected 6 (or 9) affine coefficients, got {len(t)}")
    return tuple(t[:6])


def _apply(fn, obj):
    shapely = _deps()[1]
    if isinstance(obj, shapely.Geometry) or (
            isinstance(obj, (list, tuple, np.ndarray)) and len(obj)
            and isinstance(obj[0], shapely.Geometry)):
        return shapely.transform(obj, fn)
    xy = np.asarray(obj, np.float64)
    return fn(xy.reshape(-1, 2)).reshape(xy.shape)


def pixel_to_world(obj, transform):
    """Pixel ``(x, y)`` (an ``(..., 2)`` array, a shapely geometry or a list of them) to world
    coordinates, as :func:`to_geojson` does."""
    a, b, c, d, e, f = _coeffs(transform)

    def fn(xy):
        return np.stack([a * xy[:, 0] + b * xy[:, 1] + c, d * xy[:, 0] + e * xy[:, 1] + f], 1)

    return _apply(fn, obj)


def world_to_pixel(obj, transform):
    """World coordinates (an ``(..., 2)`` array, a shapely geometry or a list of them) to pixel
    ``(x, y)``: the inverse of :func:`pixel_to_world`."""
    a, b, c, d, e, f = _coeffs(transform)
    det = a * e - b * d
    if det == 0:
        raise ValueError("transform is singular (zero pixel size?)")

    def fn(xy):
        u, v = xy[:, 0] - c, xy[:, 1] - f
        return np.stack([(e * u - b * v) / det, (a * v - d * u) / det], 1)

    return _apply(fn, obj)


def _geometries(features):
    """(list of shapely geometries or None, CRS or None) from the supported inputs."""
    shapely = _deps()[1]
    from shapely.geometry import shape as from_mapping

    crs = None
    if hasattr(features, "geometry") and hasattr(features, "crs"):  # GeoDataFrame / GeoSeries
        crs = features.crs
        items = list(features.geometry)
    elif isinstance(features, Mapping) and features.get("type") == "FeatureCollection":
        items = list(features["features"])
    elif (isinstance(features, (shapely.Geometry, Mapping))
          or hasattr(features, "__geo_interface__")):
        items = [features]
    elif isinstance(features, Iterable):
        items = list(features)
    else:
        raise TypeError(f"unsupported features: {type(features).__name__}")
    out = []
    for g in items:
        if g is not None and not isinstance(g, shapely.Geometry):
            m = g.__geo_interface__ if hasattr(g, "__geo_interface__") else g
            if not isinstance(m, Mapping):
                raise TypeError(f"unsupported feature: {type(g).__name__}")
            m = m.get("geometry", m) if m.get("type") == "Feature" else m
            g = from_mapping(m) if m is not None else None
        out.append(None if g is None or g.is_empty else shapely.force_2d(g))
    return out, crs


def _same_crs(a, b) -> bool:
    from rasterio.crs import CRS

    def norm(c):
        return CRS.from_user_input(c.to_wkt() if hasattr(c, "to_wkt") else c)

    return a is b or a == b or norm(a) == norm(b)


def _reproject(geoms, src, dst):
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp

    shapely = _deps()[1]

    def crs(c):
        return CRS.from_user_input(c.to_wkt() if hasattr(c, "to_wkt") else c)

    s, d = crs(src), crs(dst)

    def fn(xy):
        if not len(xy):
            return xy
        x, y = warp(s, d, xy[:, 0], xy[:, 1])
        return np.stack([np.asarray(x), np.asarray(y)], 1)

    return [None if g is None else shapely.transform(g, fn) for g in geoms]


def _tile_filter(geoms, transform, shape, src_crs, dst_crs):
    """Index of the geometries whose bounds touch the tile (in the features' CRS)."""
    shapely = _deps()[1]
    H, W = shape
    corners = pixel_to_world(np.array([[-1, -1], [W + 1, -1], [W + 1, H + 1], [-1, H + 1]], float),
                             transform)
    x0, y0 = corners.min(0)
    x1, y1 = corners.max(0)
    if src_crs is not None:
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        def crs(c):
            return CRS.from_user_input(c.to_wkt() if hasattr(c, "to_wkt") else c)

        x0, y0, x1, y1 = transform_bounds(crs(dst_crs), crs(src_crs), x0, y0, x1, y1,
                                          densify_pts=21)
    box = shapely.box(x0, y0, x1, y1)
    keep = [i for i, g in enumerate(geoms) if g is not None]
    hit = shapely.intersects(np.array([geoms[i] for i in keep], dtype=object), box)
    return [i for i, h in zip(keep, hit) if h]


def _parts(g, kind: str) -> list:
    """Polygon / LineString parts of any geometry (collections and multipart split)."""
    want = ("Polygon",) if kind == "polygons" else ("LineString", "LinearRing")
    if g.geom_type in want:
        return [g]
    if hasattr(g, "geoms"):
        return [q for p in g.geoms for q in _parts(p, kind)]
    return []


def _kind(g) -> str | None:
    t = g.geom_type.removeprefix("Multi")
    if t == "Polygon":
        return "polygons"
    if t in ("LineString", "LinearRing"):
        return "lines"
    if t == "GeometryCollection":
        kinds = {_kind(p) for p in g.geoms} - {None}
        return kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else None)
    return None


def to_pixels(features, transform, shape: tuple[int, int], *, crs=None, raster_crs=None,
              kind: str | None = None, min_area: float = 0.0, min_length: float = 0.0,
              return_index: bool = False):
    """Vector features in world coordinates -> pixel-coordinate geometries clipped to one tile,
    as the target encoders take them (:func:`footprint_targets`, :func:`field_targets`,
    :func:`roof_face_targets`, :func:`road_graph`).

    Args:
        features: shapely geometries, GeoJSON-like mappings (geometries, ``Feature``\\ s or a
            ``FeatureCollection``), objects with ``__geo_interface__``, or a geopandas
            ``GeoDataFrame`` / ``GeoSeries`` (its ``crs`` is used when ``crs`` is not given).
            ``None`` and empty geometries are skipped; Z is dropped.
        transform: the tile's affine geotransform in rasterio order (``dataset.transform``, an
            ``affine.Affine`` or ``(a, b, c, d, e, f)``; GDAL order through :func:`from_gdal`).
        shape: the tile's ``(H, W)``.
        crs, raster_crs: the features' and the raster's CRS (anything rasterio's
            ``CRS.from_user_input`` takes, or a pyproj CRS). When both are known and differ,
            vertices are reprojected (rasterio / PROJ; edges stay straight, so densify long
            edges first if the projections differ strongly). When either is unknown the
            features are taken to be in the raster's CRS.
        kind: ``"polygons"`` or ``"lines"``; inferred when ``None`` (mixed inputs raise).
        min_area, min_length: drop clipped parts at or below this size (px² / px; the default
            drops only empty and degenerate parts).
        return_index: also return, per output part, the index of its source feature.

    Polygons are made valid (``shapely.make_valid``), clipped to ``[0, W] x [0, H]`` and split
    into single ``Polygon``\\ s (holes kept). Lines are clipped and split into single
    ``LineString``\\ s; a line that leaves and re-enters the tile becomes two parts. Vertices
    are kept exactly where a geometry lies inside the tile, so lines that share a vertex in the
    source data (a junction) still share it in pixels.
    """
    shapely = _deps()[1]
    if kind not in (None, "polygons", "lines"):
        raise ValueError(f"kind must be 'polygons', 'lines' or None, got {kind!r}")
    H, W = shape
    geoms, fcrs = _geometries(features)
    src = crs if crs is not None else fcrs
    reproject = src is not None and raster_crs is not None and not _same_crs(src, raster_crs)
    idx = _tile_filter(geoms, transform, shape, src if reproject else None, raster_crs)
    if kind is None:
        kinds = {_kind(geoms[i]) for i in idx} - {None}
        if len(kinds) > 1 or "mixed" in kinds:
            raise ValueError("features mix polygons and lines; pass kind= to pick one")
        kind = kinds.pop() if kinds else "polygons"
    sel = [geoms[i] for i in idx]
    if reproject:
        sel = _reproject(sel, src, raster_crs)
    sel = world_to_pixel(sel, transform) if sel else []
    tile = shapely.box(0, 0, W, H)
    out, index = [], []
    for i, g in zip(idx, sel):
        if kind == "polygons":
            g = g if g.is_valid else shapely.make_valid(g)
            parts = _parts(g, kind)
            clipped = [q for p in parts
                       for q in ([p] if tile.contains(p) else _parts(p.intersection(tile), kind))]
            clipped = [q for q in clipped if q.area > min_area]
        else:
            parts = _parts(g, kind)
            clipped = [q for p in parts
                       for q in ([p] if tile.contains(p) else
                                 _parts(shapely.line_merge(p.intersection(tile)), kind))]
            clipped = [shapely.LineString(q.coords) for q in clipped if q.length > min_length]
        out += clipped
        index += [i] * len(clipped)
    return (out, np.asarray(index, np.int64)) if return_index else out
