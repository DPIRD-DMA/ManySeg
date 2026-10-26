"""Polygons from instance label maps (footprints, fields, crowns) and GeoJSON export.

Same coordinate convention as :mod:`manyseg.vectors.graph`: ``(x, y)``, pixel corners at
integers, so an affine geotransform maps polygons to world coordinates.
"""

from __future__ import annotations

import numpy as np

from .graph import VectorGraph, _deps


def _runs(mask: np.ndarray):
    """(row, col_start, col_end) of the horizontal foreground runs of a 2-D bool mask."""
    m = np.pad(mask, ((0, 0), (1, 1)))
    d = np.diff(m.astype(np.int8), axis=1)
    r0, c0 = np.nonzero(d == 1)
    r1, c1 = np.nonzero(d == -1)
    return r0, c0, c1  # rows match: starts and ends come in the same row order


def polygons_from_labels(labels, *, simplify: float = 1.0, min_area: float = 16,
                         scale: float = 1.0) -> list:
    """One ``shapely.Polygon`` per instance of an integer label map ``(H, W)`` (0 = background),
    e.g. from :func:`manyseg.instances.instances_from_outputs`. Outlines follow pixel edges
    exactly (holes kept), then Douglas-Peucker ``simplify`` (px); polygons under ``min_area``
    px² are dropped. A disconnected instance keeps its largest part. Returned in label order.

    (E028's mask baseline traced marching-squares contours instead, which cut each corner by
    half a pixel.)
    """
    shapely = _deps()[1]
    from scipy import ndimage
    from shapely import affinity

    lab = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"expected a (H, W) label map, got shape {lab.shape}")
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab.astype(np.int64)), start=1):
        if sl is None:
            continue
        r, c0, c1 = _runs(lab[sl] == i)
        y0, x0 = sl[0].start, sl[1].start
        poly = shapely.union_all(shapely.box(c0 + x0, r + y0, c1 + x0, r + 1 + y0))
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        poly = poly.simplify(simplify)
        if poly.geom_type == "Polygon" and poly.area >= min_area:
            if scale != 1:
                poly = affinity.scale(poly, 1 / scale, 1 / scale, origin=(0, 0))
            out.append(poly)
    return out


def _apply(transform, xy: np.ndarray) -> list:
    if transform is None:
        return xy.tolist()
    a, b, c, d, e, f = (transform[k] for k in range(6))  # affine.Affine / rasterio order
    return np.stack([a * xy[:, 0] + b * xy[:, 1] + c, d * xy[:, 0] + e * xy[:, 1] + f], 1).tolist()


def to_geojson(geoms, transform=None, properties: list[dict] | None = None) -> dict:
    """GeoJSON ``FeatureCollection`` from a :class:`VectorGraph` (one LineString per polyline)
    or a list of ``shapely`` polygons. ``transform``: an affine geotransform in rasterio /
    ``affine.Affine`` order ``(a, b, c, d, e, f)`` from pixel ``(x, y)`` to world; ``None``
    keeps pixel coordinates."""
    if isinstance(geoms, VectorGraph):
        feats = [{"type": "LineString", "coordinates": _apply(transform, p)} for p in geoms.polylines()]
    else:
        feats = []
        for p in geoms:
            rings = [p.exterior, *p.interiors]
            feats.append({"type": "Polygon",
                          "coordinates": [_apply(transform, np.asarray(r.coords)) for r in rings]})
    props = [{} for _ in feats] if properties is None else properties
    if len(props) != len(feats):
        raise ValueError(f"{len(props)} properties for {len(feats)} features")
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": g, "properties": pr} for g, pr in zip(feats, props)]}
