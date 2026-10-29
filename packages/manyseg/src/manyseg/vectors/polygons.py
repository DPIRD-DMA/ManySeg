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



def planar_faces(polygons, tolerance: float = 2.5) -> list:
    """Join separately decoded neighbouring faces into a planar partition (shared edges and
    vertices), e.g. roof faces from :func:`polygons_from_corners` before a mesh join.

    Faces are snapped, largest first, onto the faces already placed within ``tolerance`` px; all
    outlines are noded together and re-polygonised; each cell goes to the input face covering
    most of it (cells under 50% covered are dropped) and a face's cells are merged. Faces with no
    cell are dropped; holes are not kept. Returned in input order; MultiPolygon inputs are split
    into their parts first, and a face whose cells come out disconnected (another face cut through
    it) keeps its largest part in place with the other parts appended at the end, so no covered
    area is lost. E048b (trained LoD2 faces):
    seam closure 0.38 -> 0.72 (3DBAG's own faces: 0.79), heights better, face F1 unchanged.
    """
    shapely = _deps()[1]
    faces = [q for p in polygons for q in getattr(p, "geoms", [p])
             if q.geom_type == "Polygon" and not q.is_empty]
    if len(faces) < 2 or tolerance <= 0:
        return faces
    tree = shapely.STRtree(faces)
    placed: dict = {}
    for i in sorted(range(len(faces)), key=lambda k: -faces[k].area):
        f = faces[i]
        near = tree.query(f, predicate="dwithin", distance=tolerance)
        nb = [placed[j] for j in near if j in placed]
        g = shapely.snap(f, shapely.union_all([n.exterior for n in nb]), tolerance) if nb else f
        g = g if g.is_valid else g.buffer(0)
        placed[i] = g if g.geom_type == "Polygon" and g.area > 1 else f
    lines = shapely.union_all([placed[i].exterior for i in range(len(faces))])
    parts: dict = {}
    for c in shapely.polygonize([lines]).geoms:
        cand = tree.query(c, predicate="intersects")
        if not len(cand):
            continue
        a, j = max((faces[int(k)].intersection(c).area, int(k)) for k in cand)
        if a >= 0.5 * c.area:
            parts.setdefault(j, []).append(c)
    out, extra = [], []
    for j in range(len(faces)):
        if j not in parts:
            continue
        u = shapely.union_all(parts[j])
        pieces = [g for g in getattr(u, "geoms", [u]) if g.geom_type == "Polygon" and g.area > 1]
        pieces.sort(key=lambda g: -g.area)
        if pieces:
            out.append(shapely.Polygon(pieces[0].exterior))
            extra += [shapely.Polygon(g.exterior) for g in pieces[1:]]
    return out + extra

def _apply(transform, xy: np.ndarray) -> list:
    if transform is None:
        return xy.tolist()
    a, b, c, d, e, f = (transform[k] for k in range(6))  # affine.Affine / rasterio order
    return np.stack([a * xy[:, 0] + b * xy[:, 1] + c, d * xy[:, 0] + e * xy[:, 1] + f], 1).tolist()


def _point_rows(geoms) -> np.ndarray | None:
    """``(N, 2)`` float64 ``(x, y)`` if ``geoms`` is a plain point array / sequence, else None."""
    if hasattr(geoms, "detach"):
        raise TypeError("to_geojson takes pixel (x, y) with pixel corners at integers; "
                        "manyseg.points returns (y, x) tensors with pixel centres at integers: "
                        "convert with manyseg.points.points_to_xy")
    if isinstance(geoms, np.ndarray):
        a = geoms
    elif len(geoms) and not hasattr(geoms[0], "geom_type"):
        a = np.asarray(geoms)
    else:
        return None
    if a.dtype == object or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"points must be an (N, 2) array of (x, y), got shape {a.shape}")
    return a.astype(np.float64)


def _properties(properties, n: int) -> list[dict]:
    if properties is None:
        return [{} for _ in range(n)]
    if isinstance(properties, dict):  # columns: {"score": (N,) array, ...}
        cols = {k: np.asarray(v).tolist() for k, v in properties.items()}
        for k, v in cols.items():
            if not isinstance(v, list) or len(v) != n:
                raise ValueError(f"property {k!r} has {np.size(v)} values for {n} features")
        return [{k: v[j] for k, v in cols.items()} for j in range(n)]
    if len(properties) != n:
        raise ValueError(f"{len(properties)} properties for {n} features")
    return list(properties)


def to_geojson(geoms, transform=None, properties: list[dict] | dict | None = None) -> dict:
    """GeoJSON ``FeatureCollection`` from a :class:`VectorGraph` (one LineString per polyline),
    a list of ``shapely`` polygons, or points. ``transform``: an affine geotransform in rasterio /
    ``affine.Affine`` order ``(a, b, c, d, e, f)`` from pixel ``(x, y)`` to world; ``None``
    keeps pixel coordinates.

    Points: an ``(N, 2)`` array or a sequence of pixel ``(x, y)`` (one ``Point`` feature each),
    or shapely ``Point`` / ``MultiPoint`` geometries (alone or in the list, mixed with polygons
    if needed). Pixel corners are at integers here as for polygons, so the centre of pixel
    ``(row, col)`` is ``(col + 0.5, row + 0.5)``. :mod:`manyseg.points` (``find_peaks``) uses
    ``(y, x)`` with centres at integers: convert with :func:`manyseg.points.points_to_xy`, or
    use the ``xy`` of a :class:`manyseg.points.ScenePoints`. Torch tensors are refused for that
    reason.

    ``properties``: one dict per feature, or a dict of columns (``{"score": scores}``, one
    value per feature; numpy values become Python numbers).
    """
    if not (isinstance(geoms, (VectorGraph, np.ndarray)) or hasattr(geoms, "geom_type")
            or hasattr(geoms, "detach")):
        geoms = list(geoms)  # a generator is read once
    if isinstance(geoms, VectorGraph):
        feats = [{"type": "LineString", "coordinates": _apply(transform, p)} for p in geoms.polylines()]
    elif hasattr(geoms, "geom_type") or (rows := _point_rows(geoms)) is None:
        feats = []
        for p in ([geoms] if hasattr(geoms, "geom_type") else geoms):
            if p.geom_type == "Point":
                feats.append({"type": "Point",
                              "coordinates": _apply(transform, np.asarray(p.coords))[0]})
            elif p.geom_type == "MultiPoint":
                xy = np.asarray([q.coords[0] for q in p.geoms]).reshape(-1, 2)
                feats.append({"type": "MultiPoint", "coordinates": _apply(transform, xy)})
            else:
                rings = [_apply(transform, np.asarray(r.coords))
                         for r in [p.exterior, *p.interiors]]
                feats.append({"type": "Polygon", "coordinates": rings})
    else:
        feats = [{"type": "Point", "coordinates": c} for c in _apply(transform, rows)]
    props = _properties(properties, len(feats))
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": g, "properties": pr} for g, pr in zip(feats, props)]}
