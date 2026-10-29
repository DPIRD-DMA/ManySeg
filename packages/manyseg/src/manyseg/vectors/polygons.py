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
    from scipy import ndimage
    from shapely import affinity

    lab = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"expected a (H, W) label map, got shape {lab.shape}")
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab.astype(np.int64)), start=1):
        if sl is None:
            continue
        poly = _outline(lab[sl] == i, sl[0].start, sl[1].start).simplify(simplify)
        if poly.geom_type == "Polygon" and poly.area >= min_area:
            if scale != 1:
                poly = affinity.scale(poly, 1 / scale, 1 / scale, origin=(0, 0))
            out.append(poly)
    return out


def _outline(mask: np.ndarray, y0: int, x0: int):
    """Pixel-edge outline of a bool ``mask`` whose top-left pixel is at integer ``(y0, x0)``
    (largest part if disconnected), not simplified. Integer offsets keep the coordinates exact,
    so a mask cut from a window gives the same polygon as the same mask in the whole scene."""
    shapely = _deps()[1]
    r, c0, c1 = _runs(mask)
    poly = shapely.union_all(shapely.box(c0 + x0, r + y0, c1 + x0, r + 1 + y0))
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


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
    return [f for _, _, f in _planar_faces(polygons, tolerance)]


def _face_clusters(faces: list, tolerance: float) -> list[list[int]]:
    """Indices of ``faces`` grouped by gaps up to ``tolerance`` (connected components of
    "within ``tolerance``"), each group and the list in index order."""
    shapely = _deps()[1]
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if len(faces) > 1:
        a, b = shapely.STRtree(faces).query(faces, predicate="dwithin", distance=tolerance)
        for i, j in zip(a.tolist(), b.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    groups: dict = {}
    for i in range(len(faces)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def _planar_faces(polygons, tolerance: float) -> list:
    """:func:`planar_faces` as ``(face index, part, polygon)`` triples; the face index counts the
    Polygon parts of the input, part 0 is the kept (largest) part of a face, 1.. the others.

    Faces interact only through gaps up to ``tolerance``, so each cluster of such faces is
    joined on its own: the result for a face depends only on its cluster (not on the noding of
    unrelated outlines elsewhere), which lets block-wise decoders reproduce it exactly."""
    faces = [q for p in polygons for q in getattr(p, "geoms", [p])
             if q.geom_type == "Polygon" and not q.is_empty]
    if tolerance <= 0:
        return [(i, 0, f) for i, f in enumerate(faces)]
    main: dict = {}
    extra: dict = {}
    for cl in _face_clusters(faces, tolerance):
        for j, part, g in _join_cluster([faces[i] for i in cl], tolerance):
            (main if part == 0 else extra.setdefault(cl[j], {}))[cl[j] if part == 0 else part] = g
    out = [(j, 0, main[j]) for j in sorted(main)]
    return out + [(j, k, g) for j in sorted(extra) for k, g in sorted(extra[j].items())]


def _join_cluster(faces: list, tolerance: float) -> list:
    shapely = _deps()[1]
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
    out = []
    for j in range(len(faces)):
        if j not in parts:
            continue
        u = shapely.union_all(parts[j])
        pieces = [g for g in getattr(u, "geoms", [u]) if g.geom_type == "Polygon" and g.area > 1]
        pieces.sort(key=lambda g: -g.area)
        out += [(j, k, shapely.Polygon(g.exterior)) for k, g in enumerate(pieces)]
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
