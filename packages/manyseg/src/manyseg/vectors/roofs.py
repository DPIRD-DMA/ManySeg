"""Roof planes and meshes from roof faces and a height raster (experimental).

Faces come from :func:`polygons_from_corners` joined by :func:`planar_faces`. Each face gets a
plane ``z = a x + b y + c`` (metres above ground; ``x``, ``y`` in pixels, pixel centres at
``+0.5``) fitted robustly to a height raster (nDSM): the face eroded by ``erode`` px (edge
pixels blur across ridges and eaves), a trimmed least-squares fit (``trim`` of the largest
residuals dropped) repeated twice. :func:`roof_mesh` lifts the face rings onto their planes and
joins vertices shared by several faces. Coordinates follow
:mod:`manyseg.vectors.graph`.
"""

from __future__ import annotations

import numpy as np

from .graph import _deps

MESH_JOINS = ("plane_intersection", "mean_height")
"""``roof_mesh(join=...)`` and ``mesh=`` values: where a vertex shared by several faces goes."""


def _face_region(face, shape: tuple[int, int], origin: tuple[int, int], erode: int):
    """Pixels of ``face`` (centre strictly inside) in the window ``origin`` + ``shape`` of the
    scene, eroded by ``erode`` px unless that leaves fewer than 10."""
    shapely = _deps()[1]
    from scipy import ndimage

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    m = shapely.contains_xy(face, xx + origin[1] + 0.5, yy + origin[0] + 0.5)
    if erode:
        e = ndimage.binary_erosion(m, iterations=erode)
        if e.sum() >= 10:
            return e
    return m


def face_plane(face, height: np.ndarray, origin: tuple[int, int] = (0, 0), *, erode: int = 2,
               trim: float = 0.2) -> np.ndarray:
    """Robust plane ``(a, b, c)`` of one face (scene pixel coordinates) from ``height``, an
    (H, W) array whose top-left pixel sits at scene ``origin`` (row, col). Only the face's
    bounding box plus ``erode + 1`` px is read; the result equals a call on the whole scene as
    long as that box lies inside ``height`` or is clipped only by the scene edge."""
    H, W = height.shape
    x0, y0, x1, y1 = face.bounds
    pad = erode + 1
    r0 = max(int(np.floor(y0)) - pad - origin[0], 0)
    c0 = max(int(np.floor(x0)) - pad - origin[1], 0)
    r1 = min(int(np.ceil(y1)) + pad - origin[0], H)
    c1 = min(int(np.ceil(x1)) + pad - origin[1], W)
    if r0 >= r1 or c0 >= c1:
        return np.zeros(3)
    reg = _face_region(face, (r1 - r0, c1 - c0), (origin[0] + r0, origin[1] + c0), erode)
    rr, cc = np.nonzero(reg)
    z = height[r0:r1, c0:c1][rr, cc].astype(np.float64)
    if len(rr) < 3:
        return np.array([0.0, 0.0, float(np.median(z)) if len(rr) else 0.0])
    # scene coordinates: integer shift first, then the pixel centre (same floats as a scene call)
    X = np.c_[cc + (origin[1] + c0) + 0.5, rr + (origin[0] + r0) + 0.5, np.ones(len(rr))]
    p = np.linalg.lstsq(X, z, rcond=None)[0]
    if trim > 0 and len(rr) > 10:
        res = np.abs(X @ p - z)
        keep = res <= np.quantile(res, 1 - trim)
        p = np.linalg.lstsq(X[keep], z[keep], rcond=None)[0]
    if len(rr) > 20:  # second trimming round on the refit residuals
        res = np.abs(X @ p - z)
        keep = res <= np.quantile(res, 1 - trim)
        p = np.linalg.lstsq(X[keep], z[keep], rcond=None)[0]
    return p


def roof_planes(faces, height, origin: tuple[int, int] = (0, 0), *, erode: int = 2,
                trim: float = 0.2) -> np.ndarray:
    """(N, 3) planes ``(a, b, c)``, one per face, see :func:`face_plane`. ``height``: (H, W)
    nDSM in metres (numpy, memmap or torch)."""
    h = height.detach().float().cpu().numpy() if hasattr(height, "detach") else height
    P = [face_plane(f, h, origin, erode=erode, trim=trim) for f in faces]
    return np.array(P).reshape(-1, 3)


def roof_mesh(faces, planes: np.ndarray, *, join: str = "plane_intersection",
              step: float = 0.5, cap: float = 2.0, lam: float = 0.05) -> list:
    """Per face, its exterior ring (without the closing vertex) as (K, 3) ``(x px, y px, z m)``.

    A 2-D vertex shared by several faces (equal to 0.01 px, as :func:`planar_faces` makes them)
    takes the mean of its planes' heights when they agree within ``step`` m (watertight),
    otherwise each face keeps its own height (a step wall). ``join="plane_intersection"`` first
    moves a shared vertex (at most ``cap`` px) towards where its faces' planes meet (least
    squares, ``lam`` pulls it back to where it was), so ridges sit on plane intersections;
    ``join="mean_height"`` leaves it where it is.
    Vertices are independent of each other, so a mesh of a group of faces equals the same faces'
    rings in a mesh of the whole scene.
    """
    if join not in MESH_JOINS:
        raise ValueError(f"join must be one of {MESH_JOINS}, got {join!r}")
    P = np.asarray(planes, np.float64).reshape(-1, 3)
    shared: dict = {}
    rings = []
    for i, f in enumerate(faces):
        xy = np.asarray(f.exterior.coords)[:-1]
        rings.append(np.c_[xy, np.zeros(len(xy))])
        for k, (x, y) in enumerate(xy):
            shared.setdefault((round(x, 2), round(y, 2)), []).append((i, k))
    for (x, y), users in shared.items():
        fids = sorted({i for i, _ in users})
        v = np.array([x, y], float)
        if join == "plane_intersection" and len(fids) >= 2:
            rows = [[P[i, 0], P[i, 1], -1.0] for i in fids]
            rhs = [-P[i, 2] for i in fids]
            rows += [[np.sqrt(lam), 0, 0], [0, np.sqrt(lam), 0]]
            rhs += [np.sqrt(lam) * v[0], np.sqrt(lam) * v[1]]
            sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
            d = sol[:2] - v
            n = float(np.hypot(*d))
            v = v + (d if n <= cap else d * cap / n)
        zs = {i: float(P[i, 0] * v[0] + P[i, 1] * v[1] + P[i, 2]) for i in fids}
        if len(fids) >= 2 and max(zs.values()) - min(zs.values()) <= step:
            zm = float(np.mean(list(zs.values())))
            zs = {i: zm for i in fids}
        for i, k in users:
            rings[i][k] = [v[0], v[1], zs[i]]
    return rings


def roof_records(anchors, faces, height, origin: tuple[int, int] = (0, 0), *,
                 tolerance: float = 2.5, erode: int = 2, trim: float = 0.2,
                 mesh: str | None = "plane_intersection") -> list[dict]:
    """Join roof ``faces`` (one per instance, keyed by ``anchors``) with :func:`planar_faces`,
    fit planes from ``height`` (top-left pixel at scene ``origin``; or a callable
    ``height(face) -> (array, origin)`` giving a crop around each joined face) and mesh them.
    Faces enter
    the join in anchor order. Records sorted by ``(id, part)``: ``id`` = the instance's anchor,
    ``part`` 0 (or 1.. for pieces :func:`planar_faces` split off), ``geometry``, ``plane``
    ``(a, b, c)`` and, unless ``mesh`` is None, ``mesh`` ((K, 3) ring, :func:`roof_mesh`)."""
    from .polygons import _planar_faces

    order = sorted(range(len(faces)), key=lambda i: anchors[i])
    anchors = [anchors[i] for i in order]
    joined = _planar_faces([faces[i] for i in order], tolerance)
    geoms = [g for _, _, g in joined]
    if callable(height):
        P = np.array([face_plane(g, *height(g), erode=erode, trim=trim) for g in geoms])
        P = P.reshape(-1, 3)
    else:
        P = roof_planes(geoms, height, origin, erode=erode, trim=trim)
    rings = roof_mesh(geoms, P, join=mesh) if mesh else [None] * len(geoms)
    out = []
    for (j, part, g), p, ring in zip(joined, P, rings):
        rec = {"id": int(anchors[j]), "part": part, "geometry": g, "plane": tuple(map(float, p))}
        if ring is not None:
            rec["mesh"] = ring
        out.append(rec)
    out.sort(key=lambda r: (r["id"], r["part"]))
    return out


def roof_faces(labels, corner, direction, height, *, tolerance: float = 2.5, erode: int = 2,
               trim: float = 0.2, mesh: str | None = "plane_intersection",
               corner_threshold: float = 0.5,
               kernel: int = 3, snap_radius: float = 4.0, big_dev: float = 4.0,
               agree: float = 0.6, merge_deg: float = 8.0, simplify: float = 1.0,
               min_area: float = 16) -> list[dict]:
    """Whole-scene roof decode from instance ``labels`` (H, W), ``corner`` (H, W), ``direction``
    (2, H, W) and ``height`` (H, W, nDSM in metres): faces of :func:`polygons_from_corners`
    (same parameters) -> :func:`roof_records`. Defaults are the settings tuned on trained roof
    models (E047b), as in ``block_decoder("roofs")``, which gives the same records block by
    block on scenes too large for memory (with the same parameters and instance labels from the same thresholds)."""
    from scipy import ndimage

    from .corners import _corner_peaks, _instance_polygon

    def arr(x):
        return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    lab = arr(labels).astype(np.int64)
    cp, dr = arr(corner).astype(np.float32), arr(direction).astype(np.float32)
    peaks = _corner_peaks(cp, corner_threshold, kernel)[1]
    W = lab.shape[1]
    anchors, faces = [], []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        q = _instance_polygon(m, sl[0].start, sl[1].start, peaks, dr, snap_radius=snap_radius,
                              big_dev=big_dev, agree=agree, merge_deg=merge_deg,
                              simplify=simplify, min_area=min_area)
        if q is not None:
            fr = int(np.argmax(m.reshape(-1)))
            anchors.append((sl[0].start + fr // m.shape[1]) * W + sl[1].start + fr % m.shape[1])
            faces.append(q)
    return roof_records(anchors, faces, arr(height), tolerance=tolerance, erode=erode,
                        trim=trim, mesh=mesh)
