"""Corner-trusting polygons (experimental): instance outlines straightened between predicted
corners.

Closure comes from an instance label map (e.g. :func:`manyseg.instances.instances_from_outputs`),
so a missed corner never loses a building; geometry comes from a corner heatmap and a wall
direction field ``(cos 2t, sin 2t)`` (orientation-invariant). Each instance's exact pixel-edge
outline is split at the corner peaks snapped onto it; a wall is split further only on strong
evidence (an outline excursion above ``big_dev`` px, a real missed turn, or a mean direction
agreement below ``agree``); each wall is a trimmed total-least-squares line, so a dent in the
mask does not bend it; nearly collinear walls are merged; consecutive lines are intersected for
the final corners. Coordinates follow :mod:`manyseg.vectors.graph`.
"""

from __future__ import annotations

import math

import numpy as np

from ..utils._peaks import peak_pixels, refine
from .graph import _deps
from .polygons import _outline


def _trimmed_line(pts: np.ndarray, keep_px: float = 1.5):
    c = pts.mean(0)
    u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    rel = pts - c
    off = np.abs(u[0] * rel[:, 1] - u[1] * rel[:, 0])
    inl = pts[off <= keep_px]
    if 2 <= len(inl) < len(pts):
        c = inl.mean(0)
        u = np.linalg.svd(inl - c, full_matrices=False)[2][0]
    return c, u


def _unit(dirs: np.ndarray, min_norm: float = 0.1) -> np.ndarray:
    """Orientation only: unit vectors; near-zero ones (no evidence) become 0, agreement 0.5."""
    n = np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.where(n > min_norm, dirs / np.maximum(n, 1e-6), 0.0)


def _split(points: np.ndarray, dirs: np.ndarray, big_dev: float, agree: float) -> list[int]:
    if len(points) < 6:
        return []
    a, b = points[0], points[-1]
    ab = b - a
    L = float(np.hypot(*ab))
    if L < 1e-6:
        return []
    rel = points - a
    off = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
    th = math.atan2(ab[1], ab[0])
    per = (1 + dirs[:, 0] * math.cos(2 * th) + dirs[:, 1] * math.sin(2 * th)) / 2
    if off.max() > big_dev:
        k = int(np.argmax(off))
    elif per.mean() < agree:
        c = np.cumsum(per)
        n = len(per)
        ks = np.arange(3, n - 2)
        k = int(ks[np.argmax(np.abs(c[ks - 1] / ks - (c[-1] - c[ks - 1]) / (n - ks)))])
    else:
        return []
    left = _split(points[: k + 1], dirs[: k + 1], big_dev, agree)
    right = _split(points[k:], dirs[k:], big_dev, agree)
    return left + [k] + [k + i for i in right]


def _intersect(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + s * d1


def _polygon(ring: np.ndarray, corners: np.ndarray, direction: np.ndarray, *, snap_radius: float,
             big_dev: float, agree: float, merge_deg: float, simplify: float,
             origin: tuple[int, int] = (0, 0)):
    """``origin``: (row, col) of ``direction`` in the coordinates of ``ring`` and ``corners``."""
    shapely = _deps()[1]
    cKDTree = _deps()[2]
    H, W = direction.shape[1:]
    ring = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    K = len(ring)
    if K < 4:
        return None
    idx = []
    if len(corners):
        d, j = cKDTree(ring).query(corners)
        idx = sorted({int(x) for x, dd in zip(j, d) if dd <= snap_radius})
    def fallback():  # the instance's own outline, simplified (never drop a building the mask found)
        q = shapely.Polygon(ring).buffer(0).simplify(simplify)
        return q if q.geom_type == "Polygon" and q.area > 1 else None

    if len(idx) < 3:
        return fallback()
    c = np.clip(np.floor(ring[:, 0]).astype(int) - origin[1], 0, W - 1)
    r = np.clip(np.floor(ring[:, 1]).astype(int) - origin[0], 0, H - 1)
    dirs = _unit(direction[:, r, c].T)
    verts = []
    for a, b in zip(idx, idx[1:] + [idx[0] + K]):
        seg = np.arange(a, b + 1)
        verts += [a] + [a + s for s in _split(ring[seg % K], dirs[seg % K], big_dev, agree)]
    verts = sorted(set(verts))
    walls = []
    for a, b in zip(verts, verts[1:] + [verts[0] + K]):
        pts = ring[np.arange(a, b + 1) % K]
        if len(pts) >= 2 and not np.allclose(pts[0], pts[-1]):
            walls.append(_trimmed_line(pts) + (a,))
    merged = True
    while merged and len(walls) > 3:
        merged = False
        for i in range(len(walls)):
            cos = min(1.0, abs(float(walls[i - 1][1] @ walls[i][1])))
            if math.degrees(math.acos(cos)) < merge_deg:
                del walls[i]
                merged = True
                break
    if len(walls) < 3:
        return fallback()
    out = []
    for i in range(len(walls)):
        (p1, d1, _), (p2, d2, s2) = walls[i - 1], walls[i]
        x = _intersect(p1, d1, p2, d2)
        corner = ring[s2 % K]
        out.append(x if x is not None and np.hypot(*(x - corner)) <= 3 * snap_radius else corner)
    q = shapely.Polygon(out)
    if not q.is_valid:
        q = q.buffer(0)
        if q.geom_type == "MultiPolygon":
            q = max(q.geoms, key=lambda g: g.area)
    return q if q.geom_type == "Polygon" and q.area > 1 else fallback()


def polygons_from_corners(labels, corner, direction, *, corner_threshold: float = 0.3,
                          kernel: int = 3, snap_radius: float = 4.0, big_dev: float = 8.0,
                          agree: float = 0.4, merge_deg: float = 15.0, simplify: float = 1.0,
                          min_area: float = 16, holes: bool = True) -> list:
    """Straight-walled polygons from instance ``labels`` (H, W), a ``corner`` probability map
    (H, W) and a ``direction`` field (2, H, W) of ``(cos 2t, sin 2t)`` (numpy or torch).
    Instances with fewer than 3 snapped corners, or whose straightened polygon degenerates
    (< 3 walls, invalid, empty), fall back to their outline simplified by ``simplify`` px, so a
    building the instance mask found is never dropped. With ``holes``, an instance's holes of at
    least ``min_area`` (courtyards) are cut out as their traced outline simplified by
    ``simplify`` px: only exterior walls are straightened, as :func:`corner_targets` puts corners
    and directions only there.

    A low-level building block with generic defaults. The settings tuned on trained models are
    in ``block_decoder("footprints")`` (E038: ``corner_threshold=0.5``, ``big_dev=4``,
    ``merge_deg=8``) and ``block_decoder("roofs")`` (E047b); use those with
    :func:`decode_scene` unless you need this step on its own."""
    def _np(a):
        if hasattr(a, "detach"):
            return a.detach().float().cpu().numpy()
        return np.asarray(a, np.float32)

    cp, dr = _np(corner), _np(direction)
    from scipy import ndimage

    peaks = _corner_peaks(cp, corner_threshold, kernel)[1]
    peaks = peaks[np.argsort(peaks[:, 0], kind="stable")]  # each instance reads the ones near it
    lab = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"expected a (H, W) label map, got shape {lab.shape}")
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab.astype(np.int64)), start=1):
        if sl is None:
            continue
        x0, x1 = np.searchsorted(peaks[:, 0], (sl[1].start - snap_radius, sl[1].stop + snap_radius))
        near = peaks[x0:x1]
        near = near[(near[:, 1] >= sl[0].start - snap_radius)
                    & (near[:, 1] <= sl[0].stop + snap_radius)]
        q = _instance_polygon(lab[sl] == i, sl[0].start, sl[1].start, near, dr,
                              snap_radius=snap_radius, big_dev=big_dev, agree=agree,
                              merge_deg=merge_deg, simplify=simplify, min_area=min_area,
                              holes=holes)
        if q is not None:
            out.append(q)
    return out


def _instance_polygon(mask: np.ndarray, y0: int, x0: int, peaks: np.ndarray,
                      direction: np.ndarray, origin: tuple[int, int] = (0, 0), *,
                      snap_radius: float, big_dev: float, agree: float, merge_deg: float,
                      simplify: float, min_area: float, holes: bool = False):
    """:func:`polygons_from_corners` for one instance: bool ``mask`` with its top-left pixel at
    scene ``(y0, x0)``, ``peaks`` (x, y) and ``direction`` with its top-left pixel at scene
    ``origin`` (row, col). None if the outline is under ``min_area`` or degenerates."""
    shapely = _deps()[1]
    poly = _outline(mask, y0, x0).simplify(0.0)
    if poly.geom_type != "Polygon" or poly.area < min_area:
        return None
    ring = np.asarray(shapely.segmentize(poly.exterior, 1.0).coords)
    q = _polygon(ring, peaks, direction, snap_radius=snap_radius, big_dev=big_dev,
                 agree=agree, merge_deg=merge_deg, simplify=simplify, origin=origin)
    return _cut_holes(q, poly, simplify, min_area) if holes and q is not None else q


def _cut_holes(q, outline, simplify: float, min_area: float):
    """``q`` without the holes of ``outline`` of at least ``min_area``, each simplified by
    ``simplify`` px. A hole that straightening left outside or across ``q``'s walls cuts only
    what it overlaps; if that splits ``q``, the largest part stays."""
    shapely = _deps()[1]
    cut = [h.simplify(simplify) for h in map(shapely.Polygon, outline.interiors)]
    cut = [h for h in cut if h.area >= min_area]
    if not cut:
        return q
    out = q.difference(shapely.union_all(cut))
    if out.geom_type != "Polygon":
        out = max(getattr(out, "geoms", []), key=lambda g: g.area, default=q)
    return out if out.geom_type == "Polygon" and out.area > 1 else q


def _corner_peaks(corner: np.ndarray, threshold: float = 0.3, kernel: int = 3,
                 origin: tuple[int, int] = (0, 0)):
    """Corner peaks of a (H, W) heatmap as used by :func:`polygons_from_corners`: integer
    ``(row, col)`` pixels and sub-pixel ``(x, y)``, both shifted by ``origin`` (the array's
    position in a larger scene; integer shifts keep the values identical to a scene-wide call
    for peaks a few pixels inside the array)."""
    pix = peak_pixels(corner, threshold, kernel)
    if len(pix) == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2))
    return pix + np.asarray(origin, pix.dtype), refine(corner, pix, origin)


def corner_targets(polygons, shape: tuple[int, int], *, tolerance: float = 1.5,
                   turn_deg: float = 20.0,
                   sigma: float = 2.0, line_width: float = 1.0):
    """Training targets for :func:`polygons_from_corners` from polygons (pixel coordinates):
    ``corner`` (H, W) Gaussian blobs at vertices turning more than ``turn_deg`` after a
    ``tolerance`` px simplification (raster staircase jogs are not corners), ``direction``
    (2, H, W) ``(cos 2t, sin 2t)`` of the simplified walls within ``line_width`` px, and
    ``direction_mask`` (H, W) bool."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    shapely = _deps()[1]
    H, W = shape
    pts = []
    direction = np.zeros((2, H, W), np.float32)
    mask = np.zeros((H, W), bool)
    for p in polygons:
        s = p.simplify(tolerance)
        if s.geom_type != "Polygon" or s.is_empty:
            continue
        c = np.asarray(s.exterior.coords)[:-1]
        if len(c) >= 3:
            a, b = np.roll(c, 1, 0) - c, np.roll(c, -1, 0) - c
            cos = -(a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            pts.append(c[cos < math.cos(math.radians(turn_deg))])
        cc = np.asarray(s.exterior.coords)
        for u, v in zip(cc[:-1], cc[1:]):
            th = math.atan2(v[1] - u[1], v[0] - u[0])
            m = rasterize([(shapely.LineString([u, v]).buffer(line_width), 1)], out_shape=(H, W),
                          transform=Affine.identity(), dtype="uint8").astype(bool)
            direction[0][m] = math.cos(2 * th)
            direction[1][m] = math.sin(2 * th)
            mask |= m
    corner = np.zeros((H, W), np.float32)
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    rr = int(math.ceil(3 * sigma))
    for x, y in pts:
        c0, r0 = int(math.floor(x)) - rr, int(math.floor(y)) - rr
        cs, rs = max(c0, 0), max(r0, 0)
        ce, re = min(c0 + 2 * rr + 1, W), min(r0 + 2 * rr + 1, H)
        if cs >= ce or rs >= re:
            continue
        yy, xx = np.mgrid[rs:re, cs:ce]
        d2 = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2
        g = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        np.maximum(corner[rs:re, cs:ce], g, out=corner[rs:re, cs:ce])
    return {"corner": corner, "direction": direction, "direction_mask": mask}
