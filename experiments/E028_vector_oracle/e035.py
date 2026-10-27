"""E035 oracle: B3 = instance heads (closure, as B0) + corner map + wall direction field.

Targets on top of B0's (extent, boundary, distance): ``corner`` (Gaussian, sigma 2 px, at true
corners: vertices where the outline turns by more than 20 degrees) and ``direction`` (2 channels,
``(cos 2t, sin 2t)`` of the wall direction at outline pixels, 0 elsewhere; orientation-invariant).

Decoder: instances as B0 -> each instance's exact pixel-edge outline -> corner peaks snapped onto
it split it into walls -> each wall is one straight line unless its outline points stray more
than ``dev`` px from the corner-to-corner chord or the predicted direction disagrees with the
chord (then split at the farthest point, recursively) -> consecutive wall lines intersected for
the final corners. A missed corner therefore costs a split, not the building.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon

import e028 as E
from manyseg.vectors import polygons_from_labels

TURN_DEG = 20.0


def true_corners(p: Polygon, tol: float = 0.5) -> np.ndarray:
    c = np.asarray(p.simplify(tol).exterior.coords)[:-1]
    if len(c) < 3:
        return c
    a, b = np.roll(c, 1, 0) - c, np.roll(c, -1, 0) - c
    cos = -(a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return c[cos < math.cos(math.radians(TURN_DEG))]


def encode_b3(t) -> dict:
    """B0 targets + corner map + direction field (and the corner points, for 'missed peaks')."""
    base = E.encode(t, "B0")
    corners = [true_corners(p) for p in t.polygons]
    cpts = np.concatenate([c for c in corners if len(c)]) if any(len(c) for c in corners) else np.zeros((0, 2))
    corner = E.gaussian_points(cpts, 2.0)
    direction = np.zeros((2, E.TILE, E.TILE), np.float32)
    for p in t.polygons:
        c = np.asarray(p.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            th = math.atan2(b[1] - a[1], b[0] - a[0])
            m = rasterize([(LineString([a, b]).buffer(1.0), 1)], out_shape=(E.TILE, E.TILE),
                          transform=E.IDENT, dtype="uint8").astype(bool)
            direction[0][m] = math.cos(2 * th)
            direction[1][m] = math.sin(2 * th)
    arrays = dict(base.arrays, corner=corner, direction=direction)
    return E.Targets(arrays, base.prob + ("corner",), {"corner": cpts}, ())


def corrupt_b3(tg, t, cfg: dict, seed: int) -> dict:
    """E028's corruption for the B0 channels and the corner map (occlusion, blur, noise, missed
    peaks); the direction field is zeroed under the same occlusion disks and gets Gaussian noise of
    std 0.1 x the level's vector noise (0.1 / 0.2 / 0.3 on values in [-1, 1])."""
    out = E.corrupt(tg, t, cfg, seed)
    d = tg.arrays["direction"].copy()
    if cfg["occ"] > 0:
        d[:, E._disk_mask(E.occlusion_disks(t, cfg["occ"], seed))] = 0
    if cfg["vec"] > 0:
        d = d + E._rng(seed, t, "direction").normal(0, 0.1 * cfg["vec"], d.shape).astype(np.float32)
    out["direction"] = d
    return out


def _split(points: np.ndarray, dirs: np.ndarray, dev: float, agree: float) -> list[int]:
    """Indices (into ``points``) where a wall between its first and last point must be split:
    recursively at the farthest point while that point is more than ``dev`` px off the chord, or
    while the predicted directions disagree with the chord (mean agreement < ``agree``)."""
    if len(points) < 3:
        return []
    a, b = points[0], points[-1]
    ab = b - a
    L = np.hypot(*ab)
    if L < 1e-6:
        return []
    rel = points - a
    off = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
    th = math.atan2(ab[1], ab[0])
    if dirs is not None and len(dirs):
        agreement = float(np.mean((1 + dirs[:, 0] * math.cos(2 * th) + dirs[:, 1] * math.sin(2 * th)) / 2))
    else:
        agreement = 1.0
    k = int(np.argmax(off))
    if off[k] <= dev and agreement >= agree:
        return []
    if off[k] <= 1.0:  # nothing to split on (direction noise only)
        return []
    left = _split(points[: k + 1], None if dirs is None else dirs[: k + 1], dev, agree)
    right = _split(points[k:], None if dirs is None else dirs[k:], dev, agree)
    return left + [k] + [k + i for i in right]


def _intersect(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + s * d1


def polygon_b3(ring: np.ndarray, corner_pts: np.ndarray, dir_field: np.ndarray, p: dict) -> Polygon | None:
    """One instance's polygon from its dense outline ring (K, 2), corner peaks and directions."""
    ring = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    K = len(ring)
    if K < 4:
        return None
    # corner peaks near this outline, snapped to their nearest ring point
    idx = []
    if len(corner_pts):
        d, j = cKDTree(ring).query(corner_pts)
        idx = sorted({int(x) for x, dd in zip(j, d) if dd <= p["snap_r"]})
    if len(idx) < 3:
        q = Polygon(ring).buffer(0).simplify(p["simplify"])
        return q if q.geom_type == "Polygon" else None
    # directions sampled at ring points (pixel under each point, nudged inside the image)
    rc = np.clip(np.floor(ring).astype(int), 0, E.TILE - 1)
    dirs = dir_field[:, rc[:, 1], rc[:, 0]].T  # (K, 2)
    verts = []
    for a, b in zip(idx, idx[1:] + [idx[0] + K]):
        seg = np.arange(a, b + 1) % K
        pts, ds = ring[seg], dirs[seg]
        cuts = _split(pts, ds, p["dev"], p["agree"])
        verts += [seg[0]] + [seg[c] for c in cuts]
    verts = [v for i, v in enumerate(verts) if v != verts[i - 1]]
    if len(verts) < 3:
        return None
    # fit a line to each wall's ring points, intersect consecutive lines for the final corners
    lines = []
    for a, b in zip(verts, verts[1:] + [verts[0] + K]):
        pts = ring[np.arange(a, b + 1) % K]
        if len(pts) < 2 or np.allclose(pts[0], pts[-1]):
            continue  # degenerate wall (two snapped corners on one ring point)
        c = pts.mean(0)
        u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        lines.append((c, u, a))
    if len(lines) < 3:
        return None
    out = []
    for i in range(len(lines)):
        (p1, d1, s1), (p2, d2, s2) = lines[i - 1], lines[i]
        x = _intersect(p1, d1, p2, d2)
        corner = ring[s2 % K]
        out.append(x if x is not None and np.hypot(*(x - corner)) <= 3 * p["snap_r"] else corner)
    q = Polygon(out)
    if not q.is_valid:
        q = q.buffer(0)
        if q.geom_type == "MultiPolygon":
            q = max(q.geoms, key=lambda g: g.area)
    return q if q.geom_type == "Polygon" and q.area > 1 else None


def decode_b3(a: dict, p: dict) -> list[Polygon]:
    lab = E._labels_b0(a, p["b0"])
    exact = polygons_from_labels(lab, simplify=0.0, min_area=16)
    pk = E._peaks(a["corner"], p["thr_c"], p["kernel"])
    out = []
    for poly in exact:
        ring = np.asarray(shapely.segmentize(poly.exterior, 1.0).coords)  # a point every <= 1 px
        q = polygon_b3(ring, pk, a["direction"], p)
        if q is not None:
            out.append(q)
    return out
