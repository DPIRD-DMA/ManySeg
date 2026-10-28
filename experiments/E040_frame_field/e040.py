"""E040: Frame Field Learning's frame field (Girard et al. 2021) as B3's direction head, oracle.

Frame field per pixel: complex c0, c2 with f(z) = z^4 + c2 z^2 + c0 = (z^2 - u^2)(z^2 - v^2); the roots
+-u, +-v are two unoriented directions (no sign / order ambiguity). Buildings, as in the paper:
u = wall tangent, v = its normal, on a 1 px band around each GT edge (c2 = 0, c0 = -e^{4i theta}).

Arms (all with B0's instances for closure, as B3r):
* ``heat``  - B3r (E035b): corner heatmap + (cos 2t, sin 2t) line field.
* ``field`` - no corner heatmap: corners where the outline switches from one frame direction to the
              other (FFL's corner rule); wall test uses the frame direction nearest the outline.
* ``both``  - heatmap corners plus field corners; wall test uses the frame field.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import shapely
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "E028_vector_oracle")]
import e028 as E  # noqa: E402
import e035 as B  # noqa: E402
from manyseg.vectors import polygons_from_labels  # noqa: E402


def frame_targets(polys, shape=(E.TILE, E.TILE)) -> np.ndarray:
    """(4, H, W): Re c0, Im c0, Re c2, Im c2 on a 1 px band around every GT edge, 0 elsewhere."""
    out = np.zeros((4, *shape), np.float32)
    for p in polys:
        c = np.asarray(p.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            th = math.atan2(b[1] - a[1], b[0] - a[0])
            m = rasterize([(LineString([a, b]).buffer(1.0), 1)], out_shape=shape, transform=E.IDENT,
                          dtype="uint8").astype(bool)
            u2, v2 = np.exp(2j * th), np.exp(2j * (th + math.pi / 2))
            c0, c2 = u2 * v2, -(u2 + v2)
            out[0][m], out[1][m], out[2][m], out[3][m] = c0.real, c0.imag, c2.real, c2.imag
    return out


def encode_b3f(t):
    tg = B.encode_b3(t)
    tg.arrays["frame"] = frame_targets(t.polygons)
    return tg


def corrupt_b3f(tg, t, cfg: dict, seed: int) -> dict:
    """B3's corruption (E035) + the frame field treated like the direction field: zeroed under the
    occlusion disks, N(0, 0.1 x vec) noise on each of the 4 channels."""
    out = B.corrupt_b3(tg, t, cfg, seed)
    f = tg.arrays["frame"].copy()
    if cfg["occ"] > 0:
        f[:, E._disk_mask(E.occlusion_disks(t, cfg["occ"], seed))] = 0
    if cfg["vec"] > 0:
        f = f + E._rng(seed, t, "frame").normal(0, 0.1 * cfg["vec"], f.shape).astype(np.float32)
    out["frame"] = f
    return out


def frame_dirs(f: np.ndarray, r: np.ndarray, c: np.ndarray):
    """Frame directions (angles of u and v, radians) and field strength |c0| at pixels (r, c)."""
    c0 = f[0, r, c] + 1j * f[1, r, c]
    c2 = f[2, r, c] + 1j * f[3, r, c]
    disc = np.sqrt(c2 * c2 - 4 * c0)
    u2, v2 = -0.5 * (c2 + disc), -0.5 * (c2 - disc)
    return np.angle(u2) / 2, np.angle(v2) / 2, np.abs(c0)


def ring_labels(ring: np.ndarray, f: np.ndarray, window: int, min_strength: float = 0.2):
    """Per outline point: which frame direction (0 = u, 1 = v) is nearest the local tangent
    (central difference over +-3 points), majority-smoothed over ``window``; and the chosen
    direction as (cos 2t, sin 2t). Weak field (|c0| < min_strength) -> label -1, direction 0."""
    K = len(ring)
    H, W = f.shape[1:]
    rc = np.clip(np.floor(ring).astype(int), 0, W - 1)
    # the traced mask outline sits ~0.5 px off the wall: sample the strongest field pixel in the
    # 3x3 neighbourhood (smoke-tile fix: the pixel under the point often missed the 1 px band)
    strength = np.hypot(f[0], f[1])
    best_r, best_c, best_s = rc[:, 1].copy(), rc[:, 0].copy(), np.full(K, -1.0)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r_ = np.clip(rc[:, 1] + dr, 0, H - 1); c_ = np.clip(rc[:, 0] + dc, 0, W - 1)
            sv = strength[r_, c_]
            better = sv > best_s
            best_r[better], best_c[better], best_s[better] = r_[better], c_[better], sv[better]
    tu, tv, s = frame_dirs(f, best_r, best_c)
    tan = ring[(np.arange(K) + 3) % K] - ring[(np.arange(K) - 3) % K]
    ta = np.arctan2(tan[:, 1], tan[:, 0])
    du = np.abs(np.cos(ta - tu))
    dv = np.abs(np.cos(ta - tv))
    lab = np.where(du >= dv, 0, 1)
    lab = np.where(s >= min_strength, lab, -1)
    half = window // 2
    sm = lab.copy()
    for k in range(K):  # circular majority vote over known labels
        w = lab[(np.arange(k - half, k + half + 1)) % K]
        w = w[w >= 0]
        if len(w):
            sm[k] = int(np.round(w.mean()))
    ang = np.where(sm == 0, tu, tv)
    dirs = np.stack([np.cos(2 * ang), np.sin(2 * ang)], 1)
    dirs[sm < 0] = 0
    return sm, dirs


def field_corners(labels: np.ndarray) -> list[int]:
    """Indices where the smoothed label switches between u and v (FFL's corner rule)."""
    K = len(labels)
    known = labels >= 0
    out = []
    prev = None
    for k in list(range(K)):
        if not known[k]:
            continue
        if prev is not None and labels[k] != labels[prev]:
            out.append(k)
        prev = k
    if out and known.any():  # wrap-around switch
        first = int(np.argmax(known))
        if labels[first] != labels[prev]:
            out.append(first)
    return sorted(set(out))


def polygon_b3f(ring: np.ndarray, idx: list[int], dirs: np.ndarray, p: dict) -> Polygon | None:
    """B3r's wall assembly (E035b `polygon_b3r`) from given corner indices and per-point directions."""
    ring = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    K = len(ring)
    if K < 4:
        return None
    idx = sorted({int(i) % K for i in idx})
    if len(idx) < 3:
        q = Polygon(ring).buffer(0).simplify(p["simplify"])
        return q if q.geom_type == "Polygon" else None
    verts = []
    for a, b in zip(idx, idx[1:] + [idx[0] + K]):
        seg = np.arange(a, b + 1)
        verts += [a] + [a + c for c in B._split_r(ring[seg % K], dirs[seg % K], p["big_dev"], p["agree"])]
    verts = sorted(set(verts))
    walls = []
    for a, b in zip(verts, verts[1:] + [verts[0] + K]):
        pts = ring[np.arange(a, b + 1) % K]
        if len(pts) >= 2 and not np.allclose(pts[0], pts[-1]):
            walls.append(B._trimmed_line(pts) + (a,))
    merged = True
    while merged and len(walls) > 3:
        merged = False
        for i in range(len(walls)):
            (_, u1, _), (_, u2, _) = walls[i - 1], walls[i]
            if math.degrees(math.acos(min(1.0, abs(float(u1 @ u2))))) < p["merge_deg"]:
                del walls[i]
                merged = True
                break
    if len(walls) < 3:
        q = Polygon(ring).buffer(0).simplify(p["simplify"])
        return q if q.geom_type == "Polygon" else None
    out = []
    for i in range(len(walls)):
        (p1, d1, _), (p2, d2, s2) = walls[i - 1], walls[i]
        x = B._intersect(p1, d1, p2, d2)
        corner = ring[s2 % K]
        out.append(x if x is not None and np.hypot(*(x - corner)) <= 3 * p["snap_r"] else corner)
    q = Polygon(out)
    if not q.is_valid:
        q = q.buffer(0)
        if q.geom_type == "MultiPolygon":
            q = max(q.geoms, key=lambda g: g.area)
    return q if q.geom_type == "Polygon" and q.area > 1 else None


def decode_b3f(a: dict, p: dict) -> list[Polygon]:
    """``p['mode']`` in {"field", "both"}; ``heat`` is E035b's decode_b3r."""
    lab = E._labels_b0(a, p["b0"])
    exact = polygons_from_labels(lab, simplify=0.0, min_area=16)
    pk = E._peaks(a["corner"], p["thr_c"], p["kernel"]) if p["mode"] == "both" else np.zeros((0, 2))
    out = []
    for poly in exact:
        ring = np.asarray(shapely.segmentize(poly.exterior, 1.0).coords)
        ring_c = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
        labels, dirs = ring_labels(ring_c, a["frame"], p["window"])
        idx = field_corners(labels)
        if len(pk):
            d, j = cKDTree(ring_c).query(pk)
            heat = [int(x) for x, dd in zip(j, d) if dd <= p["snap_r"]]
            # field corners within 3 outline points of a heatmap corner are the same corner
            idx = sorted(set(heat) | {k for k in idx if all(min(abs(k - h), len(ring_c) - abs(k - h)) > 3 for h in heat)})
        q = polygon_b3f(ring_c, idx, dirs, p)
        if q is not None:
            out.append(q)
    return out
