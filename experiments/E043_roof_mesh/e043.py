"""E043: roof meshes from graph faces + per-face planes (user's idea), oracle on E031's 3DBAG tiles.

Faces come from E031b's hybrid decoder (graph + plane-segmentation fallback) on the corrupted
inputs, once per tile, so every plane arm and join gets identical faces. Planes (A, B, C) with
z = A x + B y + C (metres above ground, x / y in pixels, as E031):
* ``fit``   - E031's trimmed least-squares fit to the height raster over the whole face (baseline)
* ``robust``- height raster only: the face eroded by 2 px (edge pixels blur across ridges and eaves),
              trimmed fit (20%) repeated twice
* ``slope`` - height + two slope rasters (dz/dx, dz/dy per pixel): A, B = medians of the slope
              rasters over the eroded face; C = median of (h - A x - B y) there
Joins (mesh vertices from the faces' rings):
* ``J1`` - each 2D vertex shared by several faces takes the mean of its planes' heights if they
           agree within ``step`` m (watertight); otherwise each face keeps its own height (a step
           wall)
* ``J2`` - as J1, but a shared vertex first moves (at most ``cap`` px) to the point closest to where
           its faces' planes meet (regularised least squares), so ridges sit on plane intersections
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shapely
from rasterio.features import rasterize
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E031_lod2_oracle"))
import e031 as E  # noqa: E402

STEP_M, CAP_PX, ERODE = 0.5, 2.0, 2

# extra corruption per level: low-frequency height ramps (m) and slope-raster noise (m / px)
EXTRA = {0: dict(h_ramp=0.0, s_noise=0.0), 1: dict(h_ramp=0.1, s_noise=0.02),
         2: dict(h_ramp=0.3, s_noise=0.05), 3: dict(h_ramp=0.5, s_noise=0.1)}


def slope_targets(t) -> np.ndarray:
    """(2, H, W) dz/dx, dz/dy of the GT face under each pixel (0 off-roof)."""
    lab = rasterize([(p, i + 1) for i, p in enumerate(t.faces)], out_shape=(E.TILE, E.TILE),
                    transform=E.IDENT, dtype="int32")
    P = np.vstack([[0, 0, 0], np.asarray(t.planes)])
    return np.stack([P[lab, 0], P[lab, 1]]).astype(np.float32)


def corrupt43(t, level: int, seed: int) -> dict:
    """E031's corruption (graph tensor, mask, height) + a low-frequency height ramp field and noisy,
    blurred slope rasters (same blur and holes treatment as the height)."""
    cfg = E.LEVELS[level]
    a = E.corrupt(E.encode(t), t, cfg, seed)
    ex = EXTRA[level]
    if ex["h_ramp"] > 0:
        f = ndimage.gaussian_filter(E._rng(seed, t, "ramp").normal(0, 1, a["height"].shape), 40.0)
        a["height"] = (a["height"] + ex["h_ramp"] * f / max(f.std(), 1e-9) * t.mask).astype(np.float32)
    s = slope_targets(t)
    if cfg["h_blur"] > 0:
        s = ndimage.gaussian_filter(s, (0, cfg["h_blur"], cfg["h_blur"]))
    if ex["s_noise"] > 0:
        s = s + E._rng(seed, t, "slope").normal(0, ex["s_noise"], s.shape).astype(np.float32)
    a["slope"] = s.astype(np.float32)
    return a


def _region(face, erode: int) -> np.ndarray:
    m = E._face_region(face)
    if erode:
        e = ndimage.binary_erosion(m, iterations=erode)
        if e.sum() >= 10:
            return e
    return m


def planes(faces, a: dict, arm: str) -> np.ndarray:
    if arm == "fit":
        return E.planes_for(faces, a["height"])
    out = []
    for f in faces:
        reg = _region(f, ERODE)
        if arm == "robust":
            p = E.fit_plane(a["height"], reg, trim=0.2)
            rr, cc = np.nonzero(reg)
            if len(rr) > 20:  # second trimming round on the refit residuals
                X = np.c_[cc + 0.5, rr + 0.5, np.ones(len(rr))]
                z = a["height"][rr, cc].astype(np.float64)
                res = np.abs(X @ p - z)
                keep = res <= np.quantile(res, 0.8)
                p = np.linalg.lstsq(X[keep], z[keep], rcond=None)[0]
        else:  # slope
            rr, cc = np.nonzero(reg)
            if len(rr) == 0:
                p = np.zeros(3)
            else:
                A = float(np.median(a["slope"][0, rr, cc]))
                B = float(np.median(a["slope"][1, rr, cc]))
                C = float(np.median(a["height"][rr, cc] - A * (cc + 0.5) - B * (rr + 0.5)))
                p = np.array([A, B, C])
        out.append(p)
    return np.array(out).reshape(-1, 3)


def _vertices(faces):
    """Shared 2D vertices: key (rounded x, y) -> list of (face index, position in ring)."""
    shared: dict = {}
    for i, f in enumerate(faces):
        for k, (x, y) in enumerate(np.asarray(f.exterior.coords)[:-1]):
            shared.setdefault((round(x, 2), round(y, 2)), []).append((i, k))
    return shared


def mesh(faces, P: np.ndarray, join: str):
    """Per face: ring vertices (K, 3) in (x px, y px, z m); plus join statistics."""
    shared = _vertices(faces)
    rings = [np.c_[np.asarray(f.exterior.coords)[:-1], np.zeros(len(f.exterior.coords) - 1)] for f in faces]
    steps = 0
    for (x, y), users in shared.items():
        fids = sorted({i for i, _ in users})
        xy = np.array([x, y], float)
        if join == "J2" and len(fids) >= 2:
            # min sum_i (A_i x + B_i y + C_i - z)^2 + lam |(x, y) - xy0|^2 over (x, y, z)
            lam = 0.05
            rows, rhs = [], []
            for i in fids:
                A, B, C = P[i]
                rows.append([A, B, -1.0]); rhs.append(-C)
            rows += [[np.sqrt(lam), 0, 0], [0, np.sqrt(lam), 0]]
            rhs += [np.sqrt(lam) * xy[0], np.sqrt(lam) * xy[1]]
            sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
            d = sol[:2] - xy
            n = float(np.hypot(*d))
            xy = xy + (d if n <= CAP_PX else d * CAP_PX / n)
        zs = {i: float(P[i, 0] * xy[0] + P[i, 1] * xy[1] + P[i, 2]) for i in fids}
        if len(fids) >= 2 and max(zs.values()) - min(zs.values()) <= STEP_M:
            zm = float(np.mean(list(zs.values())))
            zs = {i: zm for i in fids}
        elif len(fids) >= 2:
            steps += 1
        for i, k in users:
            rings[i][k] = [xy[0], xy[1], zs[i]]
    return rings, {"step_vertices": steps, "shared_vertices": sum(1 for u in shared.values() if len({i for i, _ in u}) >= 2)}


def mesh_metrics(t, faces, rings, P) -> dict:
    """Against the GT *planes* (not the height raster, which at a boundary pixel belongs to only one
    of the meeting faces): vertex height error = min over GT faces within 1.5 px of the vertex of
    |z - GT plane there|; a step vertex (faces disagree by > STEP_M) is false if the GT planes within
    1.5 px agree within STEP_M there; mean_gap = mean dz between faces at shared vertices."""
    gtree = shapely.STRtree(t.faces)
    errs, false_steps, gaps, steps = [], 0, [], 0
    for (x, y), users in _vertices(faces).items():
        pt = shapely.Point(x, y)
        near = [int(j) for j in gtree.query(pt, predicate="dwithin", distance=1.5)]
        gz = [float(t.planes[j][0] * x + t.planes[j][1] * y + t.planes[j][2]) for j in near]
        zs = [rings[i][k][2] for i, k in users]
        if gz:
            errs.append(min(abs(float(np.mean(zs)) - z) for z in gz) if len({i for i, _ in users}) < 2
                        else float(np.mean([min(abs(z - g) for g in gz) for z in zs])))
        if len({i for i, _ in users}) >= 2:
            gaps.append(max(zs) - min(zs))
            if max(zs) - min(zs) > STEP_M:
                steps += 1
                if gz and max(gz) - min(gz) <= STEP_M:
                    false_steps += 1
    return {"vertex_z_err": float(np.mean(errs)) if errs else float("nan"), "n_vertices": len(errs),
            "false_steps": false_steps, "steps": steps, "mean_gap": float(np.mean(gaps)) if gaps else 0.0}
