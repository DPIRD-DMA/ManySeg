"""E031 oracle: LoD2 roof faces from a roof-edge graph tensor (A1) vs height-model plane
segmentation (A0), with LoD1 as a reference.

Tile pixel coordinates follow manyseg.vectors: (x, y), pixel corners at integers. A plane is
(A, B, C) with height above ground z = A x + B y + C in metres, x and y in pixels.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

from manyseg.vectors import encode_graph_tensor, polygons_from_labels, polygons_from_tensor, vertex_graph
from manyseg_bench.evaluation.vectors import polygon_metrics

TILE, GSD = 512, 0.25
IDENT = Affine(1, 0, 0, 0, 1, 0)
MAX_TILES = 40

# graph tensor / mask corruption (E028 levels) and height-model corruption
LEVELS = {
    0: dict(occ=0.0, blur=0.0, noise=0.0, miss=0.0, vec=0.0, h_noise=0.0, h_blur=0.0, h_holes=0.0),
    1: dict(occ=0.5, blur=1.0, noise=0.1, miss=0.05, vec=1.0, h_noise=0.05, h_blur=0.0, h_holes=0.0),
    2: dict(occ=1.0, blur=1.5, noise=0.2, miss=0.10, vec=2.0, h_noise=0.15, h_blur=1.0, h_holes=0.0),
    3: dict(occ=2.0, blur=2.0, noise=0.3, miss=0.20, vec=3.0, h_noise=0.30, h_blur=2.0, h_holes=0.02),
}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def _plane_lsq(pts: np.ndarray) -> np.ndarray:
    """Least-squares z = a X + b Y + c through (K, 3) points."""
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    return np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]


def load_faces(area_dir: Path) -> list[dict]:
    """Every LoD2.2 RoofSurface: plan-view polygon (RD metres), world plane (a, b, c), ground z,
    building id."""
    faces = []
    for f in sorted(area_dir.glob("*.city.json")):
        d = json.loads(f.read_text())
        s, t = np.array(d["transform"]["scale"]), np.array(d["transform"]["translate"])
        V = np.array(d["vertices"], np.float64) * s + t
        for oid, o in d["CityObjects"].items():
            if o["type"] != "BuildingPart":
                continue
            g = next((g for g in o.get("geometry", []) if g["lod"] == "2.2"), None)
            if g is None or g["type"] != "Solid":
                continue
            surf = g["semantics"]["surfaces"]
            vals = g["semantics"]["values"][0]
            shell = g["boundaries"][0]
            ground = [V[np.concatenate([np.array(r) for r in face])] for face, v in zip(shell, vals)
                      if v is not None and surf[v]["type"] == "GroundSurface"]
            if not ground:
                continue
            gz = float(np.concatenate(ground)[:, 2].mean())
            for face, v in zip(shell, vals):
                if v is None or surf[v]["type"] != "RoofSurface":
                    continue
                rings = [V[np.array(r)] for r in face]
                if len(rings[0]) < 3:
                    continue
                poly = Polygon(rings[0][:, :2], [r[:, :2] for r in rings[1:] if len(r) >= 3])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type != "Polygon" or poly.area < 1.0:
                    continue
                faces.append({"poly": poly, "plane": _plane_lsq(np.concatenate(rings)), "ground": gz,
                              "building": o.get("parents", [oid])[0]})
    return faces


@dataclass
class Tile:
    area: str
    index: int
    x0: float
    y1: float
    faces: list = field(default_factory=list)  # shapely polygons in pixels
    planes: np.ndarray = None  # (F, 3) pixel planes (height above ground)
    height: np.ndarray = None  # (H, W) ground-truth height model
    mask: np.ndarray = None  # (H, W) bool


def make_tiles(area: str, faces: list[dict], max_tiles: int = MAX_TILES) -> list[Tile]:
    tree = shapely.STRtree([f["poly"] for f in faces])
    b = shapely.total_bounds(np.array([f["poly"] for f in faces]))
    size = TILE * GSD
    cands = []
    for y1 in np.arange(b[3], b[1] + size, -size):
        for x0 in np.arange(b[0], b[2] - size, size):
            win = box(x0, y1 - size, x0 + size, y1)
            idx = tree.query(win, predicate="intersects")
            if len({faces[i]["building"] for i in idx}) >= 3:
                cands.append((x0, y1, idx))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(cands), size=min(max_tiles, len(cands)), replace=False))
    out = []
    for k, ci in enumerate(pick):
        x0, y1, idx = cands[ci]
        t = Tile(area, k, x0, y1)
        win = box(x0, y1 - size, x0 + size, y1)
        polys, planes = [], []
        for i in idx:
            c = faces[i]["poly"].intersection(win)
            for part in getattr(c, "geoms", [c]):
                if part.geom_type != "Polygon" or part.area < 1.0:
                    continue
                px = shapely.transform(part, lambda xy: np.c_[(xy[:, 0] - x0) / GSD, (y1 - xy[:, 1]) / GSD])
                if not px.is_valid:
                    px = px.buffer(0)
                if px.geom_type != "Polygon" or px.area < 16:
                    continue
                a, bb, cc = faces[i]["plane"]
                polys.append(px)
                planes.append((a * GSD, -bb * GSD, a * x0 + bb * y1 + cc - faces[i]["ground"]))
        t.faces, t.planes = polys, np.array(planes).reshape(-1, 3)
        t.height, t.mask = height_map(polys, t.planes)
        out.append(t)
    return out


def height_map(polys, planes) -> tuple[np.ndarray, np.ndarray]:
    if not polys:
        return np.zeros((TILE, TILE), np.float32), np.zeros((TILE, TILE), bool)
    lab = rasterize([(p, i + 1) for i, p in enumerate(polys)], out_shape=(TILE, TILE),
                    transform=IDENT, dtype="int32")
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    P = np.vstack([[0, 0, 0], np.asarray(planes)])
    h = P[lab, 0] * (xx + 0.5) + P[lab, 1] * (yy + 0.5) + P[lab, 2]
    return np.where(lab > 0, h, 0).astype(np.float32), lab > 0


# --------------------------------------------------------------------------------------
# Targets and corruptions
# --------------------------------------------------------------------------------------


def _rng(seed, t: Tile, key: str):
    return np.random.default_rng([seed, zlib.crc32(t.area.encode()), t.index, zlib.crc32(key.encode())])


def encode(t: Tile) -> dict:
    vg = vertex_graph(t.faces, 20.0, border=(TILE, TILE))
    g = encode_graph_tensor(vg, (TILE, TILE))
    return {"vertex": g["vertex"], "edge": g["edge"], "vector": g["vector"],
            "mask": t.mask.astype(np.float32), "height": t.height.copy(), "_vertices": vg.xy}


def _disks(xy_r: np.ndarray) -> np.ndarray:
    m = np.zeros((TILE, TILE), bool)
    for x, y, r in xy_r:
        r0, r1 = max(0, int(y - r) - 1), min(TILE, int(y + r) + 2)
        c0, c1 = max(0, int(x - r) - 1), min(TILE, int(x + r) + 2)
        if r0 < r1 and c0 < c1:
            yy, xx = np.mgrid[r0:r1, c0:c1]
            m[r0:r1, c0:c1] |= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r
    return m


def _correlated(rng, shape, std, sigma=2.0):
    z = rng.normal(0, 1, shape).astype(np.float32)
    z = ndimage.gaussian_filter(z, (0, sigma, sigma) if len(shape) == 3 else sigma, mode="wrap")
    return z * (std / max(float(z.std()), 1e-9))


def corrupt(tg: dict, t: Tile, cfg: dict, seed: int) -> dict:
    out = {k: v.copy() for k, v in tg.items() if not k.startswith("_")}
    if cfg["occ"] > 0:  # disks along roof edges (E028: per 100 m of edge)
        segs = [(a, b) for p in t.faces for a, b in zip(np.asarray(p.exterior.coords)[:-1],
                                                         np.asarray(p.exterior.coords)[1:])]
        lens = np.array([np.hypot(*(b - a)) for a, b in segs])
        n = int(round(cfg["occ"] * lens.sum() * GSD / 100))
        rng = _rng(seed, t, "occ")
        if n and len(segs):
            cum = np.concatenate([[0], np.cumsum(lens)])
            s = rng.uniform(0, cum[-1], n)
            k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(segs) - 1)
            f = ((s - cum[k]) / np.maximum(lens[k], 1e-9))[:, None]
            a = np.array([segs[i][0] for i in k])
            bb = np.array([segs[i][1] for i in k])
            occ = _disks(np.c_[a + (bb - a) * f, rng.uniform(4, 12, n)])
            for key in ("vertex", "edge", "mask"):
                out[key][..., occ] = 0
    if cfg["miss"] > 0 and len(tg["_vertices"]):
        rng = _rng(seed, t, "miss")
        drop = tg["_vertices"][rng.random(len(tg["_vertices"])) < cfg["miss"]]
        if len(drop):
            out["vertex"][_disks(np.c_[drop, np.full(len(drop), 3.0)])] = 0
    for key in ("vertex", "edge", "mask"):
        a = out[key]
        if cfg["blur"] > 0:
            a = ndimage.gaussian_filter(a, (0, cfg["blur"], cfg["blur"]) if a.ndim == 3 else cfg["blur"])
        if cfg["noise"] > 0:
            a = a + _correlated(_rng(seed, t, "noise/" + key), a.shape, cfg["noise"])
        out[key] = np.clip(a, 0, 1).astype(np.float32)
    if cfg["vec"] > 0:
        out["vector"] = out["vector"] + _rng(seed, t, "vec").normal(0, cfg["vec"], out["vector"].shape).astype(np.float32)
    h = out["height"]
    if cfg["h_blur"] > 0:
        h = ndimage.gaussian_filter(h, cfg["h_blur"])
    if cfg["h_noise"] > 0:
        h = h + _rng(seed, t, "h_noise").normal(0, cfg["h_noise"], h.shape).astype(np.float32)
    if cfg["h_holes"] > 0:
        lost = _rng(seed, t, "h_holes").random(h.shape) < cfg["h_holes"]
        idx = ndimage.distance_transform_edt(lost, return_distances=False, return_indices=True)
        h = h[idx[0], idx[1]]
    out["height"] = h.astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------------------


def fit_plane(height: np.ndarray, region: np.ndarray, trim: float = 0.2) -> np.ndarray:
    """Trimmed least-squares plane (A, B, C) through the height pixels of a boolean region."""
    rr, cc = np.nonzero(region)
    if len(rr) < 3:
        return np.array([0.0, 0.0, float(np.median(height[region])) if len(rr) else 0.0])
    X = np.c_[cc + 0.5, rr + 0.5, np.ones(len(rr))]
    z = height[rr, cc].astype(np.float64)
    p = np.linalg.lstsq(X, z, rcond=None)[0]
    if trim > 0 and len(rr) > 10:
        res = np.abs(X @ p - z)
        keep = res <= np.quantile(res, 1 - trim)
        p = np.linalg.lstsq(X[keep], z[keep], rcond=None)[0]
    return p


def _face_region(poly: Polygon) -> np.ndarray:
    return rasterize([(poly, 1)], out_shape=(TILE, TILE), transform=IDENT, dtype="uint8").astype(bool)


def planes_for(faces: list, height: np.ndarray) -> np.ndarray:
    if not faces:
        return np.zeros((0, 3))
    lab = rasterize([(p, i + 1) for i, p in enumerate(faces)], out_shape=(TILE, TILE),
                    transform=IDENT, dtype="int32")
    return np.array([fit_plane(height, lab == i + 1) for i in range(len(faces))])


def _segment(mask: np.ndarray, h: np.ndarray, p: dict) -> np.ndarray:
    """A0 region labels: join 4-neighbours inside the mask with similar gradient and height."""
    hs = ndimage.gaussian_filter(h, p["smooth"]) if p["smooth"] > 0 else h
    gy, gx = np.gradient(hs)
    idx = -np.ones(mask.shape, np.int64)
    rr, cc = np.nonzero(mask)
    idx[rr, cc] = np.arange(len(rr))
    rows, cols = [], []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask[: TILE - dr, : TILE - dc] & mask[dr:, dc:]
        ok = a & (np.abs(gx[: TILE - dr, : TILE - dc] - gx[dr:, dc:]) < p["tg"]) \
            & (np.abs(gy[: TILE - dr, : TILE - dc] - gy[dr:, dc:]) < p["tg"]) \
            & (np.abs(hs[: TILE - dr, : TILE - dc] - hs[dr:, dc:]) < p["tz"])
        r, c = np.nonzero(ok)
        rows.append(idx[r, c])
        cols.append(idx[r + dr, c + dc])
    n = len(rr)
    if n == 0:
        return np.zeros(mask.shape, np.int32)
    g = coo_matrix((np.ones(sum(len(x) for x in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    lab = np.zeros(mask.shape, np.int32)
    lab[rr, cc] = comp + 1
    sizes = np.bincount(lab.ravel())
    small = (sizes < p["min_px"]) & (np.arange(len(sizes)) > 0)
    lab[small[lab]] = 0
    if (lab > 0).any():  # small regions join their nearest kept region
        _, (ir, ic) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(mask, lab[ir, ic], 0)
    return lab


def _merge_similar(lab: np.ndarray, h: np.ndarray, max_deg: float, max_dz: float) -> np.ndarray:
    """Merge 4-adjacent regions whose fitted planes have close normals and meet at the same height."""
    ids = [i for i in np.unique(lab) if i > 0]
    if len(ids) < 2:
        return lab
    planes = {i: fit_plane(h, lab == i, trim=0.0) for i in ids}
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = set()
    for dr, dc in ((0, 1), (1, 0)):
        a, b = lab[: TILE - dr, : TILE - dc], lab[dr:, dc:]
        m = (a > 0) & (b > 0) & (a != b)
        pairs |= set(zip(a[m].tolist(), b[m].tolist()))
    for i, j in pairs:
        pi, pj = planes[i], planes[j]
        ni = np.array([-pi[0] / GSD, -pi[1] / GSD, 1.0])
        nj = np.array([-pj[0] / GSD, -pj[1] / GSD, 1.0])
        ang = math.degrees(math.acos(np.clip(ni @ nj / np.linalg.norm(ni) / np.linalg.norm(nj), -1, 1)))
        if ang < max_deg:
            rr, cc = np.nonzero(lab == i)
            k = len(rr) // 2
            x, y = cc[k] + 0.5, rr[k] + 0.5
            if abs((pi[0] - pj[0]) * x + (pi[1] - pj[1]) * y + pi[2] - pj[2]) < max_dz:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    lut = np.zeros(lab.max() + 1, np.int32)
    for i in ids:
        lut[i] = find(i)
    return lut[lab]


def decode_a0(a: dict, p: dict) -> tuple[list, np.ndarray]:
    mask = a["mask"] > 0.5
    lab = _merge_similar(_segment(mask, a["height"], p), a["height"], 5.0, 0.3)
    faces = polygons_from_labels(lab, simplify=p["simplify"], min_area=16)
    return faces, planes_for(faces, a["height"])


def decode_a1(a: dict, p: dict) -> tuple[list, np.ndarray]:
    faces = polygons_from_tensor(a["vertex"], a["edge"], a["vector"], a["mask"],
                                 vertex_threshold=p["vertex_threshold"], link_radius=p["link_radius"],
                                 gap=p["gap"], simplify=p["simplify"], edge_threshold=0.5, kernel=3)
    return faces, planes_for(faces, a["height"])


def decode_lod1(a: dict, p: dict) -> tuple[list, np.ndarray]:
    lab, _ = ndimage.label(a["mask"] > 0.5)
    faces = polygons_from_labels(lab, simplify=1.0, min_area=16)
    reg = rasterize([(f, i + 1) for i, f in enumerate(faces)], out_shape=(TILE, TILE),
                    transform=IDENT, dtype="int32") if faces else np.zeros((TILE, TILE), np.int32)
    planes = np.array([[0.0, 0.0, float(np.median(a["height"][reg == i + 1])) if (reg == i + 1).any() else 0.0]
                       for i in range(len(faces))]).reshape(-1, 3)
    return faces, planes


DECODERS = {"A0": decode_a0, "A1": decode_a1, "LoD1": decode_lod1}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def _corners(polys, tol=0.5) -> np.ndarray:
    pts = [np.asarray(p.simplify(tol).exterior.coords)[:-1] for p in polys]
    if not pts:
        return np.zeros((0, 2))
    pts = np.concatenate(pts)
    keep, tree = [], cKDTree(pts)
    taken = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if not taken[i]:
            keep.append(pts[i])
            taken[tree.query_ball_point(pts[i], 1.0)] = True
    return np.array(keep)


def _match(a, b, r) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    cost = np.where(d <= r, d, 1e6)
    i, j = linear_sum_assignment(cost)
    return int((cost[i, j] <= r).sum())


def _normal(p) -> np.ndarray:
    n = np.array([-p[0] / GSD, -p[1] / GSD, 1.0])
    return n / np.linalg.norm(n)


def metrics(t: Tile, faces: list, planes: np.ndarray) -> dict:
    pm = polygon_metrics(t.faces, faces, gsd=GSD)
    pred_h, _ = height_map(faces, planes) if faces else (np.zeros((TILE, TILE), np.float32), None)
    err = (pred_h - t.height)[t.mask]
    gc, pc = _corners(t.faces), _corners(faces)
    tp_c = _match(gc, pc, 0.5 / GSD)
    # normal error over faces matched at IoU > 0.5 (greedy, as polygon_metrics)
    angs = []
    if faces:
        tree = shapely.STRtree(faces)
        cand = []
        for i, g in enumerate(t.faces):
            for j in tree.query(g, predicate="intersects"):
                inter = g.intersection(faces[j]).area
                iou = inter / (g.area + faces[j].area - inter) if inter > 0 else 0
                if iou > 0.5:
                    cand.append((iou, i, int(j)))
        used_g, used_p = set(), set()
        for iou, i, j in sorted(cand, reverse=True):
            if i not in used_g and j not in used_p:
                used_g.add(i)
                used_p.add(j)
                angs.append(math.degrees(math.acos(np.clip(_normal(t.planes[i]) @ _normal(planes[j]), -1, 1))))
    return {"face_f1": pm["det_f1"], "ciou": pm["ciou"], "tp": pm["tp"], "n_gt": pm["n_gt"], "n_pred": pm["n_pred"],
            "h_rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0,
            "h_mae": float(np.mean(np.abs(err))) if len(err) else 0.0,
            "corner_tp": tp_c, "corner_gt": len(gc), "corner_pred": len(pc),
            "corner_f1": 2 * tp_c / max(1, len(gc) + len(pc)),
            "normal_err_deg": float(np.mean(angs)) if angs else float("nan"), "n_normal": len(angs)}


def score(m: dict) -> float:
    """Tuning objective (README): face F1 x (1 - min(1, height RMSE / 1 m))."""
    return m["face_f1"] * (1 - min(1.0, m["h_rmse"] / 1.0))
