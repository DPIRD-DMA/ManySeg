"""E028 oracle: encoders, corruptions, decoders and metrics for road graphs and footprints.

Geometry is in tile pixel coordinates ``(x, y)`` as floats: pixel ``(row, col)`` covers
``[col, col + 1) x [row, row + 1)``, so its centre is ``(col + 0.5, row + 0.5)``. Graphs are
``networkx.Graph`` with a node attribute ``xy`` (np.ndarray of 2), edges straight segments.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
import torch
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, skeletonize

from manyseg.instances import instance_targets, instances_from_outputs
from manyseg.points import find_peaks

TILE = 1024
GSD = {"roads": 1.0, "buildings": 0.5}
STRIDE = {"roads": 512, "buildings": 1024}
ROAD_WIDTH_M = {"motorway": 10, "trunk": 9, "primary": 9, "secondary": 8, "tertiary": 7,
                "residential": 6}
SECTORS = 6
VERTEX_SPACING = 20.0  # px, graph-tensor vertex spacing
VERTEX_RADIUS = 3.0  # px, graph-tensor support around each vertex
IDENT = Affine(1, 0, 0, 0, 1, 0)

# Corruption levels (README table).
LEVELS = {
    0: dict(occ=0.0, blur=0.0, noise=0.0, miss=0.0, vec=0.0),
    1: dict(occ=0.5, blur=1.0, noise=0.1, miss=0.05, vec=1.0),
    2: dict(occ=1.0, blur=1.5, noise=0.2, miss=0.10, vec=2.0),
    3: dict(occ=2.0, blur=2.0, noise=0.3, miss=0.20, vec=3.0),
}
FACTORS = ("occ", "blur", "noise", "miss", "vec")


def factor_only(name: str) -> dict:
    """Level-2 setting of one corruption family alone (secondary analysis)."""
    return {k: (LEVELS[2][k] if k == name else 0.0) for k in FACTORS}


# --------------------------------------------------------------------------------------
# Areas and tiles
# --------------------------------------------------------------------------------------


@dataclass
class Tile:
    area: str
    task: str  # "roads" | "buildings"
    index: int
    x0: float  # world (MGA) of the tile's left edge
    y1: float  # world of the tile's top edge
    gsd: float
    graph: nx.Graph | None = None  # roads
    crossings: list = field(default_factory=list)  # roads: GT grade-separated crossings (xy)
    polygons: list = field(default_factory=list)  # buildings

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.x0) / self.gsd, (self.y1 - xy[..., 1]) / self.gsd], -1)


def inner_box(bbox_wgs84, crs: str) -> tuple[float, float, float, float]:
    s, w, n, e = bbox_wgs84
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tf.transform([w, e, e, w], [s, s, n, n])
    return max(xs[0], xs[3]), max(ys[0], ys[1]), min(xs[1], xs[2]), min(ys[2], ys[3])


def tile_origins(bounds, task: str) -> list[tuple[float, float]]:
    xmin, ymin, xmax, ymax = bounds
    size = TILE * GSD[task]
    step = STRIDE[task] * GSD[task]
    nx_ = int((xmax - xmin - size) // step) + 1
    ny_ = int((ymax - ymin - size) // step) + 1
    return [(xmin + i * step, ymax - j * step) for j in range(ny_) for i in range(nx_)]


def _clip_t(p, q, lo=0.0, hi=float(TILE)):
    """Liang-Barsky: parameter range [t0, t1] of segment p->q inside the square, or None."""
    t0, t1 = 0.0, 1.0
    d = q - p
    for k in range(2):
        for pk, qk in ((-d[k], p[k] - lo), (d[k], hi - p[k])):
            if pk == 0:
                if qk < 0:
                    return None
            else:
                t = qk / pk
                if pk < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
    return (t0, t1) if t0 < t1 else None


def load_area(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def road_tiles(area_name: str, data: dict) -> list[Tile]:
    bounds = inner_box(data["bbox_wgs84"], data["crs"])
    nodes: dict[int, np.ndarray] = {}
    edges = []  # (u, v, bridge, layer, width_m)
    for r in data["roads"]:
        hw = r["highway"].removesuffix("_link")
        width = ROAD_WIDTH_M.get(hw, 5) if not r["highway"].endswith("_link") else 5
        for nid, xy in zip(r["nodes"], r["coords"]):
            nodes[nid] = np.asarray(xy)
        for a, b in zip(r["nodes"][:-1], r["nodes"][1:]):
            if a != b:
                edges.append((a, b, r["bridge"], r["layer"], width))
    tiles = []
    for i, (x0, y1) in enumerate(tile_origins(bounds, "roads")):
        t = Tile(area_name, "roads", i, x0, y1, GSD["roads"])
        g = nx.Graph()
        for k, (a, b, bridge, layer, width) in enumerate(edges):
            p, q = t.to_px(nodes[a]), t.to_px(nodes[b])
            tt = _clip_t(p, q)
            if tt is None:
                continue
            ua = a if tt[0] == 0 else ("b", k, 0)
            vb = b if tt[1] == 1 else ("b", k, 1)
            pa, qb = p + tt[0] * (q - p), p + tt[1] * (q - p)
            if np.allclose(pa, qb):
                continue
            g.add_node(ua, xy=pa)
            g.add_node(vb, xy=qb)
            g.add_edge(ua, vb, bridge=bridge, layer=layer, width=width / t.gsd)
        if g.number_of_edges() == 0:
            continue
        t.graph = g
        t.crossings = _crossings(g)
        tiles.append(t)
    return tiles


def _crossings(g: nx.Graph) -> list[np.ndarray]:
    """Points where two GT edges cross without sharing a node (bridges, overpasses)."""
    es = list(g.edges())
    lines = [LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]) for u, v in es]
    tree = shapely.STRtree(lines)
    left, right = tree.query(lines, predicate="crosses")
    pts = []
    for i, j in zip(left, right):
        if i >= j or set(es[i]) & set(es[j]):
            continue
        p = lines[i].intersection(lines[j])
        if p.geom_type == "Point":
            pts.append(np.array([p.x, p.y]))
    # merge crossings within 5 px (a bridge crossing a dual carriageway counts per carriageway)
    out: list[np.ndarray] = []
    for p in pts:
        if all(np.hypot(*(p - o)) > 5 for o in out):
            out.append(p)
    return out


def building_tiles(area_name: str, data: dict, min_buildings: int = 5) -> list[Tile]:
    bounds = inner_box(data["bbox_wgs84"], data["crs"])
    polys = []
    for b in data["buildings"]:
        p = Polygon(b["coords"])
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == "Polygon" and p.area > 4:  # > 4 m^2
            polys.append(p)
    # drop buildings overlapping a larger one by > 10% of their area (duplicates, parts)
    tree = shapely.STRtree(polys)
    keep = np.ones(len(polys), bool)
    order = np.argsort([-p.area for p in polys])
    for i in order:
        if not keep[i]:
            continue
        for j in tree.query(polys[i], predicate="intersects"):
            if j != i and keep[j] and polys[j].area <= polys[i].area:
                if polys[i].intersection(polys[j]).area > 0.1 * polys[j].area:
                    keep[j] = False
    polys = [p for p, k in zip(polys, keep) if k]
    tree = shapely.STRtree(polys)
    tiles = []
    for i, (x0, y1) in enumerate(tile_origins(bounds, "buildings")):
        t = Tile(area_name, "buildings", i, x0, y1, GSD["buildings"])
        size = TILE * t.gsd
        wbox = box(x0, y1 - size, x0 + size, y1)
        out = []
        for j in tree.query(wbox, predicate="intersects"):
            c = polys[j].intersection(wbox)
            if c.geom_type != "Polygon":
                continue
            px = Polygon(t.to_px(np.asarray(c.exterior.coords)))
            if not px.is_valid:
                px = px.buffer(0)
            if px.geom_type == "Polygon" and px.area >= 16:  # >= 4 m^2 at 0.5 m
                out.append(px)
        if len(out) >= min_buildings:
            t.polygons = out
            tiles.append(t)
    return tiles


# --------------------------------------------------------------------------------------
# Graph utilities
# --------------------------------------------------------------------------------------


def graph_chains(g: nx.Graph) -> list[list]:
    """Node paths between key nodes (degree != 2); isolated cycles start and end at one node."""
    key = {n for n in g if g.degree(n) != 2}
    seen = set()
    chains = []
    for k in key:
        for nb in g.neighbors(k):
            if frozenset((k, nb)) in seen:
                continue
            path = [k, nb]
            seen.add(frozenset((k, nb)))
            while path[-1] not in key:
                cur, prev = path[-1], path[-2]
                nxt = [n for n in g.neighbors(cur) if n != prev]
                if not nxt:  # cannot happen for degree 2, kept for safety
                    break
                seen.add(frozenset((cur, nxt[0])))
                path.append(nxt[0])
                if path[-1] == k and len(path) > 2:
                    break
            chains.append(path)
    for u, v in g.edges():
        if frozenset((u, v)) in seen:
            continue
        path = [u, v]  # a cycle of degree-2 nodes
        seen.add(frozenset((u, v)))
        while path[-1] != u:
            cur, prev = path[-1], path[-2]
            nxt = [n for n in g.neighbors(cur) if n != prev]
            if not nxt:
                break
            if frozenset((cur, nxt[0])) in seen and nxt[0] != u:
                break
            seen.add(frozenset((cur, nxt[0])))
            path.append(nxt[0])
        chains.append(path)
    return chains


def graph_length(g: nx.Graph) -> float:
    return sum(float(np.hypot(*(g.nodes[u]["xy"] - g.nodes[v]["xy"]))) for u, v in g.edges())


def polylines_to_graph(nodes: dict, chains: list[tuple]) -> nx.Graph:
    """``chains``: (u, v, coords (K, 2)) with coords[0] at u and coords[-1] at v."""
    g = nx.Graph()
    for n, xy in nodes.items():
        g.add_node(n, xy=np.asarray(xy, float))
    for ci, (u, v, coords) in enumerate(chains):
        prev = u
        for k in range(1, len(coords) - 1):
            nid = ("c", ci, k)
            g.add_node(nid, xy=np.asarray(coords[k], float))
            g.add_edge(prev, nid)
            prev = nid
        if prev != v:
            g.add_edge(prev, v)
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    return g


def _segments(g: nx.Graph):
    es = list(g.edges())
    p = np.array([g.nodes[u]["xy"] for u, _ in es]).reshape(-1, 2)
    q = np.array([g.nodes[v]["xy"] for _, v in es]).reshape(-1, 2)
    return es, p, q


# --------------------------------------------------------------------------------------
# Rasterization helpers and encoders
# --------------------------------------------------------------------------------------


def gaussian_points(pts: np.ndarray, sigma: float, shape=(TILE, TILE)) -> np.ndarray:
    """Max-composited Gaussians at float pixel coordinates ``(x, y)``."""
    out = np.zeros(shape, np.float32)
    r = int(math.ceil(3 * sigma))
    for x, y in np.asarray(pts).reshape(-1, 2):
        c0, r0 = int(math.floor(x)) - r, int(math.floor(y)) - r
        cs = np.arange(c0, c0 + 2 * r + 1)
        rs = np.arange(r0, r0 + 2 * r + 1)
        cm, rm = (cs >= 0) & (cs < shape[1]), (rs >= 0) & (rs < shape[0])
        if not cm.any() or not rm.any():
            continue
        dx = (cs[cm] + 0.5 - x)[None, :]
        dy = (rs[rm] + 0.5 - y)[:, None]
        val = np.exp(-(dx**2 + dy**2) / (2 * sigma**2)).astype(np.float32)
        sl = (slice(rs[rm][0], rs[rm][-1] + 1), slice(cs[cm][0], cs[cm][-1] + 1))
        np.maximum(out[sl], val, out=out[sl])
    return out


def road_mask(t: Tile) -> np.ndarray:
    g = t.graph
    shapes = [(LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]).buffer(d["width"] / 2), 1)
              for u, v, d in g.edges(data=True)]
    return rasterize(shapes, out_shape=(TILE, TILE), transform=IDENT, dtype="uint8").astype(np.float32)


def key_points(g: nx.Graph) -> np.ndarray:
    return np.array([g.nodes[n]["xy"] for n in g if g.degree(n) != 2]).reshape(-1, 2)


@dataclass
class VertexGraph:
    xy: np.ndarray  # (V, 2)
    edges: np.ndarray  # (E, 2) vertex indices
    unrepresented: int = 0  # GT edge ends that found no free sector
    total_ends: int = 0


def _densify(coords: np.ndarray, spacing: float) -> np.ndarray:
    out = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def vertex_graph(lines: list[np.ndarray], spacing=VERTEX_SPACING, merge=1.0,
                 planar: bool = False) -> VertexGraph:
    """Simplify (RDP 1 px) and densify polylines, then merge vertices within ``merge`` px.

    ``planar=False`` (roads) merges only polyline end points, i.e. shared GT nodes, so roads
    that cross without a junction stay apart. ``planar=True`` (footprints) merges any vertices,
    so adjacent buildings share corners.
    """
    pts, edges, ends = [], [], []
    for c in lines:
        c = np.asarray(c, float)
        closed = len(c) > 3 and np.allclose(c[0], c[-1])
        geom = Polygon(c).exterior if closed else LineString(c)
        s = np.asarray(geom.simplify(1.0, preserve_topology=False).coords)
        if len(s) < 2:
            continue
        d = _densify(s, spacing)
        base = len(pts)
        ends += [base, base + len(d) - 1]
        pts.extend(d)
        edges.extend((base + k, base + k + 1) for k in range(len(d) - 1))
    if not pts:
        return VertexGraph(np.zeros((0, 2)), np.zeros((0, 2), int))
    pts = np.array(pts)
    # union-find merge of near-coincident vertices
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cand = np.arange(len(pts)) if planar else np.unique(ends)
    for i, j in cKDTree(pts[cand]).query_pairs(merge):
        i, j = cand[i], cand[j]
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(pts))])
    uniq, inv = np.unique(roots, return_inverse=True)
    xy = pts[uniq]
    e = {tuple(sorted((inv[a], inv[b]))) for a, b in edges if inv[a] != inv[b]}
    return VertexGraph(xy, np.array(sorted(e), int).reshape(-1, 2))


def graph_lines(g: nx.Graph) -> list[np.ndarray]:
    return [np.array([g.nodes[n]["xy"] for n in ch]) for ch in graph_chains(g)]


def sector_of(angle: np.ndarray) -> np.ndarray:
    return (np.floor((angle % (2 * np.pi)) / (2 * np.pi / SECTORS)).astype(int)) % SECTORS


def graph_tensor(vg: VertexGraph, shape=(TILE, TILE)) -> dict[str, np.ndarray]:
    """Sat2Graph-style tensor: ``vertex`` (H, W), ``edge`` (S, H, W), ``vec`` (S, 2, H, W)."""
    V = len(vg.xy)
    nb = np.zeros((V + 1, SECTORS, 2), np.float32)  # row V: "no vertex"
    has = np.zeros((V + 1, SECTORS), bool)
    width = 2 * np.pi / SECTORS
    unrep = 0
    adj: list[list[int]] = [[] for _ in range(V)]
    for a, b in vg.edges:
        adj[a].append(b)
        adj[b].append(a)
    for v in range(V):
        for u in adj[v]:
            d = vg.xy[u] - vg.xy[v]
            ang = math.atan2(d[1], d[0]) % (2 * np.pi)
            own = int(ang // width) % SECTORS
            # own sector, then the adjacent sector on the nearer side, then the other one
            frac = ang / width - own
            order = [own, (own + 1) % SECTORS, (own - 1) % SECTORS] if frac >= 0.5 else \
                [own, (own - 1) % SECTORS, (own + 1) % SECTORS]
            for k in order:
                if not has[v, k]:
                    has[v, k] = True
                    nb[v, k] = vg.xy[u]
                    break
            else:
                unrep += 1
    vg.unrepresented, vg.total_ends = unrep, 2 * len(vg.edges)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    pix = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)
    if V:
        dist, idx = cKDTree(vg.xy).query(pix, distance_upper_bound=VERTEX_RADIUS)
    else:
        dist, idx = np.full(len(pix), np.inf), np.zeros(len(pix), int)
    inside = np.isfinite(dist)
    idx = np.where(inside, idx, V)
    vertex = np.where(inside, np.exp(-dist**2 / (2 * 1.5**2)), 0).astype(np.float32)
    edge = has[idx].T.astype(np.float32)  # (S, N)
    vec = (nb[idx] - pix[:, None, :].astype(np.float32)) * has[idx][..., None]
    return {"vertex": vertex.reshape(H, W), "edge": edge.reshape(SECTORS, H, W),
            "vec": vec.transpose(1, 2, 0).reshape(SECTORS, 2, H, W)}


def instance_raster(polys: list[Polygon]) -> np.ndarray:
    if not polys:
        return np.zeros((TILE, TILE), np.int32)
    return rasterize([(p, i + 1) for i, p in enumerate(polys)], out_shape=(TILE, TILE),
                     transform=IDENT, dtype="int32")


def gt_polygon_vertices(p: Polygon, tol: float = 0.5) -> np.ndarray:
    return np.asarray(p.simplify(tol).exterior.coords)[:-1]


def tile_border_lines() -> list[LineString]:
    c = [(0, 0), (TILE, 0), (TILE, TILE), (0, TILE), (0, 0)]
    return [LineString([a, b]) for a, b in zip(c[:-1], c[1:])]


def _on_border(a, b, eps=1e-6) -> bool:
    for k in range(2):
        for v in (0.0, float(TILE)):
            if abs(a[k] - v) < eps and abs(b[k] - v) < eps:
                return True
    return False


def building_lines(polys: list[Polygon]) -> list[np.ndarray]:
    """Building rings as polylines, without the parts that lie on the tile border."""
    lines = []
    for p in polys:
        c = np.asarray(p.exterior.coords)
        if not any(_on_border(a, b) for a, b in zip(c[:-1], c[1:])):
            lines.append(c)
            continue
        run = [c[0]]
        for a, b in zip(c[:-1], c[1:]):
            if _on_border(a, b):
                if len(run) > 1:
                    lines.append(np.array(run))
                run = [b]
            else:
                run.append(b)
        if len(run) > 1:
            lines.append(np.array(run))
    return lines


@dataclass
class Targets:
    arrays: dict[str, np.ndarray]
    prob: tuple[str, ...]  # channels that get occlusion, blur, noise
    heat: dict[str, np.ndarray] = field(default_factory=dict)  # channel -> GT peak points
    vec: tuple[str, ...] = ()
    info: dict = field(default_factory=dict)


def encode(t: Tile, rep: str) -> Targets:
    if rep in ("R0", "R1"):
        arr = {"mask": road_mask(t)}
        if rep == "R0":
            return Targets(arr, ("mask",))
        kp = key_points(t.graph)
        arr["junction"] = gaussian_points(kp, 2.0)
        return Targets(arr, ("mask", "junction"), {"junction": kp})
    if rep == "R2":
        vg = vertex_graph(graph_lines(t.graph))
        arr = graph_tensor(vg)
        return Targets(arr, ("vertex", "edge"), {"vertex": vg.xy}, ("vec",),
                       {"unrepresented": vg.unrepresented, "ends": vg.total_ends})
    inst = instance_raster(t.polygons)
    if rep in ("B0", "B1"):
        tg = instance_targets(torch.from_numpy(inst.astype(np.int64)))
        arr = {k: v[0, 0].numpy().astype(np.float32) for k, v in tg.items()}
        if rep == "B0":
            return Targets(arr, ("extent", "boundary", "distance"))
        corners = np.concatenate([gt_polygon_vertices(p) for p in t.polygons])
        arr["corner"] = gaussian_points(corners, 2.0)
        return Targets(arr, ("extent", "boundary", "distance", "corner"), {"corner": corners})
    if rep == "B2":
        vg = vertex_graph(building_lines(t.polygons), planar=True)
        arr = graph_tensor(vg)
        arr["mask"] = (inst > 0).astype(np.float32)
        return Targets(arr, ("vertex", "edge", "mask"), {"vertex": vg.xy}, ("vec",),
                       {"unrepresented": vg.unrepresented, "ends": vg.total_ends})
    raise ValueError(rep)


# --------------------------------------------------------------------------------------
# Corruptions
# --------------------------------------------------------------------------------------


def _rng(seed: int, t: Tile, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(t.area.encode()), t.index,
                                  zlib.crc32(t.task.encode()), zlib.crc32(key.encode())])


def gt_lines(t: Tile) -> list[np.ndarray]:
    if t.task == "roads":
        _, p, q = _segments(t.graph)
        return [np.stack([a, b]) for a, b in zip(p, q)]
    return [np.asarray(p.exterior.coords) for p in t.polygons]


def occlusion_disks(t: Tile, rate_per_100m: float, seed: int) -> np.ndarray:
    """(K, 3) disks (x, y, r) along the GT lines; the same for every representation."""
    if rate_per_100m <= 0:
        return np.zeros((0, 3))
    segs = [(a, b) for c in gt_lines(t) for a, b in zip(c[:-1], c[1:])]
    lens = np.array([np.hypot(*(b - a)) for a, b in segs])
    total_m = lens.sum() * t.gsd
    n = int(round(rate_per_100m * total_m / 100))
    rng = _rng(seed, t, "occlusion")
    if n == 0:
        return np.zeros((0, 3))
    cum = np.concatenate([[0], np.cumsum(lens)])
    s = rng.uniform(0, cum[-1], n)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(segs) - 1)
    f = (s - cum[k]) / np.maximum(lens[k], 1e-9)
    a = np.array([segs[i][0] for i in k])
    b = np.array([segs[i][1] for i in k])
    xy = a + (b - a) * f[:, None]
    r = rng.uniform(4, 12, n)
    return np.concatenate([xy, r[:, None]], 1)


def _disk_mask(disks: np.ndarray, shape=(TILE, TILE)) -> np.ndarray:
    m = np.zeros(shape, bool)
    for x, y, r in disks:
        r0, r1 = max(0, int(y - r) - 1), min(shape[0], int(y + r) + 2)
        c0, c1 = max(0, int(x - r) - 1), min(shape[1], int(x + r) + 2)
        if r0 >= r1 or c0 >= c1:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        m[r0:r1, c0:c1] |= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r
    return m


NOISE_SIGMA = 2.0  # px; spatial correlation of the additive noise (amendment 1)


def correlated_noise(rng: np.random.Generator, shape, std: float) -> np.ndarray:
    """White noise smoothed with a Gaussian (sigma ``NOISE_SIGMA`` px per image plane), then
    rescaled to ``std``: smooth errors like a network's, not salt-and-pepper."""
    z = rng.normal(0, 1, shape).astype(np.float32)
    sig = (0, NOISE_SIGMA, NOISE_SIGMA) if len(shape) == 3 else NOISE_SIGMA
    z = ndimage.gaussian_filter(z, sig, mode="wrap")
    return z * (std / max(float(z.std()), 1e-9))


def corrupt(tg: Targets, t: Tile, cfg: dict, seed: int) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in tg.arrays.items()}
    occ = _disk_mask(occlusion_disks(t, cfg["occ"], seed)) if cfg["occ"] > 0 else None
    for key in tg.prob:
        a = out[key]
        if occ is not None:
            a[..., occ] = 0
        if key in tg.heat and cfg["miss"] > 0:
            pts = tg.heat[key]
            rng = _rng(seed, t, "miss/" + key)
            drop = pts[rng.random(len(pts)) < cfg["miss"]]
            if len(drop):
                a[..., _disk_mask(np.concatenate([drop, np.full((len(drop), 1), 3.0)], 1))] = 0
        if cfg["blur"] > 0:
            sig = (0, cfg["blur"], cfg["blur"]) if a.ndim == 3 else cfg["blur"]
            a = ndimage.gaussian_filter(a, sig, mode="nearest")
        if cfg["noise"] > 0:
            a = a + correlated_noise(_rng(seed, t, "noise/" + key), a.shape, cfg["noise"])
        out[key] = np.clip(a, 0, 1).astype(np.float32)
    for key in tg.vec:
        if cfg["vec"] > 0:
            out[key] = out[key] + _rng(seed, t, "vec/" + key).normal(
                0, cfg["vec"], out[key].shape).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# Decoders: roads
# --------------------------------------------------------------------------------------

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_chains(sk: np.ndarray):
    """Skeleton -> (nodes {id: xy}, chains [(u, v, coords)]), junction clusters collapsed."""
    rr, cc = np.nonzero(sk)
    if len(rr) == 0:
        return {}, []
    idx = -np.ones((sk.shape[0] + 2, sk.shape[1] + 2), np.int64)
    idx[rr + 1, cc + 1] = np.arange(len(rr))
    pg = nx.Graph()
    pg.add_nodes_from(range(len(rr)))
    for dr, dc in _OFFS:
        nbr = idx[rr + 1 + dr, cc + 1 + dc]
        ok = nbr > np.arange(len(rr))
        if dr != 0 and dc != 0:  # drop a diagonal if a 4-neighbour path joins the same pixels
            ok &= (idx[rr + 1 + dr, cc + 1] < 0) & (idx[rr + 1, cc + 1 + dc] < 0)
        for a, b in zip(np.nonzero(ok)[0], nbr[ok]):
            pg.add_edge(int(a), int(b))
    xy = np.stack([cc + 0.5, rr + 0.5], 1)
    junc = [n for n in pg if pg.degree(n) >= 3]
    cluster = {}
    sub = pg.subgraph(junc)
    nodes = {}
    for ci, comp in enumerate(nx.connected_components(sub)):
        comp = list(comp)
        for n in comp:
            cluster[n] = ("j", ci)
        nodes[("j", ci)] = xy[comp].mean(0)
    for n in pg:
        if pg.degree(n) == 1:
            cluster[n] = ("e", n)
            nodes[("e", n)] = xy[n]
    chains = []
    seen = set()
    for start in list(cluster):
        for nb in pg.neighbors(start):
            if nb in cluster and cluster[nb] == cluster[start]:
                continue
            if frozenset((start, nb)) in seen:
                continue
            seen.add(frozenset((start, nb)))
            path = [start, nb]
            while path[-1] not in cluster:
                cur, prev = path[-1], path[-2]
                nxt = [n for n in pg.neighbors(cur) if n != prev and frozenset((cur, n)) not in seen]
                if not nxt:
                    break
                seen.add(frozenset((cur, nxt[0])))
                path.append(nxt[0])
            if path[-1] not in cluster:
                continue
            u, v = cluster[path[0]], cluster[path[-1]]
            coords = np.concatenate([[nodes[u]], xy[path[1:-1]], [nodes[v]]])
            chains.append((u, v, coords))
    # pure cycles (no junction or end): one node on the cycle
    rest = [n for n in pg if n not in cluster and not any(frozenset((n, m)) in seen for m in pg.neighbors(n))]
    restg = pg.subgraph(rest)
    for comp in nx.connected_components(restg):
        comp = list(comp)
        if len(comp) < 3:
            continue
        cyc = [comp[0]]
        prev = None
        while True:
            nxt = [n for n in restg.neighbors(cyc[-1]) if n != prev and n != cyc[-1]]
            if not nxt or nxt[0] == cyc[0]:
                break
            prev = cyc[-1]
            cyc.append(nxt[0])
            if len(cyc) > len(comp):
                break
        nid = ("o", comp[0])
        nodes[nid] = xy[cyc[0]]
        chains.append((nid, nid, np.concatenate([xy[cyc], [xy[cyc[0]]]])))
    return nodes, chains


def _chain_len(c) -> float:
    return float(np.hypot(*np.diff(c, axis=0).T).sum())


def _degrees(chains) -> dict:
    deg: dict = {}
    for u, v, _ in chains:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def contract_short(nodes, chains, min_len=3.0):
    """Merge nodes joined by a chain shorter than ``min_len`` (junction artefacts)."""
    changed = True
    while changed:
        changed = False
        for i, (u, v, c) in enumerate(chains):
            if u != v and _chain_len(c) < min_len:
                nodes[u] = (np.asarray(nodes[u]) + np.asarray(nodes[v])) / 2
                new = []
                for j, (a, b, cc) in enumerate(chains):
                    if j == i:
                        continue
                    a2, b2 = (u if a == v else a), (u if b == v else b)
                    cc = cc.copy()
                    if a2 != a or a == u:
                        cc[0] = nodes[u]
                    if b2 != b or b == u:
                        cc[-1] = nodes[u]
                    new.append((a2, b2, cc))
                del nodes[v]
                chains = new
                changed = True
                break
    return nodes, chains


def join_degree2(nodes, chains):
    """Join chains through nodes of degree 2."""
    while True:
        deg = _degrees(chains)
        n2 = next((n for n, d in deg.items() if d == 2 and
                   sum((a == n) + (b == n) for a, b, _ in chains if a == b == n) == 0), None)
        if n2 is None:
            return nodes, chains
        ids = [i for i, (a, b, _) in enumerate(chains) if a == n2 or b == n2]
        if len(ids) != 2:
            return nodes, chains
        (a1, b1, c1), (a2, b2, c2) = chains[ids[0]], chains[ids[1]]
        if b1 != n2:
            a1, b1, c1 = b1, a1, c1[::-1]
        if a2 != n2:
            a2, b2, c2 = b2, a2, c2[::-1]
        merged = (a1, b2, np.concatenate([c1, c2[1:]]))
        chains = [c for i, c in enumerate(chains) if i not in ids] + [merged]
        nodes.pop(n2, None)


def _near_border(p, margin=3.0) -> bool:
    return min(p[0], p[1], TILE - p[0], TILE - p[1]) <= margin


def prune_spurs(nodes, chains, spur_len, keep_end=None, passes=2):
    for _ in range(passes):
        deg = _degrees(chains)
        new = []
        for u, v, c in chains:
            ends = [n for n in (u, v) if deg.get(n, 0) == 1]
            if ends and u != v and _chain_len(c) < spur_len and deg.get(u, 0) + deg.get(v, 0) > 2:
                e = ends[0]
                if _near_border(nodes[e]) or (keep_end is not None and keep_end(nodes[e])):
                    new.append((u, v, c))
                continue
            new.append((u, v, c))
        chains = new
        nodes, chains = join_degree2(nodes, chains)
    return nodes, chains


def finish_graph(nodes, chains, simplify, min_len) -> nx.Graph:
    out = []
    for u, v, c in chains:
        if u == v:
            s = np.asarray(LineString(c).simplify(simplify).coords)
            if len(s) < 3:
                continue
        else:
            s = np.asarray(LineString(c).simplify(simplify).coords) if len(c) > 2 else c
        s = s.copy()
        s[0], s[-1] = nodes[u], nodes[v]
        out.append((u, v, s))
    g = polylines_to_graph({n: nodes[n] for n in nodes}, out)
    for comp in list(nx.connected_components(g)):
        if graph_length(g.subgraph(comp)) < min_len:
            g.remove_nodes_from(comp)
    return g


def bridge_ends(g: nx.Graph, gap: float) -> nx.Graph:
    """Join mutually-nearest dangling ends (degree 1, not at the tile border) within ``gap``
    px, unless they are already joined by a path shorter than 3 x ``gap`` (amendment 1)."""
    if gap <= 0:
        return g
    ends = [n for n in g if g.degree(n) == 1 and not _near_border(g.nodes[n]["xy"])]
    if len(ends) < 2:
        return g
    xy = np.array([g.nodes[n]["xy"] for n in ends])
    d, k = cKDTree(xy).query(xy, k=2, distance_upper_bound=gap)
    for i in range(len(ends)):
        j = k[i, 1]
        if not np.isfinite(d[i, 1]) or j <= i or k[j, 1] != i:
            continue
        a, b = ends[i], ends[j]
        try:
            close = nx.shortest_path_length(g, a, b) is not None and \
                nx.dijkstra_path_length(g, a, b, weight=lambda u, v, _: float(
                    np.hypot(*(g.nodes[u]["xy"] - g.nodes[v]["xy"])))) < 3 * gap
        except nx.NetworkXNoPath:
            close = False
        if not close:
            g.add_edge(a, b)
    return g


CACHE: dict = {}  # per-job memo of work shared across a parameter grid (run.py clears it)


def _skeleton(a: dict, thr: float):
    key = ("skeleton", id(a["mask"]), thr)
    if key not in CACHE:
        m = remove_small_objects(a["mask"] > thr, max_size=29)
        CACHE[key] = contract_short(*skeleton_chains(skeletonize(m)))
    nodes, chains = CACHE[key]
    return dict(nodes), list(chains)


def decode_r0(a: dict, p: dict) -> nx.Graph:
    nodes, chains = _skeleton(a, p["thr"])
    nodes, chains = join_degree2(nodes, chains)
    nodes, chains = prune_spurs(nodes, chains, p["spur"])
    return bridge_ends(finish_graph(nodes, chains, p["simplify"], p["min_len"]), p.get("gap", 0))


def _peak_pixels(hm: np.ndarray, thr: float, kernel: int) -> np.ndarray:
    """``find_peaks`` (y, x) pixels, then greedy suppression within the kernel radius: a
    plateau (e.g. a Gaussian centred between pixels) otherwise gives several tied peaks."""
    pts, scores = find_peaks(torch.from_numpy(hm)[None], thr, kernel=kernel)[0]
    pts = pts.numpy()
    if len(pts) < 2:
        return pts
    tree = cKDTree(pts)
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):  # sorted by score, descending
        if keep[i]:
            for j in tree.query_ball_point(pts[i], (kernel // 2) * 1.5):
                if j > i:
                    keep[j] = False
    return pts[keep]


def _refine(hm: np.ndarray, pix: np.ndarray) -> np.ndarray:
    """Sub-pixel peak positions ``(x, y)``: per-axis parabola through the log heatmap at the
    peak and its two neighbours (exact for a Gaussian), clipped to +-1 px."""
    if len(pix) == 0:
        return np.zeros((0, 2))
    H, W = hm.shape
    lg = np.log(np.clip(hm, 1e-4, None))
    r, c = pix[:, 0], pix[:, 1]

    def offset(lo, mid, hi):
        den = lo - 2 * mid + hi
        return np.clip(np.where(den < -1e-9, 0.5 * (lo - hi) / np.where(den < -1e-9, den, -1), 0), -1, 1)

    inx, iny = (c > 0) & (c < W - 1), (r > 0) & (r < H - 1)  # both neighbours inside
    dx = np.where(inx, offset(lg[r, np.maximum(c - 1, 0)], lg[r, c], lg[r, np.minimum(c + 1, W - 1)]), 0)
    dy = np.where(iny, offset(lg[np.maximum(r - 1, 0), c], lg[r, c], lg[np.minimum(r + 1, H - 1), c]), 0)
    return np.stack([c + 0.5 + dx, r + 0.5 + dy], 1)


def _peaks(hm: np.ndarray, thr: float, kernel: int) -> np.ndarray:
    return _refine(hm, _peak_pixels(hm, thr, kernel))


def decode_r1(a: dict, p: dict) -> nx.Graph:
    nodes, chains = _skeleton(a, p["thr"])
    peaks = _peaks(a["junction"], p["thr_j"], p["kernel"])
    if len(peaks) and nodes:
        tree = cKDTree(peaks)
        ids = [n for n in nodes]
        d, k = tree.query(np.array([nodes[n] for n in ids]), distance_upper_bound=p["merge_r"])
        target = {}
        for n, dd, kk in zip(ids, d, k):
            if np.isfinite(dd):
                target[n] = ("p", int(kk))
        for n, tnode in target.items():
            nodes[tnode] = peaks[tnode[1]]
        new = []
        for u, v, c in chains:
            u2, v2 = target.get(u, u), target.get(v, v)
            c = c.copy()
            c[0], c[-1] = nodes[u2], nodes[v2]
            if u2 == v2 and len(c) <= 3:
                continue
            new.append((u2, v2, c))
        chains = new
        for n in target:
            nodes.pop(n, None)
        used = {x for u, v, _ in chains for x in (u, v)}
        nodes = {n: xy for n, xy in nodes.items() if n in used}

    def keep_end(xy):
        return len(peaks) > 0 and np.min(np.hypot(*(peaks - xy).T)) <= p["merge_r"]

    nodes, chains = join_degree2(nodes, chains)
    nodes, chains = prune_spurs(nodes, chains, p["spur"], keep_end=keep_end)
    return bridge_ends(finish_graph(nodes, chains, p["simplify"], p["min_len"]), p.get("gap", 0))


def decode_graph_tensor(a: dict, p: dict) -> nx.Graph:
    pi = _peak_pixels(a["vertex"], p["thr_v"], p["kernel"])
    g = nx.Graph()
    if len(pi) == 0:
        return g
    centre = np.stack([pi[:, 1] + 0.5, pi[:, 0] + 0.5], 1)  # vectors are relative to these
    xy = _refine(a["vertex"], pi)
    if p.get("snap_border"):
        xy = np.where(xy < 1.5, 0.0, np.where(xy > TILE - 1.5, float(TILE), xy))
    for i, q in enumerate(xy):
        g.add_node(i, xy=q)
    tree = cKDTree(xy)
    prop = set()
    for i, (r, c) in enumerate(pi):
        for k in range(SECTORS):
            if a["edge"][k, r, c] <= p["thr_e"]:
                continue
            end = centre[i] + a["vec"][k, :, r, c]
            d, j = tree.query(end, k=2, distance_upper_bound=p["link_r"])
            for dd, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
                if np.isfinite(dd) and jj != i:
                    prop.add((i, int(jj)))
                    break
    for i, j in prop:
        if not p.get("mutual") or (j, i) in prop:
            g.add_edge(i, j)
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    return bridge_ends(g, p.get("gap", 0))


def decode_r2(a: dict, p: dict) -> nx.Graph:
    return decode_graph_tensor(a, p)


# --------------------------------------------------------------------------------------
# Decoders: footprints
# --------------------------------------------------------------------------------------


def _instance_contours(lab: np.ndarray):
    """Yield (label, dense contour (K, 2) xy) for each instance (its longest contour)."""
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        r0, c0 = sl[0].start, sl[1].start
        m = np.pad(lab[sl] == i, 1).astype(np.float32)
        cs = find_contours(m, 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        xy = np.stack([c[:, 1] - 1 + c0 + 0.5, c[:, 0] - 1 + r0 + 0.5], 1)
        if len(xy) >= 4:
            yield i, xy


def _valid_polygon(xy) -> Polygon | None:
    if len(xy) < 3:
        return None
    p = Polygon(xy)
    if not p.is_valid:
        p = p.buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
    return p if p.geom_type == "Polygon" and p.area > 1 else None


def _labels_b0(a: dict, p: dict) -> np.ndarray:
    key = ("labels", id(a["extent"]), p["thr"], p["thr_b"], p["thr_s"])
    if key not in CACHE:
        CACHE[key] = _labels_b0_uncached(a, p)
    return CACHE[key]


def _labels_b0_uncached(a: dict, p: dict) -> np.ndarray:
    t = lambda k: torch.from_numpy(a[k])[None, None]
    lab = instances_from_outputs(t("extent"), t("boundary"), t("distance"), logits=False,
                                 extent_threshold=p["thr"], boundary_threshold=p["thr_b"],
                                 seed_threshold=p["thr_s"], min_size=16)
    return lab[0].numpy().astype(np.int32)


def decode_b0(a: dict, p: dict) -> list[Polygon]:
    out = []
    for _, xy in _instance_contours(_labels_b0(a, p)):
        q = _valid_polygon(xy)
        if q is not None:
            q = q.simplify(p["simplify"])
            if q.geom_type == "Polygon" and q.area > 1:
                out.append(q)
    return out


def decode_b1(a: dict, p: dict) -> list[Polygon]:
    peaks = _peaks(a["corner"], p["thr_c"], p["kernel"])
    tree = cKDTree(peaks) if len(peaks) else None
    out = []
    for _, xy in _instance_contours(_labels_b0(a, p)):
        q = None
        if tree is not None:
            near = tree.query_ball_point(xy, p["snap_r"])
            cand = sorted({j for js in near for j in js})
            if len(cand) >= 3:
                ctree = cKDTree(xy)
                _, pos = ctree.query(peaks[cand])
                order = np.argsort(pos, kind="stable")
                q = _valid_polygon(peaks[np.array(cand)[order]])
        if q is None:
            q = _valid_polygon(xy)
            if q is not None:
                q = q.simplify(p["simplify"])
        if q is not None and q.geom_type == "Polygon" and q.area > 1:
            out.append(q)
    return out


def decode_b2(a: dict, p: dict) -> list[Polygon]:
    g = decode_graph_tensor(a, dict(p, snap_border=True))
    lines = [LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]) for u, v in g.edges()]
    lines += tile_border_lines()
    faces = list(polygonize(unary_union(lines)))
    if not faces:
        return []
    lab = rasterize([(f, i + 1) for i, f in enumerate(faces)], out_shape=(TILE, TILE),
                    transform=IDENT, dtype="int32")
    n = len(faces) + 1
    cnt = np.bincount(lab.ravel(), minlength=n)
    s = np.bincount(lab.ravel(), weights=(a["mask"] > 0.5).ravel(), minlength=n)
    out = []
    for i, f in enumerate(faces, start=1):
        if cnt[i] >= 16 and s[i] / cnt[i] > p["face_thr"]:
            q = f.simplify(p["simplify"])
            q = Polygon(q.exterior) if q.geom_type == "Polygon" else None
            if q is not None and q.is_valid and q.area > 1:
                out.append(q)
    return out


DECODERS = {"R0": decode_r0, "R1": decode_r1, "R2": decode_r2,
            "B0": decode_b0, "B1": decode_b1, "B2": decode_b2}


# --------------------------------------------------------------------------------------
# Metrics: road graphs
# --------------------------------------------------------------------------------------


def control_graph(g: nx.Graph, spacing: float):
    """Control points (key nodes + every ``spacing`` along chains) and weighted edges."""
    pts: list[np.ndarray] = []
    index: dict = {}
    edges = []

    def node(n):
        if n not in index:
            index[n] = len(pts)
            pts.append(np.asarray(g.nodes[n]["xy"], float))
        return index[n]

    for ch in graph_chains(g):
        c = np.array([g.nodes[n]["xy"] for n in ch], float)
        seg = np.hypot(*np.diff(c, axis=0).T)
        L = seg.sum()
        if L <= 0:
            continue
        cum = np.concatenate([[0], np.cumsum(seg)])
        prev, prev_s = node(ch[0]), 0.0
        step = min(spacing, L / 2) if ch[0] == ch[-1] else spacing  # a short loop: 2 points
        s = step
        while s < L - 1e-6:
            k = min(int(np.searchsorted(cum, s, side="right") - 1), len(seg) - 1)
            f = (s - cum[k]) / max(seg[k], 1e-12)
            pts.append(c[k] + (c[k + 1] - c[k]) * f)
            cur = len(pts) - 1
            edges.append((prev, cur, s - prev_s))
            prev, prev_s = cur, s
            s += step
        end = node(ch[-1])
        if end != prev or L - prev_s > 1e-9:
            edges.append((prev, end, L - prev_s))
    return np.array(pts).reshape(-1, 2), edges


def _csr(n: int, edges) -> csr_matrix:
    """Symmetric weight matrix; parallel edges keep their minimum (scipy would sum them)."""
    best: dict = {}
    for a, b, w in edges:
        if a == b:
            continue
        k = (min(a, b), max(a, b))
        best[k] = min(best.get(k, np.inf), float(w))
    if not best:
        return csr_matrix((n, n))
    i, j = map(np.asarray, zip(*best))
    w = np.fromiter(best.values(), float)
    w = np.maximum(w.astype(float), 1e-9)
    m = csr_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                   shape=(n, n))
    return m


def _apls_one(src: nx.Graph, dst: nx.Graph, spacing: float, buffer: float) -> float:
    cp, ce = control_graph(src, spacing)
    n = len(cp)
    if n < 2:
        return 0.0
    la = dijkstra(_csr(n, ce), directed=False)
    es, p, q = _segments(dst)
    if len(es) == 0:
        return 0.0
    lines = shapely.linestrings(np.stack([p, q], 1))
    tree = shapely.STRtree(lines)
    (ci, ei), dist = tree.query_nearest(shapely.points(cp), max_distance=buffer, return_distance=True,
                                        all_matches=False)
    # augmented destination graph: its nodes, then one node per snapped control point
    dnodes = {nd: k for k, nd in enumerate(dst.nodes())}
    nb = len(dnodes)
    snap = -np.ones(n, int)
    on_edge: dict[int, list] = {}
    for c, e in zip(ci, ei):
        if snap[c] >= 0:
            continue
        d = q[e] - p[e]
        t = float(np.clip(np.dot(cp[c] - p[e], d) / max(np.dot(d, d), 1e-12), 0, 1))
        snap[c] = nb
        on_edge.setdefault(int(e), []).append((t, nb))
        nb += 1
    bedges = []
    for k, (u, v) in enumerate(es):
        L = float(np.hypot(*(q[k] - p[k])))
        chain = [(0.0, dnodes[u])] + sorted(on_edge.get(k, [])) + [(1.0, dnodes[v])]
        for (t0, a), (t1, b) in zip(chain[:-1], chain[1:]):
            bedges.append((a, b, (t1 - t0) * L))
    snapped = np.nonzero(snap >= 0)[0]
    lb = np.full((n, n), np.inf)
    if len(snapped):
        dm = dijkstra(_csr(nb, bedges), directed=False, indices=snap[snapped])
        lb[np.ix_(snapped, snapped)] = dm[:, snap[snapped]]
    iu = np.triu_indices(n, 1)
    A, B = la[iu], lb[iu]
    ok = np.isfinite(A) & (A > 0)
    if not ok.any():
        return 0.0
    A, B = A[ok], B[ok]
    pen = np.where(np.isfinite(B), np.minimum(1.0, np.abs(A - B) / A), 1.0)
    return float(1 - pen.mean())


def apls(gt: nx.Graph, pred: nx.Graph, gsd: float, spacing_m=50.0, buffer_m=5.0) -> float:
    if pred.number_of_edges() == 0:
        return 0.0
    a = _apls_one(gt, pred, spacing_m / gsd, buffer_m / gsd)
    b = _apls_one(pred, gt, spacing_m / gsd, buffer_m / gsd)
    return 0.0 if a <= 0 or b <= 0 else 2 * a * b / (a + b)


def _samples(g: nx.Graph, step: float) -> np.ndarray:
    """Points every ``step`` px of arc length along each chain (not per edge, so the number of
    samples does not depend on how finely a road is split into edges)."""
    out = []
    for ch in graph_chains(g):
        c = np.array([g.nodes[n]["xy"] for n in ch], float)
        seg = np.hypot(*np.diff(c, axis=0).T)
        L = seg.sum()
        if L <= 0:
            continue
        s = np.arange(step / 2, L, step)
        cum = np.concatenate([[0], np.cumsum(seg)])
        k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
        f = (s - cum[k]) / np.maximum(seg[k], 1e-12)
        out.append(c[k] + (c[k + 1] - c[k]) * f[:, None])
    return np.concatenate(out) if out else np.zeros((0, 2))


def _dist_to_graph(pts: np.ndarray, g: nx.Graph) -> np.ndarray:
    es, p, q = _segments(g)
    if len(es) == 0 or len(pts) == 0:
        return np.full(len(pts), np.inf)
    tree = shapely.STRtree(shapely.linestrings(np.stack([p, q], 1)))
    (ii, jj), d = tree.query_nearest(shapely.points(pts), max_distance=1e9, return_distance=True,
                                      all_matches=False)
    out = np.full(len(pts), np.inf)
    out[ii] = d
    return out


def geo_f1(gt, pred, gsd, buffer_m=5.0) -> dict:
    sg, sp = _samples(gt, 1.0 / gsd), _samples(pred, 1.0 / gsd)
    if len(sp) == 0 or len(sg) == 0:
        return {"geo_p": 0.0, "geo_r": 0.0, "geo_f1": 0.0}
    prec = float((_dist_to_graph(sp, gt) <= buffer_m / gsd).mean())
    rec = float((_dist_to_graph(sg, pred) <= buffer_m / gsd).mean())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"geo_p": prec, "geo_r": rec, "geo_f1": f1}


def junctions(g: nx.Graph) -> np.ndarray:
    return np.array([g.nodes[n]["xy"] for n in g if g.degree(n) >= 3]).reshape(-1, 2)


def match_points(a: np.ndarray, b: np.ndarray, radius: float) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    cost = np.where(d <= radius, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return int((cost[r, c] <= radius).sum())


def junction_f1(gt, pred, gsd, radius_m=5.0) -> float:
    a, b = junctions(gt), junctions(pred)
    if len(a) + len(b) == 0:
        return 1.0
    return 2 * match_points(a, b, radius_m / gsd) / (len(a) + len(b))


def crossing_score(t: Tile, pred: nx.Graph) -> tuple[int, int, int]:
    """(no_junction, covered_ok, covered) over GT crossings. ``no_junction``: no predicted junction
    within 8 m (the registered metric; a missing road also counts). ``covered``: predicted road
    within 5 m; ``covered_ok``: covered and no junction within 8 m."""
    if not t.crossings:
        return 0, 0, 0
    cr = np.array(t.crossings)
    cov = _dist_to_graph(cr, pred) <= 5 / t.gsd
    jn = junctions(pred)
    if len(jn):
        dj = cKDTree(jn).query(cr)[0]
    else:
        dj = np.full(len(cr), np.inf)
    nj = dj > 8 / t.gsd
    return int(nj.sum()), int((cov & nj).sum()), int(cov.sum())


def road_metrics(t: Tile, pred: nx.Graph) -> dict:
    m = {"apls": apls(t.graph, pred, t.gsd), "junction_f1": junction_f1(t.graph, pred, t.gsd)}
    m.update(geo_f1(t.graph, pred, t.gsd))
    m["cross_nojunc"], m["cross_ok"], m["cross_cov"] = crossing_score(t, pred)
    m["cross_n"] = len(t.crossings)
    return m


# --------------------------------------------------------------------------------------
# Metrics: polygons
# --------------------------------------------------------------------------------------


def n_vertices(p: Polygon) -> int:
    return len(p.exterior.coords) - 1


def polis(a: Polygon, b: Polygon) -> float:
    va, vb = np.asarray(a.exterior.coords)[:-1], np.asarray(b.exterior.coords)[:-1]
    da = shapely.distance(shapely.points(va), b.exterior).mean()
    db = shapely.distance(shapely.points(vb), a.exterior).mean()
    return float(da / 2 + db / 2)


def polygon_metrics(t: Tile, pred: list[Polygon]) -> dict:
    gt = t.polygons
    gt_n = [len(gt_polygon_vertices(g)) for g in gt]
    pairs = []
    if pred:
        tree = shapely.STRtree(pred)
        for i, g in enumerate(gt):
            for j in tree.query(g, predicate="intersects"):
                inter = g.intersection(pred[j]).area
                if inter > 0:
                    iou = inter / (g.area + pred[j].area - inter)
                    if iou > 0.5:
                        pairs.append((iou, i, int(j)))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            matched.append((iou, i, j))
    tp = len(matched)
    out = {"det_f1": 2 * tp / (len(gt) + len(pred)) if gt or pred else 1.0,
           "n_gt": len(gt), "n_pred": len(pred), "tp": tp}
    if matched:
        ciou, ious, npred, ngt, pl = [], [], 0, 0, []
        for iou, i, j in matched:
            np_, ng = n_vertices(pred[j]), gt_n[i]
            ciou.append(iou * (1 - abs(np_ - ng) / (np_ + ng)))
            ious.append(iou)
            npred += np_
            ngt += ng
            pl.append(polis(gt[i], pred[j]) * t.gsd)
        out.update(ciou=float(np.mean(ciou)), miou=float(np.mean(ious)), n_ratio=npred / ngt,
                   polis=float(np.mean(pl)))
    else:
        out.update(ciou=0.0, miou=0.0, n_ratio=float("nan"), polis=float("nan"))
    return out


def evaluate(t: Tile, rep: str, pred) -> dict:
    return road_metrics(t, pred) if t.task == "roads" else polygon_metrics(t, pred)
