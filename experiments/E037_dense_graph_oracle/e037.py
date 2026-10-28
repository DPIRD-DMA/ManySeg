"""E037 oracle: the dense pixel graph (P8: node confidence, 8-neighbour links, sub-pixel offset per
pixel), its displaced-vertex variant (P8r) and mask-supported local links for roads (R3).

Coordinates follow E028: ``(x, y)`` pixels, pixel ``(row r, col c)`` covers ``[c, c+1) x [r, r+1)``,
its centre is ``(c + 0.5, r + 0.5)``. Links are indexed N, NE, E, SE, S, SW, W, NW.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "E028_vector_oracle"), str(HERE.parent / "E031_lod2_oracle")]
import e028 as E  # noqa: E402

DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # (dr, dc)
OPP = [4, 5, 6, 7, 0, 1, 2, 3]
ZONE_R = 10.0  # px, R3 link zones around keypoints


# --------------------------------------------------------------------------------------
# GT graphs
# --------------------------------------------------------------------------------------


def lines_to_graph(lines, merge: float = 1e-3) -> nx.Graph:
    """Polylines (K, 2) -> graph with ``xy`` nodes; coincident vertices merged; duplicate segments
    (shared walls / roof edges) kept once."""
    g = nx.Graph()
    ids: dict = {}

    def nid(p):
        k = (round(p[0] / merge), round(p[1] / merge))
        if k not in ids:
            ids[k] = len(ids)
            g.add_node(ids[k], xy=np.asarray(p, float))
        return ids[k]

    for ln in lines:
        ln = np.asarray(ln, float)
        for a, b in zip(ln[:-1], ln[1:]):
            u, v = nid(a), nid(b)
            if u != v:
                g.add_edge(u, v)
    return g


def chain_ids(g: nx.Graph) -> dict:
    """Edge (sorted pair) -> id of its degree-2 chain (the topological edge it belongs to)."""
    out, k = {}, 0
    for u, v in g.edges():
        e = (min(u, v), max(u, v))
        if e in out:
            continue
        stack = [e]
        while stack:
            a, b = stack.pop()
            if (a, b) in out:
                continue
            out[(a, b)] = k
            for n in (a, b):
                if g.degree(n) == 2:
                    for m in g.neighbors(n):
                        f = (min(n, m), max(n, m))
                        if f not in out:
                            stack.append(f)
        k += 1
    return out


# --------------------------------------------------------------------------------------
# P8 encoding
# --------------------------------------------------------------------------------------


def _cell(xy, H, W):
    return (min(max(int(math.floor(xy[1])), 0), H - 1), min(max(int(math.floor(xy[0])), 0), W - 1))


def dda(c0, c1) -> list:
    """8-connected path of cells (r, c) from c0 to c1, inclusive."""
    dr, dc = c1[0] - c0[0], c1[1] - c0[1]
    n = max(abs(dr), abs(dc))
    if n == 0:
        return [c0]
    return [(int(math.floor(c0[0] + dr * k / n + 0.5)), int(math.floor(c0[1] + dc * k / n + 0.5))) for k in range(n + 1)]


@dataclass
class P8:
    node: np.ndarray
    link: np.ndarray
    offset: np.ndarray
    collisions: int = 0  # cells where topology changed
    collision_cells: list = field(default_factory=list)
    moved: int = 0  # P8r: vertices placed on a neighbouring pixel


def _closest_offset(cell, a, b):
    cx, cy = cell[1] + 0.5, cell[0] + 0.5
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 == 0 else min(max(((cx - a[0]) * ab[0] + (cy - a[1]) * ab[1]) / L2, 0.0), 1.0)
    q = a + s * ab
    return q[0] - cx, q[1] - cy


def encode_p8(g: nx.Graph, shape, displace: bool = False) -> P8:
    """Rasterise each GT segment as the DDA path between its end vertices' cells. ``displace``
    (P8r): each vertex, highest degree first, may move to one of its 9 neighbouring cells to
    minimise collisions on its incident segments; offsets still point at the true position."""
    H, W = shape
    xy = {n: np.asarray(g.nodes[n]["xy"], float) for n in g}
    cell = {n: _cell(xy[n], H, W) for n in g}
    cid = chain_ids(g)
    segs = [(min(u, v), max(u, v)) for u, v in g.edges()]

    def path(e):
        return dda(cell[e[0]], cell[e[1]])

    occ: dict = {}  # cell -> {seg: count}

    def add(e, sign):
        for c in path(e):
            d = occ.setdefault(c, {})
            d[e] = d.get(e, 0) + sign
            if d[e] <= 0:
                del d[e]

    for e in segs:
        add(e, 1)
    moved = 0
    if displace:
        def bad(e):  # cells of e shared with a segment of another chain not meeting e there
            n = 0
            for c in path(e):
                for f in occ.get(c, {}):
                    if f != e and cid[f] != cid[e] and not ({*e} & {*f} and any(cell[v] == c for v in {*e} & {*f})):
                        n += 1
            return n

        for v in sorted(g, key=lambda n: -g.degree(n)):
            inc = [(min(v, m), max(v, m)) for m in g.neighbors(v)]
            if not inc:
                continue
            home = cell[v]
            best, best_c = None, home
            for dr in (0, -1, 1):
                for dc in (0, -1, 1):
                    c = (home[0] + dr, home[1] + dc)
                    if not (0 <= c[0] < H and 0 <= c[1] < W) or (dr or dc) and math.hypot(
                            xy[v][0] - (c[1] + 0.5), xy[v][1] - (c[0] + 0.5)) > 1.5:
                        continue
                    for e in inc:
                        add(e, -1)
                    cell[v] = c
                    for e in inc:
                        add(e, 1)
                    score = sum(bad(e) for e in inc) + 1e-3 * (dr != 0 or dc != 0)
                    if best is None or score < best:
                        best, best_c = score, c
                    for e in inc:
                        add(e, -1)
                    cell[v] = home
                    for e in inc:
                        add(e, 1)
            if best_c != home:
                for e in inc:
                    add(e, -1)
                cell[v] = best_c
                for e in inc:
                    add(e, 1)
                moved += 1

    node = np.zeros((H, W), np.float32)
    link = np.zeros((8, H, W), np.float32)
    offset = np.zeros((2, H, W), np.float32)
    dir_of = {d: k for k, d in enumerate(DIRS)}
    vcell: dict = {}
    for v in g:
        vcell.setdefault(cell[v], []).append(v)
    for e in segs:
        p = path(e)
        a, b = xy[e[0]], xy[e[1]]
        for c in p:
            node[c] = 1
            if c not in vcell:
                offset[0][c], offset[1][c] = _closest_offset(c, a, b)
        for c0, c1 in zip(p[:-1], p[1:]):
            k = dir_of[(c1[0] - c0[0], c1[1] - c0[1])]
            link[k][c0] = 1
            link[OPP[k]][c1] = 1
    for c, vs in vcell.items():
        v = max(vs, key=g.degree)
        node[c] = 1
        offset[0][c], offset[1][c] = xy[v][0] - (c[1] + 0.5), xy[v][1] - (c[0] + 0.5)
    # collisions: a cell holding segments of >= 2 chains that do not all meet at one vertex in that cell,
    # or holding vertices of >= 2 chains-ends that are not adjacent (merged distinct junctions)
    coll = []
    for c, d in occ.items():
        chains = {cid[e] for e in d}
        vs = vcell.get(c, [])
        if len(chains) > 1:
            shared = [v for v in vs if all(v in e for e in d)]
            if not shared:
                coll.append(c)
                continue
        if len(vs) > 1:
            sub = g.subgraph(vs)
            if nx.number_connected_components(sub) > 1:
                coll.append(c)
    return P8(node, link, offset, len(coll), coll, moved)


# --------------------------------------------------------------------------------------
# P8 decoding
# --------------------------------------------------------------------------------------


def pixel_graph(node, link, offset, *, thr: float = 0.5, rule: str = "and", region=None) -> nx.Graph:
    """Pixel nodes (confidence > thr, optionally inside ``region``) and their links; an edge needs
    both reciprocal links ("and") or either ("or"). Node attribute ``xy`` = centre + offset."""
    H, W = node.shape
    on = node > thr
    if region is not None:
        on &= region
    rr, cc = np.nonzero(on)
    g = nx.Graph()
    for r, c in zip(rr, cc):
        g.add_node((r, c), xy=np.array([c + 0.5 + offset[0, r, c], r + 0.5 + offset[1, r, c]]))
    for k, (dr, dc) in enumerate(DIRS[:4]):  # each undirected pair once
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        r, c, r2, c2 = rr[ok], cc[ok], r2[ok], c2[ok]
        fwd = link[k, r, c] > thr
        bwd = link[OPP[k], r2, c2] > thr
        lk = (fwd & bwd) if rule == "and" else (fwd | bwd)
        lk &= on[r2, c2]
        for a, b, x, y in zip(r[lk], c[lk], r2[lk], c2[lk]):
            g.add_edge((a, b), (x, y))
    return g


def graph_to_chains(pg: nx.Graph):
    """Pixel graph -> (nodes {id: xy}, chains [(u, v, coords)]) between nodes of degree != 2;
    pure cycles get one node."""
    nodes = {n: pg.nodes[n]["xy"] for n in pg if pg.degree(n) != 2}
    seen = set()
    chains = []
    for n in list(nodes):
        for m in pg.neighbors(n):
            if (n, m) in seen:
                continue
            path = [n, m]
            seen.add((n, m))
            seen.add((m, n))
            while path[-1] not in nodes:
                nxt = [x for x in pg.neighbors(path[-1]) if x != path[-2]]
                if not nxt:
                    break
                seen.add((path[-1], nxt[0]))
                seen.add((nxt[0], path[-1]))
                path.append(nxt[0])
            if path[-1] in nodes:
                chains.append((n, path[-1], np.array([pg.nodes[x]["xy"] for x in path])))
    rest = pg.subgraph([n for n in pg if pg.degree(n) == 2 and not any((n, m) in seen for m in pg.neighbors(n))])
    for comp in nx.connected_components(rest):
        sub = pg.subgraph(comp)
        try:
            cyc = [u for u, _ in nx.find_cycle(sub)]
        except nx.NetworkXNoCycle:
            continue
        n0 = cyc[0]
        nodes[n0] = pg.nodes[n0]["xy"]
        chains.append((n0, n0, np.array([pg.nodes[x]["xy"] for x in cyc + [n0]])))
    return nodes, chains


def finish(pg: nx.Graph, simplify: float, min_len: float = 0.0) -> nx.Graph:
    nodes, chains = graph_to_chains(pg)
    if not chains:
        return nx.Graph()
    return E.finish_graph(nodes, chains, simplify, min_len)


def decode_p8_roads(a: dict, p: dict) -> nx.Graph:
    pg = pixel_graph(a["node"], a["link"], a["offset"], thr=p.get("thr", 0.5), rule=p.get("rule", "and"))
    return E.bridge_ends(finish(pg, p.get("simplify", 1.0), p.get("min_len", 0.0)), p.get("gap", 0))


def faces_from_graph(g: nx.Graph, mask: np.ndarray, simplify: float, face_thr: float = 0.5) -> list:
    lines = [LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]) for u, v in g.edges()]
    H, W = mask.shape
    c = [(0, 0), (W, 0), (W, H), (0, H), (0, 0)]
    lines += [LineString([a, b]) for a, b in zip(c[:-1], c[1:])]
    faces = list(polygonize(unary_union(lines)))
    if not faces:
        return []
    from rasterio.features import rasterize
    lab = rasterize([(f, i + 1) for i, f in enumerate(faces)], out_shape=mask.shape,
                    transform=E.IDENT, dtype="int32")
    n = len(faces) + 1
    cnt = np.bincount(lab.ravel(), minlength=n)
    s = np.bincount(lab.ravel(), weights=(mask > 0.5).ravel(), minlength=n)
    out = []
    for i, f in enumerate(faces, start=1):
        if cnt[i] >= 16 and s[i] / cnt[i] > face_thr:
            q = f.simplify(simplify)
            q = Polygon(q.exterior) if q.geom_type == "Polygon" else None
            if q is not None and q.is_valid and q.area > 1:
                out.append(q)
    return out


def decode_p8_faces(a: dict, p: dict) -> list:
    pg = pixel_graph(a["node"], a["link"], a["offset"], thr=p.get("thr", 0.5), rule=p.get("rule", "and"))
    g = E.bridge_ends(finish(pg, 0.0), p.get("gap", 0))
    return faces_from_graph(g, a["mask"], p.get("simplify", 0.5))


# --------------------------------------------------------------------------------------
# R3: mask skeleton outside keypoint zones, P8 links inside
# --------------------------------------------------------------------------------------


def road_keypoints(t) -> np.ndarray:
    g = t.graph
    kp = [g.nodes[n]["xy"] for n in g if g.degree(n) != 2]
    kp += [np.asarray(c, float) for c in t.crossings]
    return np.array(kp).reshape(-1, 2)


def zone_mask(kp: np.ndarray, shape, r: float = ZONE_R) -> np.ndarray:
    H, W = shape
    z = np.zeros(shape, bool)
    for x, y in kp:
        r0, r1 = max(int(y - r), 0), min(int(y + r) + 2, H)
        c0, c1 = max(int(x - r), 0), min(int(x + r) + 2, W)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        z[r0:r1, c0:c1] |= (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r
    return z


def merge_zone(t, shape, grow: float = 12.0) -> np.ndarray:
    """Pixels where the road masks of two different chains (degree-2 runs) overlap or come
    within 2 px of each other, dilated by ``grow`` px: where the mask skeleton cannot separate
    roads (dual carriageways, slip lanes)."""
    from rasterio.features import rasterize
    from scipy.ndimage import binary_dilation
    g = nx.convert_node_labels_to_integers(t.graph)
    cid = chain_ids(g)
    by_chain: dict = {}
    for u, v, d in g.edges(data=True):
        by_chain.setdefault(cid[(min(u, v), max(u, v))], []).append(
            LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]).buffer(d["width"] / 2 + 1.0))
    count = np.zeros(shape, np.int32)
    for polys in by_chain.values():
        count += rasterize([(unary_union(polys), 1)], out_shape=shape, transform=E.IDENT, dtype="uint8")
    m = count >= 2
    r = int(math.ceil(grow))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return binary_dilation(m, structure=xx ** 2 + yy ** 2 <= grow * grow)


def skeleton_pixel_graph(sk: np.ndarray) -> nx.Graph:
    """8-connected skeleton pixels, a diagonal dropped where a 4-neighbour path joins the pair."""
    H, W = sk.shape
    rr, cc = np.nonzero(sk)
    g = nx.Graph()
    for r, c in zip(rr, cc):
        g.add_node((r, c), xy=np.array([c + 0.5, r + 0.5]))
    for r, c in zip(rr, cc):
        for dr, dc in DIRS[:4]:
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < H and 0 <= c2 < W) or not sk[r2, c2]:
                continue
            if dr and dc and (sk[r + dr, c] or sk[r, c + dc]):
                continue
            g.add_edge((r, c), (r2, c2))
    return g


def decode_r3(a: dict, p: dict) -> nx.Graph:
    """R0's decoded graph clipped outside the zones + the P8 graph decoded inside them, joined in
    vector space: each line end on a zone boundary pairs with the nearest end of the other kind
    (one-to-one, nearest first, within ``stitch`` px)."""
    zone_poly = a["zone_poly"]
    r0 = E.decode_r0(a, dict(p, gap=0))
    outside = []
    for u, v in r0.edges():
        ln = LineString([r0.nodes[u]["xy"], r0.nodes[v]["xy"]]).difference(zone_poly)
        outside += [g for g in getattr(ln, "geoms", [ln]) if not g.is_empty and g.length > 0]
    lg = pixel_graph(a["node"], a["link"], a["offset"], thr=0.5, rule=p.get("rule", "and"), region=a["zone"])
    inside_g = finish(lg, p.get("simplify", 1.0))
    inside = [LineString([inside_g.nodes[u]["xy"], inside_g.nodes[v]["xy"]]) for u, v in inside_g.edges()]
    lines = [np.asarray(l.coords) for l in outside + inside]
    g = lines_to_graph(lines, merge=1e-3)
    bnd = zone_poly.boundary

    def ends(kind_lines, tol):
        pts = []
        for l in kind_lines:
            for q in (l.coords[0], l.coords[-1]):
                if bnd.distance(shapely.Point(q)) <= tol:
                    pts.append(q)
        return pts

    eo, ei = ends(outside, 0.5), ends(inside, 2.0)
    deg = dict(g.degree())
    key = {(round(g.nodes[n]["xy"][0] / 1e-3), round(g.nodes[n]["xy"][1] / 1e-3)): n for n in g}
    eo = [key[(round(q[0] / 1e-3), round(q[1] / 1e-3))] for q in eo]
    ei = [key[(round(q[0] / 1e-3), round(q[1] / 1e-3))] for q in ei]
    eo = [n for n in dict.fromkeys(eo) if deg[n] == 1]
    ei = [n for n in dict.fromkeys(ei) if deg[n] == 1]
    if eo and ei:
        xo = np.array([g.nodes[n]["xy"] for n in eo])
        xi = np.array([g.nodes[n]["xy"] for n in ei])
        d = np.hypot(*(xo[:, None] - xi[None]).transpose(2, 0, 1))
        uo, ui = set(), set()
        for flat in np.argsort(d, axis=None):
            k, m = divmod(int(flat), d.shape[1])
            if d[k, m] > p.get("stitch", 4.0):
                break
            if k not in uo and m not in ui:
                uo.add(k)
                ui.add(m)
                g.add_edge(eo[k], ei[m])
    for comp in list(nx.connected_components(g)):
        if E.graph_length(g.subgraph(comp)) < p.get("min_len", 20):
            g.remove_nodes_from(comp)
    return E.bridge_ends(g, p.get("gap", 0))


# --------------------------------------------------------------------------------------
# Targets per task
# --------------------------------------------------------------------------------------


def road_graph_px(t) -> nx.Graph:
    g = nx.Graph()
    for n in t.graph:
        g.add_node(n, xy=np.asarray(t.graph.nodes[n]["xy"], float))
    g.add_edges_from(t.graph.edges())
    return nx.convert_node_labels_to_integers(g)  # E028 border nodes are tuples


def encode_roads(t, rep: str) -> dict:
    shape = (E.TILE, E.TILE)
    if rep in ("P8", "P8r"):
        enc = encode_p8(road_graph_px(t), shape, displace=rep == "P8r")
        return {"node": enc.node, "link": enc.link, "offset": enc.offset, "_enc": enc}
    if rep == "R3":
        enc = encode_p8(road_graph_px(t), shape, displace=True)
        kp = road_keypoints(t)
        zone = zone_mask(kp, shape) | merge_zone(t, shape)
        from rasterio.features import shapes as _shapes
        zpoly = unary_union([shapely.geometry.shape(gm) for gm, v in _shapes(zone.astype(np.uint8), mask=zone, transform=E.IDENT)])
        return {"mask": E.road_mask(t), "node": enc.node, "link": enc.link, "offset": enc.offset,
                "zone": zone, "zone_poly": zpoly, "_enc": enc}
    return E.encode(t, rep).arrays


def encode_buildings(t, rep: str) -> dict:
    shape = (E.TILE, E.TILE)
    if rep in ("P8", "P8r"):
        enc = encode_p8(lines_to_graph(E.building_lines(t.polygons)), shape, displace=rep == "P8r")
        return {"node": enc.node, "link": enc.link, "offset": enc.offset,
                "mask": (E.instance_raster(t.polygons) > 0).astype(np.float32), "_enc": enc}
    raise ValueError(rep)


def roof_lines(faces, tile: int) -> list:
    """Roof-face edges (each shared edge once via lines_to_graph), without tile-border parts."""
    border = shapely.box(0, 0, tile, tile).exterior
    out = []
    for f in faces:
        for poly in getattr(f, "geoms", [f]):
            for ring in [poly.exterior, *poly.interiors]:
                c = np.asarray(ring.coords)
                for a, b in zip(c[:-1], c[1:]):
                    if LineString([a, b]).distance(border) < 1e-6 and LineString([a, b]).within(border.buffer(1e-6)):
                        continue
                    out.append(np.array([a, b]))
    return out
