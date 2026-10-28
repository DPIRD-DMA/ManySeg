"""E039 R4: the unified head set on roads. Road mask + junction / end / crossing keypoint heatmaps +
a two-mode (cos 2t, sin 2t) direction field; decoded as R0's skeleton graph outside keypoint zones
and arm pairing inside them (README)."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from rasterio.features import rasterize
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "E028_vector_oracle"), str(HERE.parent / "E037_dense_graph_oracle")]
import e028 as E  # noqa: E402

CLASSES = ("junction", "end", "crossing")


def _chains(g: nx.Graph) -> dict:
    """Edge -> id of its degree-2 chain."""
    out, k = {}, 0
    for u, v in g.edges():
        if frozenset((u, v)) in out:
            continue
        stack = [(u, v)]
        while stack:
            a, b = stack.pop()
            e = frozenset((a, b))
            if e in out:
                continue
            out[e] = k
            for n in (a, b):
                if g.degree(n) == 2:
                    stack += [(n, m) for m in g.neighbors(n) if frozenset((n, m)) not in out]
        k += 1
    return out


def encode_r4(t) -> E.Targets:
    g = t.graph
    S = (E.TILE, E.TILE)
    arr = {"mask": E.road_mask(t)}
    # lines head = thin centreline band (3 px), the road analogue of B3's outline channel: touching
    # road surfaces (dual carriageways) stay separate centrelines (smoke-tile fix, README)
    arr["line"] = rasterize([(LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]]).buffer(1.5), 1) for u, v in g.edges()],
                            out_shape=S, transform=E.IDENT, dtype="uint8").astype(np.float32)
    pts = {
        "junction": np.array([g.nodes[n]["xy"] for n in g if g.degree(n) >= 3]).reshape(-1, 2),
        "end": np.array([g.nodes[n]["xy"] for n in g if g.degree(n) == 1 and not E._near_border(g.nodes[n]["xy"])]).reshape(-1, 2),
        "crossing": np.array(t.crossings, float).reshape(-1, 2),
    }
    for c in CLASSES:
        arr[c] = E.gaussian_points(pts[c], 2.0) if len(pts[c]) else np.zeros(S, np.float32)
    d = np.zeros((4, *S), np.float32)  # mode 1: channels 0-1, mode 2: channels 2-3
    owner = np.full(S, -1, np.int64)
    ang1 = np.full(S, np.nan, np.float32)
    cid = _chains(g)
    for u, v, data in g.edges(data=True):
        p, q = g.nodes[u]["xy"], g.nodes[v]["xy"]
        th = math.atan2(q[1] - p[1], q[0] - p[0])
        m = rasterize([(LineString([p, q]).buffer(data["width"] / 2), 1)], out_shape=S,
                      transform=E.IDENT, dtype="uint8").astype(bool)
        c = cid[frozenset((u, v))]
        first = m & (owner < 0)
        d[0][first], d[1][first] = math.cos(2 * th), math.sin(2 * th)
        owner[first], ang1[first] = c, th
        dang = np.abs(((ang1 - th) + math.pi / 2) % math.pi - math.pi / 2)
        second = m & (owner >= 0) & (owner != c) & (dang > math.radians(20))
        d[2][second], d[3][second] = math.cos(2 * th), math.sin(2 * th)
    arr["direction"] = d
    return E.Targets(arr, ("mask", "line") + CLASSES, {c: pts[c] for c in CLASSES if len(pts[c])})


def corrupt_r4(tg, t, cfg: dict, seed: int) -> dict:
    out = E.corrupt(tg, t, cfg, seed)
    d = tg.arrays["direction"].copy()
    if cfg["occ"] > 0:
        d[:, E._disk_mask(E.occlusion_disks(t, cfg["occ"], seed))] = 0
    if cfg["vec"] > 0:
        d = d + E._rng(seed, t, "direction").normal(0, 0.1 * cfg["vec"], d.shape).astype(np.float32)
    out["direction"] = d
    return out


def _clip(ln: LineString, parts: list, tree, inside: bool):
    """``ln`` minus (or intersected with) the union of the zone ``parts`` it touches; only those
    parts are unioned (speed fix: differencing against the whole multi-disk union dominated)."""
    hit = tree.query(ln, predicate="intersects")
    if len(hit) == 0:
        return ln if not inside else LineString()
    z = unary_union([parts[i] for i in hit]) if len(hit) > 1 else parts[int(hit[0])]
    return ln.intersection(z) if inside else ln.difference(z)


def _heading(line_coords: np.ndarray, at_end: bool, back: float = 6.0) -> np.ndarray:
    """Unit vector pointing *into* the zone from the arm's line (the end at the zone boundary)."""
    c = line_coords if at_end else line_coords[::-1]
    tip = c[-1]
    L = LineString(c)
    p = np.asarray(L.interpolate(max(L.length - back, 0)).coords[0])
    v = tip - p
    n = np.hypot(*v)
    return v / n if n > 1e-6 else np.zeros(2)


def _refine_heading(h: np.ndarray, tip: np.ndarray, d: np.ndarray, back: float = 6.0) -> np.ndarray:
    """Direction-field axis nearest the skeleton heading, sampled over the last ``back`` px; sign
    from the skeleton. Falls back to the skeleton heading where the field is weak."""
    if not h.any():
        return h
    H, W = d.shape[1:]
    best, best_dot = None, 0.0
    for k in (0, 2):
        vs = []
        for s in np.linspace(0, back, 4):
            x, y = tip - h * s
            c, r = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
            vs.append(d[k:k + 2, r, c])
        cs = np.mean(vs, 0)
        if np.hypot(*cs) < 0.3:
            continue
        th = 0.5 * math.atan2(cs[1], cs[0])
        axis = np.array([math.cos(th), math.sin(th)])
        dot = abs(float(axis @ h))
        if dot > best_dot:
            best, best_dot = axis * np.sign(axis @ h), dot
    return best if best is not None and best_dot > math.cos(math.radians(30)) else h


def decode_r4(a: dict, p: dict) -> nx.Graph:
    surf = E.decode_r0({"mask": a["mask"]}, dict(p["r0"], gap=0))
    H, W = a["mask"].shape

    def on_mask(q):  # keypoints live on the line support (smoke-tile fix: noise peaks off-road)
        if not len(q):
            return q
        c = np.clip(q[:, 0].astype(int), 0, W - 1); r = np.clip(q[:, 1].astype(int), 0, H - 1)
        return q[a["mask"][r, c] > p["r0"]["thr"]]

    peaks = {c: on_mask(E._peaks(a[c], p["thr_k"], 5)) for c in CLASSES}
    allk = [(c, q) for c in CLASSES for q in peaks[c]]
    mode = p.get("mode", "hybrid")
    if mode == "surface":
        r0 = surf
    elif mode == "mincost":  # R6 (E042)
        line = E.decode_r0({"mask": a["line"]}, dict(p["r0"], thr=p["thr_line"], gap=0))
        r0 = mincost_heal(line, a, p, peaks["end"])
    elif mode == "centre":
        line = E.decode_r0({"mask": a["line"]}, dict(p["r0"], thr=p["thr_line"], gap=0))
        r0 = surface_heal(line, surf, 80.0)
    else:
        r0 = with_centrelines(surf, a, p, np.array([q for c, q in allk if c != "end"]).reshape(-1, 2))
    if not allk:
        return E.bridge_ends(r0 if p.get("mode") != "surface" else surf, p["gap"])
    zone = unary_union([Point(q).buffer(p["zr"]) for _, q in allk])
    zones = list(getattr(zone, "geoms", [zone]))
    ztree0 = shapely.STRtree(zones)
    outside = []
    for u, v in r0.edges():
        ln = _clip(LineString([r0.nodes[u]["xy"], r0.nodes[v]["xy"]]), zones, ztree0, inside=False)
        outside += [np.asarray(x.coords) for x in getattr(ln, "geoms", [ln]) if not x.is_empty and x.length > 0]
    segs = list(outside)
    bnd = [z.boundary for z in zones]
    btree = shapely.STRtree(bnd)  # spatial index (speed fix; same results)
    arms = {i: [] for i in range(len(zones))}  # zone -> [(tip, heading)]
    tips_seen = set()
    for c in outside:
        for at_end in (False, True):
            tip = c[-1] if at_end else c[0]
            key = (round(tip[0], 3), round(tip[1], 3))
            hits = sorted(int(i) for i in btree.query(Point(tip), predicate="dwithin", distance=0.5))
            if not hits or key in tips_seen:
                continue
            tips_seen.add(key)
            h = _refine_heading(_heading(c, at_end), tip, a["direction"])
            arms[hits[0]].append((tip, h))
    cos_pair = math.cos(math.radians(p["pair_deg"]))
    ztree = shapely.STRtree(zones)
    inzone = {c: defaultdict(list) for c in CLASSES}
    for c in CLASSES:
        for q in peaks[c]:
            for i in ztree.query(Point(q), predicate="within"):
                inzone[c][int(i)].append(q)
    for i, z in enumerate(zones):
        A = arms[i]
        if not A:
            continue
        ks = {c: inzone[c][i] for c in CLASSES}
        if ks["junction"]:
            J = np.array(ks["junction"])
            for tip, _ in A:
                segs.append(np.array([tip, J[np.argmin(np.hypot(*(J - tip).T))]]))
            if len(J) > 1:
                D = np.hypot(*(J[:, None] - J[None]).transpose(2, 0, 1))
                mst = minimum_spanning_tree(D).tocoo()
                segs += [np.array([J[r], J[c]]) for r, c in zip(mst.row, mst.col)]
        elif ks["crossing"]:
            free = list(range(len(A)))
            cand = sorted(((float(A[x][1] @ A[y][1]), x, y) for x in free for y in free if x < y))
            used = set()
            for dot, x, y in cand:  # most opposite first (dot near -1)
                if dot > cos_pair:
                    break
                if x in used or y in used:
                    continue
                used |= {x, y}
                segs.append(np.array([A[x][0], A[y][0]]))
            C = np.array(ks["crossing"][0])
            segs += [np.array([A[x][0], C]) for x in free if x not in used]
        elif ks["end"]:
            Q = np.array(ks["end"][0])
            segs += [np.array([tip, Q]) for tip, _ in A]
    g = _graph(segs)
    for comp in list(nx.connected_components(g)):
        if E.graph_length(g.subgraph(comp)) < 20:
            g.remove_nodes_from(comp)
    return E.bridge_ends(g, p["gap"])


def with_centrelines(surf: nx.Graph, a: dict, p: dict, kp: np.ndarray, r: float = 15.0) -> nx.Graph:
    """Robust base = road-surface skeleton. Where it branches with no junction/crossing keypoint
    within ``zr`` (a mask artefact: touching parallel roads make ladders), replace it inside a
    disk of radius ``r`` by the thin centreline channel's skeleton, joined in vector space
    (smoke-tile design, README)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(kp) if len(kp) else None
    art = [surf.nodes[n]["xy"] for n in surf if surf.degree(n) >= 3
           and (tree is None or tree.query(surf.nodes[n]["xy"])[0] > p["zr"])]
    if not art:
        return surf
    zone = unary_union([Point(q).buffer(r) for q in art])
    parts = list(getattr(zone, "geoms", [zone]))
    ptree = shapely.STRtree(parts)
    line = E.decode_r0({"mask": a["line"]}, dict(p["r0"], thr=p["thr_line"], gap=0, spur=5, min_len=0))
    out_l, in_l = [], []
    for g_, keep_out, dst in ((surf, True, out_l), (line, False, in_l)):
        for u, v in g_.edges():
            ln = _clip(LineString([g_.nodes[u]["xy"], g_.nodes[v]["xy"]]), parts, ptree, inside=not keep_out)
            dst += [np.asarray(x.coords) for x in getattr(ln, "geoms", [ln]) if not x.is_empty and x.length > 0 and x.geom_type == "LineString"]
    g = _graph(out_l + in_l)
    bnd = zone.boundary
    deg = dict(g.degree())
    eo = [n for n in g if deg[n] == 1 and bnd.distance(Point(g.nodes[n]["xy"])) <= 0.5]
    ei = [n for n in g if deg[n] == 1 and bnd.distance(Point(g.nodes[n]["xy"])) <= 2.0 and n not in eo]
    if eo and ei:
        xo = np.array([g.nodes[n]["xy"] for n in eo]); xi = np.array([g.nodes[n]["xy"] for n in ei])
        d = np.hypot(*(xo[:, None] - xi[None]).transpose(2, 0, 1))
        uo, ui = set(), set()
        for flat in np.argsort(d, axis=None):
            k, m = divmod(int(flat), d.shape[1])
            if d[k, m] > 6.0:
                break
            if k not in uo and m not in ui:
                uo.add(k); ui.add(m); g.add_edge(eo[k], ei[m])
    return g


def mincost_heal(gl: nx.Graph, a: dict, p: dict, end_kp: np.ndarray) -> nx.Graph:
    """R6 (E042): from each dead end of the centreline graph (not at the tile border, not within
    ``zr`` of an end keypoint), run a geometric minimum-cost path search in a window of radius
    ``heal_r`` over cost = -log(max(line, alpha * surface)) (centreline preferred, road surface a
    weak corridor, off-road expensive) to the nearest pixel of a *different* part of the graph
    (another component, or > 3x the straight distance away along the graph); the path is added if
    its cost is below ``heal_cost``. Ordered paths first, edges second (Codex's R5 diagnosis)."""
    from scipy.spatial import cKDTree
    from skimage.graph import MCP_Geometric
    H, W = a["line"].shape
    R = int(p["heal_r"])
    support = np.maximum(a["line"], p.get("alpha", 0.3) * a["mask"])
    cost_full = np.minimum(-np.log(np.clip(support, 1e-3, 1.0)), p.get("cost_cap", 7.0)) + 0.05  # capped: occluded gaps stay crossable
    g = gl.copy()
    if g.number_of_nodes() == 0:
        return g
    comp = {n: k for k, c in enumerate(nx.connected_components(g)) for n in c}
    etree = cKDTree(end_kp) if len(end_kp) else None
    w = lambda u, v, _: float(np.hypot(*(g.nodes[u]["xy"] - g.nodes[v]["xy"])))  # noqa: E731
    ends = [n for n in gl if gl.degree(n) == 1 and not E._near_border(gl.nodes[n]["xy"])
            and (etree is None or etree.query(gl.nodes[n]["xy"])[0] > p["zr"])]
    for e in ends:
        x, y = g.nodes[e]["xy"]
        r0, r1, c0, c1 = max(int(y) - R, 0), min(int(y) + R + 1, H), max(int(x) - R, 0), min(int(x) + R + 1, W)
        # graph pixels in the window that belong to another part of the graph
        along = nx.single_source_dijkstra_path_length(g, e, cutoff=3 * R, weight=w)
        tgt = {}
        for u, v in g.edges():
            pu, pv = g.nodes[u]["xy"], g.nodes[v]["xy"]
            L = float(np.hypot(*(pv - pu)))
            for t in np.linspace(0, 1, max(int(L), 1) + 1):
                q = pu + t * (pv - pu)
                rr, cc = int(q[1]), int(q[0])
                if not (r0 <= rr < r1 and c0 <= cc < c1):
                    continue
                n = u if t < 0.5 else v
                straight = float(np.hypot(*(g.nodes[n]["xy"] - g.nodes[e]["xy"])))
                if n == e or (comp.get(n) == comp.get(e) and along.get(n, np.inf) <= 3 * max(straight, 1.0)):
                    continue
                tgt[(rr - r0, cc - c0)] = n
        if not tgt:
            continue
        mcp = MCP_Geometric(cost_full[r0:r1, c0:c1])
        start = (min(max(int(y) - r0, 0), r1 - r0 - 1), min(max(int(x) - c0, 0), c1 - c0 - 1))
        cum, _ = mcp.find_costs([start])
        best = min(tgt, key=lambda k: cum[k])
        if cum[best] > p["heal_cost"]:
            continue
        path = mcp.traceback(best)
        prev = e
        for k, (rr, cc) in enumerate(path[1:-1:3]):  # every 3rd pixel keeps the geometry, few nodes
            nid = ("mc", e, k)
            g.add_node(nid, xy=np.array([cc + c0 + 0.5, rr + r0 + 0.5]))
            g.add_edge(prev, nid)
            prev = nid
        g.add_edge(prev, tgt[best])
        comp_e, comp_t = comp.get(e), comp.get(tgt[best])
        if comp_e != comp_t:  # merge components so later ends see the joined graph
            for n, k in list(comp.items()):
                if k == comp_t:
                    comp[n] = comp_e
    return g


def surface_heal(gl: nx.Graph, gs: nx.Graph, max_len: float = 80.0, snap: float = 4.0) -> nx.Graph:
    """Fallback to the robust channel: from each dead end of the centreline graph ``gl``, follow
    the road-surface skeleton ``gs`` (shortest path, <= ``max_len`` px) to the nearest surface node
    that lies on a *different* part of ``gl`` (another component, or > 3x the straight distance
    away along ``gl``), and add that path. Paths must start at a dead end, so ladder rungs between
    touching lanes (which join lanes mid-way) are never added (smoke-tile fix, README)."""
    from scipy.spatial import cKDTree
    if gl.number_of_nodes() == 0 or gs.number_of_nodes() == 0:
        return gl
    g = gl.copy()
    ln = list(g); lxy = np.array([g.nodes[n]["xy"] for n in ln]); ltree = cKDTree(lxy)
    sn = list(gs); sxy = np.array([gs.nodes[n]["xy"] for n in sn]); stree = cKDTree(sxy)
    for u, v in gs.edges():
        gs.edges[u, v]["w"] = float(np.hypot(*(gs.nodes[u]["xy"] - gs.nodes[v]["xy"])))
    comp = {n: k for k, c in enumerate(nx.connected_components(g)) for n in c}
    ends = [n for n in gl if gl.degree(n) == 1 and not E._near_border(gl.nodes[n]["xy"])]
    for e in ends:
        d0, j0 = stree.query(g.nodes[e]["xy"])
        if d0 > snap:
            continue
        src = sn[j0]
        dist, paths = nx.single_source_dijkstra(gs, src, cutoff=max_len, weight="w")
        # one bounded search along gl: the test below only asks whether gl's path is > 3 x L
        # with L <= max_len, so a 3 x max_len cutoff gives identical answers (speed fix)
        along_all = nx.single_source_dijkstra_path_length(
            gl, e, cutoff=3 * max_len,
            weight=lambda a_, b_, _: float(np.hypot(*(gl.nodes[a_]["xy"] - gl.nodes[b_]["xy"]))))
        best = None
        for t_, L in sorted(dist.items(), key=lambda kv: kv[1]):
            if L < 2 * snap:
                continue
            dd, jj = ltree.query(gs.nodes[t_]["xy"])
            if dd > snap:
                continue
            tgt = ln[jj]
            if tgt == e:
                continue
            if comp.get(tgt) != comp.get(e):
                best = (t_, tgt); break
            along = along_all.get(tgt, np.inf)
            if along > 3 * L:
                best = (t_, tgt); break
        if best is None:
            continue
        path = paths[best[0]]
        prev = e
        for k, s_ in enumerate(path):
            nid = ("heal", e, k)
            g.add_node(nid, xy=np.asarray(gs.nodes[s_]["xy"], float))
            g.add_edge(prev, nid)
            prev = nid
        g.add_edge(prev, best[1])
    return g


def _drop_rungs(g: nx.Graph, kp: np.ndarray, zr: float, max_len: float = 30.0):
    """Nodes come from keypoints: a node of degree >= 3 with no keypoint within ``zr`` is a mask
    artefact (ladder rungs between touching parallel roads); drop its shortest incident chain if
    shorter than ``max_len`` px (smoke-tile fix, README)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(kp) if len(kp) else None
    for _ in range(3):
        changed = False
        for n in [n for n in g if g.degree(n) >= 3]:
            if n not in g or g.degree(n) < 3:
                continue
            if tree is not None and tree.query(g.nodes[n]["xy"])[0] <= zr:
                continue
            best = None
            for m in g.neighbors(n):  # walk each incident chain to its next branch node
                path, prev, cur, L = [n, m], n, m, float(np.hypot(*(g.nodes[m]["xy"] - g.nodes[n]["xy"])))
                while g.degree(cur) == 2:
                    nxt = next(x for x in g.neighbors(cur) if x != prev)
                    L += float(np.hypot(*(g.nodes[nxt]["xy"] - g.nodes[cur]["xy"])))
                    prev, cur = cur, nxt
                    path.append(cur)
                    if L > max_len:
                        break
                if g.degree(cur) >= 3 and L <= max_len and (best is None or L < best[0]):
                    best = (L, path)
            if best is not None:
                g.remove_edges_from(zip(best[1][:-1], best[1][1:]))
                g.remove_nodes_from([x for x in best[1][1:-1] if g.degree(x) == 0])
                changed = True
        if not changed:
            break


def _graph(segs) -> nx.Graph:
    """Polylines -> graph; coincident points merged; crossings of different lines are NOT noded."""
    g = nx.Graph()
    ids: dict = {}

    def nid(q):
        k = (round(float(q[0]), 3), round(float(q[1]), 3))
        if k not in ids:
            ids[k] = len(ids)
            g.add_node(ids[k], xy=np.array([float(q[0]), float(q[1])]))
        return ids[k]

    for s in segs:
        for u, v in zip(s[:-1], s[1:]):
            a, b = nid(u), nid(v)
            if a != b:
                g.add_edge(a, b)
    return g
