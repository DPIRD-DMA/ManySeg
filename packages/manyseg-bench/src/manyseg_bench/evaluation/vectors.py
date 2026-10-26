"""Metrics for vector outputs: road graphs (APLS, GEO, junction F1, crossings) and polygons
(C-IoU, detection F1, PoLiS, N-ratio).

Graphs are ``networkx.Graph`` with an ``xy`` node attribute, or any object with ``xy`` ``(N, 2)``
and ``edges`` ``(E, 2)`` arrays (e.g. ``manyseg.vectors.VectorGraph``). Coordinates are pixels;
``gsd`` (metres per pixel) converts the metric thresholds. Reviewed against the definitions in
E028 (Codex review, 2026-10-26), with regression tests for the bugs it found. Needs shapely,
networkx and scipy (the ``vectors`` extra).
"""

from __future__ import annotations

import math

import numpy as np


def _graph(g):
    import networkx as nx

    if isinstance(g, nx.Graph):
        return g
    out = nx.Graph()
    xy = np.asarray(g.xy, float).reshape(-1, 2)
    for i, p in enumerate(xy):
        out.add_node(i, xy=p)
    out.add_edges_from((int(a), int(b)) for a, b in np.asarray(g.edges).reshape(-1, 2) if a != b)
    return out


def _chains(g) -> list[list]:
    """Node paths between key nodes (degree != 2); a pure cycle starts and ends at one node."""
    key = {n for n in g if g.degree(n) != 2}
    seen: set = set()
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
                seen.add(frozenset((cur, nxt[0])))
                path.append(nxt[0])
                if path[-1] == k:
                    break
            chains.append(path)
    for u, v in g.edges():
        if u == v or frozenset((u, v)) in seen:
            continue
        path = [u, v]
        seen.add(frozenset((u, v)))
        while path[-1] != u:
            cur, prev = path[-1], path[-2]
            nxt = [n for n in g.neighbors(cur) if n != prev]
            if not nxt or (frozenset((cur, nxt[0])) in seen and nxt[0] != u):
                break
            seen.add(frozenset((cur, nxt[0])))
            path.append(nxt[0])
        chains.append(path)
    return chains


def _segments(g):
    es = list(g.edges())
    p = np.array([g.nodes[u]["xy"] for u, _ in es], float).reshape(-1, 2)
    q = np.array([g.nodes[v]["xy"] for _, v in es], float).reshape(-1, 2)
    return es, p, q


def _control_graph(g, spacing: float):
    """Control points (key nodes, then every ``spacing`` along each chain; a loop gets at least
    two) and weighted edges between consecutive control points."""
    pts: list = []
    index: dict = {}
    edges = []

    def node(n):
        if n not in index:
            index[n] = len(pts)
            pts.append(np.asarray(g.nodes[n]["xy"], float))
        return index[n]

    for ch in _chains(g):
        c = np.array([g.nodes[n]["xy"] for n in ch], float)
        seg = np.hypot(*np.diff(c, axis=0).T)
        L = seg.sum()
        if L <= 0:
            continue
        cum = np.concatenate([[0], np.cumsum(seg)])
        prev, prev_s = node(ch[0]), 0.0
        step = min(spacing, L / 2) if ch[0] == ch[-1] else spacing
        s = step
        while s < L - 1e-6:
            k = min(int(np.searchsorted(cum, s, side="right") - 1), len(seg) - 1)
            pts.append(c[k] + (c[k + 1] - c[k]) * (s - cum[k]) / max(seg[k], 1e-12))
            edges.append((prev, len(pts) - 1, s - prev_s))
            prev, prev_s = len(pts) - 1, s
            s += step
        end = node(ch[-1])
        if end != prev or L - prev_s > 1e-9:
            edges.append((prev, end, L - prev_s))
    return np.array(pts).reshape(-1, 2), edges


def _csr(n: int, edges):
    """Symmetric weights; parallel edges keep their minimum (SciPy would sum duplicates)."""
    from scipy.sparse import csr_matrix

    best: dict = {}
    for a, b, w in edges:
        if a != b:
            k = (min(a, b), max(a, b))
            best[k] = min(best.get(k, np.inf), float(w))
    if not best:
        return csr_matrix((n, n))
    i, j = map(np.asarray, zip(*best))
    w = np.maximum(np.fromiter(best.values(), float), 1e-9)
    return csr_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                      shape=(n, n))


def _apls_one(src, dst, spacing: float, buffer: float) -> float:
    import shapely
    from scipy.sparse.csgraph import dijkstra

    cp, ce = _control_graph(src, spacing)
    n = len(cp)
    es, p, q = _segments(dst)
    if n < 2 or len(es) == 0:
        return 0.0
    la = dijkstra(_csr(n, ce), directed=False)
    tree = shapely.STRtree(shapely.linestrings(np.stack([p, q], 1)))
    (ci, ei), _ = tree.query_nearest(shapely.points(cp), max_distance=buffer, return_distance=True,
                                     all_matches=False)
    dnodes = {nd: k for k, nd in enumerate(dst.nodes())}
    nb = len(dnodes)
    snap = -np.ones(n, int)
    on_edge: dict = {}
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
        bedges += [(a, b, (t1 - t0) * L) for (t0, a), (t1, b) in zip(chain[:-1], chain[1:])]
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
    return float(1 - np.where(np.isfinite(B), np.minimum(1.0, np.abs(A - B) / A), 1.0).mean())


def apls(gt, pred, gsd: float = 1.0, *, spacing_m: float = 50.0, buffer_m: float = 5.0) -> float:
    """Average Path Length Similarity (SpaceNet): control points every ``spacing_m`` plus all
    key nodes, snapped to the other graph within ``buffer_m``; per pair of control points the
    penalty is ``min(1, |L_gt - L_pred| / L_gt)``, 1 for a missing path. Harmonic mean of the
    GT -> prediction and prediction -> GT scores, in [0, 1]."""
    gt, pred = _graph(gt), _graph(pred)
    if pred.number_of_edges() == 0 or gt.number_of_edges() == 0:
        return 0.0
    a = _apls_one(gt, pred, spacing_m / gsd, buffer_m / gsd)
    b = _apls_one(pred, gt, spacing_m / gsd, buffer_m / gsd)
    return 0.0 if a <= 0 or b <= 0 else 2 * a * b / (a + b)


def _samples(g, step: float) -> np.ndarray:
    """Points every ``step`` of arc length along each chain."""
    out = []
    for ch in _chains(g):
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


def _dist_to_graph(pts: np.ndarray, g) -> np.ndarray:
    import shapely

    es, p, q = _segments(g)
    out = np.full(len(pts), np.inf)
    if len(es) == 0 or len(pts) == 0:
        return out
    tree = shapely.STRtree(shapely.linestrings(np.stack([p, q], 1)))
    (ii, _), d = tree.query_nearest(shapely.points(pts), return_distance=True, all_matches=False)
    out[ii] = d
    return out


def geo_f1(gt, pred, gsd: float = 1.0, *, buffer_m: float = 5.0) -> dict:
    """Geometric precision / recall / F1: samples every metre of each graph within
    ``buffer_m`` of the other graph."""
    gt, pred = _graph(gt), _graph(pred)
    sg, sp = _samples(gt, 1.0 / gsd), _samples(pred, 1.0 / gsd)
    if len(sp) == 0 or len(sg) == 0:
        return {"geo_p": 0.0, "geo_r": 0.0, "geo_f1": 0.0}
    prec = float((_dist_to_graph(sp, gt) <= buffer_m / gsd).mean())
    rec = float((_dist_to_graph(sg, pred) <= buffer_m / gsd).mean())
    return {"geo_p": prec, "geo_r": rec, "geo_f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)}


def _junctions(g) -> np.ndarray:
    return np.array([g.nodes[n]["xy"] for n in g if g.degree(n) >= 3], float).reshape(-1, 2)


def _match(a: np.ndarray, b: np.ndarray, radius: float) -> int:
    from scipy.optimize import linear_sum_assignment

    if len(a) == 0 or len(b) == 0:
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    cost = np.where(d <= radius, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return int((cost[r, c] <= radius).sum())


def junction_f1(gt, pred, gsd: float = 1.0, *, radius_m: float = 5.0) -> float:
    """F1 of degree >= 3 nodes, matched one-to-one within ``radius_m``."""
    a, b = _junctions(_graph(gt)), _junctions(_graph(pred))
    if len(a) + len(b) == 0:
        return 1.0
    return 2 * _match(a, b, radius_m / gsd) / (len(a) + len(b))


def crossing_score(crossings, pred, gsd: float = 1.0, *, junction_m: float = 8.0,
                   cover_m: float = 5.0) -> dict:
    """Grade-separated crossings (bridges, overpasses; ``(K, 2)`` GT points where roads cross
    without a junction): ``no_junction`` counts those with no predicted junction within
    ``junction_m``; ``covered`` those with a predicted road within ``cover_m``; ``covered_ok``
    both."""
    from scipy.spatial import cKDTree

    pred = _graph(pred)
    cr = np.asarray(crossings, float).reshape(-1, 2)
    if len(cr) == 0:
        return {"n": 0, "no_junction": 0, "covered": 0, "covered_ok": 0}
    cov = _dist_to_graph(cr, pred) <= cover_m / gsd
    jn = _junctions(pred)
    dj = cKDTree(jn).query(cr)[0] if len(jn) else np.full(len(cr), np.inf)
    nj = dj > junction_m / gsd
    return {"n": len(cr), "no_junction": int(nj.sum()), "covered": int(cov.sum()),
            "covered_ok": int((cov & nj).sum())}


def graph_metrics(gt, pred, gsd: float = 1.0) -> dict:
    """APLS, GEO precision / recall / F1 and junction F1 in one dict."""
    gt, pred = _graph(gt), _graph(pred)
    return {"apls": apls(gt, pred, gsd), "junction_f1": junction_f1(gt, pred, gsd), **geo_f1(gt, pred, gsd)}


# --------------------------------------------------------------------------------------
# Polygons
# --------------------------------------------------------------------------------------


def _n_vertices(p) -> int:
    return len(p.exterior.coords) - 1


def polis(a, b) -> float:
    """PoLiS distance between two polygons (mean vertex-to-boundary distance, both ways)."""
    import shapely

    va, vb = np.asarray(a.exterior.coords)[:-1], np.asarray(b.exterior.coords)[:-1]
    da = shapely.distance(shapely.points(va), b.exterior).mean()
    db = shapely.distance(shapely.points(vb), a.exterior).mean()
    return float(da / 2 + db / 2)


def polygon_metrics(gt: list, pred: list, gsd: float = 1.0, *, iou_threshold: float = 0.5,
                    gt_vertex_tolerance: float = 0.5) -> dict:
    """Polygon matching (one-to-one, greedy by IoU above ``iou_threshold``) and metrics:

    * ``det_f1``: detection F1 of the matching;
    * ``ciou``: mean over matched pairs of IoU × (1 - |N_pred - N_gt| / (N_pred + N_gt)), with
      N the exterior vertex count (GT simplified by ``gt_vertex_tolerance`` px to drop collinear
      nodes); only matched pairs count, so read it with ``det_f1`` (``ciou * det_f1`` is a
      PQ-like single number);
    * ``miou``, ``n_ratio`` (Σ N_pred / Σ N_gt over matched pairs) and ``polis`` (metres).
    """
    import shapely

    gt_n = [len(g.simplify(gt_vertex_tolerance).exterior.coords) - 1 for g in gt]
    pairs = []
    if pred and gt:
        tree = shapely.STRtree(pred)
        for i, g in enumerate(gt):
            for j in tree.query(g, predicate="intersects"):
                inter = g.intersection(pred[j]).area
                if inter > 0:
                    iou = inter / (g.area + pred[j].area - inter)
                    if iou > iou_threshold:
                        pairs.append((iou, i, int(j)))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            matched.append((iou, i, j))
    out = {"det_f1": 2 * len(matched) / (len(gt) + len(pred)) if gt or pred else 1.0,
           "n_gt": len(gt), "n_pred": len(pred), "tp": len(matched)}
    if not matched:
        return {**out, "ciou": 0.0, "miou": 0.0, "n_ratio": math.nan, "polis": math.nan}
    np_ = np.array([_n_vertices(pred[j]) for _, _, j in matched])
    ng = np.array([gt_n[i] for _, i, _ in matched])
    ious = np.array([m[0] for m in matched])
    return {**out, "ciou": float(np.mean(ious * (1 - np.abs(np_ - ng) / (np_ + ng)))),
            "miou": float(ious.mean()), "n_ratio": float(np_.sum() / ng.sum()),
            "polis": float(np.mean([polis(gt[i], pred[j]) for _, i, j in matched]) * gsd)}
