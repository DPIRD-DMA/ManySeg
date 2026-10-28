"""Road graphs from the unified road heads (E039 R4, adopted as the road default in E044b).

Heads: road surface probability, keypoint heatmaps (junction, end, crossing) and a two-mode
``(cos 2t, sin 2t)`` direction field. Decoding: the surface skeleton graph (as
:func:`~manyseg.vectors.graph_from_mask`) is cut out of a disk around every keypoint; inside a
zone the arms that reach it are rejoined by the keypoint's class: to the junction (several
junctions in one zone are linked by a minimum spanning tree), paired straight through a
crossing (most opposite headings first, so an overpass stays two unconnected roads), or to
the road end. On City-Scale (E044b, three seeds) this kept ~55% of grade-separated crossings
against ~32% for the plain skeleton, with no loss of APLS or junction F1.
"""

from __future__ import annotations

import math

import numpy as np

from .graph import VectorGraph, _deps, _nx_from_skeleton, bridge_ends
from .graph_tensor import _peak_pixels, _refine

CLASSES = ("junction", "end", "crossing")


def _heading(c: np.ndarray, at_end: bool, back: float = 6.0) -> np.ndarray:
    """Unit vector pointing into the zone along an arm (the arm's end at the zone boundary)."""
    shapely = _deps()[1]
    c = c if at_end else c[::-1]
    L = shapely.LineString(c)
    v = c[-1] - np.asarray(L.interpolate(max(L.length - back, 0)).coords[0])
    n = float(np.hypot(*v))
    return v / n if n > 1e-6 else np.zeros(2)


def _refine_heading(h: np.ndarray, tip: np.ndarray, d: np.ndarray, back: float = 6.0) -> np.ndarray:
    """The direction-field mode nearest the skeleton heading (sampled over the last ``back`` px),
    signed by the skeleton; the skeleton heading where the field is weak or disagrees > 30°."""
    if not h.any():
        return h
    H, W = d.shape[1:]
    best, best_dot = None, 0.0
    for k in (0, 2):
        vs = []
        for s in np.linspace(0, back, 4):
            x, y = tip - h * s
            vs.append(d[k : k + 2, int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
        cs = np.mean(vs, 0)
        if np.hypot(*cs) < 0.3:
            continue
        th = 0.5 * math.atan2(cs[1], cs[0])
        axis = np.array([math.cos(th), math.sin(th)])
        dot = abs(float(axis @ h))
        if dot > best_dot:
            best, best_dot = axis * np.sign(axis @ h), dot
    return best if best is not None and best_dot > math.cos(math.radians(30)) else h


def _segments_graph(segs):
    """Polylines -> graph; coincident points merged; crossing lines are not noded."""
    nx = _deps()[0]
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


def graph_from_road_heads(
    mask,
    keypoints,
    direction,
    *,
    threshold: float = 0.3,
    keypoint_threshold: float = 0.2,
    zone_radius: float = 8.0,
    pair_deg: float = 150.0,
    spur: float = 30,
    gap: float = 50,
    simplify: float = 2.0,
    min_length: float = 20,
    min_area: int = 30,
    scale: float = 1.0,
) -> VectorGraph:
    """Road graph from the unified road heads (numpy or torch; probabilities, not logits).

    Args:
        mask: ``(H, W)`` road-surface probability.
        keypoints: ``(3, H, W)`` junction, end and crossing heatmaps.
        direction: ``(4, H, W)`` two ``(cos 2t, sin 2t)`` modes (the second only at crossings).
        threshold, spur, gap, simplify, min_length, min_area: as :func:`graph_from_mask` on the
            surface (gap bridging runs once, on the final graph).
        keypoint_threshold: Heatmap peak threshold; peaks off the road surface are ignored.
        zone_radius: Radius (px) of the disk around each keypoint where arms are rejoined.
        pair_deg: Arms through a crossing are paired if their headings are at least this far apart.
        scale: The heads' ``scale``; coordinates are returned in input pixels.

    Defaults are E044's City-Scale tuning (1 m pixels).
    """
    nx, shapely, _, remove_small_objects, skeletonize = _deps()
    from scipy.sparse.csgraph import minimum_spanning_tree

    def arr(x):
        return (
            x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, np.float32)
        )

    m, kp, d = arr(mask), arr(keypoints), arr(direction)
    H, W = m.shape
    fg = remove_small_objects(m > threshold, max_size=max(min_area - 1, 0))
    surf = _nx_from_skeleton(
        *np.nonzero(skeletonize(fg)), (H, W), spur=spur, simplify=simplify, min_length=min_length
    )

    def on_mask(q):
        if not len(q):
            return q
        return q[
            m[np.clip(q[:, 1].astype(int), 0, H - 1), np.clip(q[:, 0].astype(int), 0, W - 1)]
            > threshold
        ]

    peaks = {
        c: on_mask(_refine(kp[i], _peak_pixels(kp[i], keypoint_threshold, 5)))
        for i, c in enumerate(CLASSES)
    }
    allk = [q for c in CLASSES for q in peaks[c]]
    if not allk:
        g = surf
    else:
        g = _rejoin(surf, peaks, allk, d, zone_radius, pair_deg, nx, shapely, minimum_spanning_tree)
        for comp in list(nx.connected_components(g)):
            sub = g.subgraph(comp)
            if (
                sum(float(np.hypot(*(g.nodes[a]["xy"] - g.nodes[b]["xy"]))) for a, b in sub.edges())
                < 20
            ):
                g.remove_nodes_from(comp)
    bridge_ends(g, gap, (H, W))
    vg = VectorGraph.from_networkx(g)
    if scale != 1:
        vg.xy = vg.xy / scale
    return vg


def _rejoin(surf, peaks, allk, d, zr, pair_deg, nx, shapely, minimum_spanning_tree):
    from collections import defaultdict

    zone = shapely.union_all([shapely.Point(q).buffer(zr) for q in allk])
    zones = list(getattr(zone, "geoms", [zone]))
    ztree = shapely.STRtree(zones)

    def clip(ln):
        hit = ztree.query(ln, predicate="intersects")
        if len(hit) == 0:
            return ln
        z = shapely.union_all([zones[i] for i in hit]) if len(hit) > 1 else zones[int(hit[0])]
        return ln.difference(z)

    outside = []
    for u, v in surf.edges():
        ln = clip(shapely.LineString([surf.nodes[u]["xy"], surf.nodes[v]["xy"]]))
        outside += [
            np.asarray(x.coords)
            for x in getattr(ln, "geoms", [ln])
            if not x.is_empty and x.length > 0
        ]
    segs = list(outside)
    btree = shapely.STRtree([z.boundary for z in zones])
    arms = {i: [] for i in range(len(zones))}  # zone -> [(tip, heading into the zone)]
    seen = set()
    for c in outside:
        for at_end in (False, True):
            tip = c[-1] if at_end else c[0]
            key = (round(tip[0], 3), round(tip[1], 3))
            hits = sorted(
                int(i) for i in btree.query(shapely.Point(tip), predicate="dwithin", distance=0.5)
            )
            if not hits or key in seen:
                continue
            seen.add(key)
            arms[hits[0]].append((tip, _refine_heading(_heading(c, at_end), tip, d)))
    cos_pair = math.cos(math.radians(pair_deg))
    inzone = {c: defaultdict(list) for c in CLASSES}
    for c in CLASSES:
        for q in peaks[c]:
            for i in ztree.query(shapely.Point(q), predicate="within"):
                inzone[c][int(i)].append(q)
    for i in range(len(zones)):
        A = arms[i]
        if not A:
            continue
        if inzone["junction"][i]:
            J = np.array(inzone["junction"][i])
            segs += [np.array([tip, J[np.argmin(np.hypot(*(J - tip).T))]]) for tip, _ in A]
            if len(J) > 1:
                D = np.hypot(*(J[:, None] - J[None]).transpose(2, 0, 1))
                mst = minimum_spanning_tree(D).tocoo()
                segs += [np.array([J[r], J[c]]) for r, c in zip(mst.row, mst.col)]
        elif inzone["crossing"][i]:
            cand = sorted(
                (float(A[x][1] @ A[y][1]), x, y)
                for x in range(len(A))
                for y in range(x + 1, len(A))
            )
            used = set()
            for dot, x, y in cand:  # most opposite first
                if dot > cos_pair:
                    break
                if x in used or y in used:
                    continue
                used |= {x, y}
                segs.append(np.array([A[x][0], A[y][0]]))
            C = np.array(inzone["crossing"][i][0])
            segs += [np.array([A[x][0], C]) for x in range(len(A)) if x not in used]
        elif inzone["end"][i]:
            Q = np.array(inzone["end"][i][0])
            segs += [np.array([tip, Q]) for tip, _ in A]
    return _segments_graph(segs)
