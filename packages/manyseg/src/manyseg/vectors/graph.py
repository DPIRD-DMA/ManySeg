"""Line networks (roads, tracks, fences) as graphs, decoded from a line-probability head.

Coordinates are ``(x, y)`` floats in GIS raster convention: pixel ``(row, col)`` covers
``[col, col + 1) x [row, row + 1)``, so its centre is ``(col + 0.5, row + 0.5)`` and an affine
geotransform maps them straight to world coordinates (:func:`manyseg.vectors.to_geojson`). Note
the difference from :mod:`manyseg.points`, which returns ``(y, x)`` with pixel centres at
integers. For a head with ``scale`` < 1, pass ``scale=`` and coordinates come back in input
pixels (``x_input = x_head / scale``).

The decoder is a plain mask baseline: threshold, skeletonize, trace the pixel graph, merge junction
artefacts, prune spurs, join nearby dangling ends, simplify. It keeps no information about
grade separation: two roads that cross on a bridge become a junction.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


def _deps():
    try:
        import networkx
        import shapely
        from scipy.spatial import cKDTree
        from skimage.morphology import skeletonize
    except ImportError as e:  # pragma: no cover - exercised without the extra
        raise ImportError("manyseg.vectors needs the 'vectors' extra: "
                          "pip install 'manyseg[vectors]'") from e
    return networkx, shapely, cKDTree, skeletonize


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """``mask`` without 4-connected blobs under ``min_area`` pixels (``remove_small_objects``
    renamed its size argument between the scikit-image versions we support)."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if not n or min_area <= 1:
        return mask
    small = np.bincount(lab.reshape(-1), minlength=n + 1) < min_area
    small[0] = False
    return mask & ~small[lab]


@dataclass
class VectorGraph:
    """An undirected graph of straight segments: ``xy`` ``(N, 2)`` node coordinates and
    ``edges`` ``(E, 2)`` node indices."""

    xy: np.ndarray
    edges: np.ndarray

    def __post_init__(self):
        self.xy = np.asarray(self.xy, np.float64).reshape(-1, 2)
        self.edges = np.asarray(self.edges, np.int64).reshape(-1, 2)

    @property
    def degree(self) -> np.ndarray:
        return np.bincount(self.edges.ravel(), minlength=len(self.xy))

    @property
    def length(self) -> float:
        d = self.xy[self.edges[:, 1]] - self.xy[self.edges[:, 0]]
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    def junctions(self) -> np.ndarray:
        """Coordinates of nodes with degree >= 3."""
        return self.xy[self.degree >= 3]

    def to_networkx(self):
        """A ``networkx.Graph``: nodes ``0..N-1`` with an ``xy`` attribute, the edges."""
        nx = _deps()[0]
        g = nx.Graph()
        for i, p in enumerate(self.xy):
            g.add_node(i, xy=p)
        g.add_edges_from(map(tuple, self.edges))
        return g

    @classmethod
    def from_networkx(cls, g) -> VectorGraph:
        """From a ``networkx.Graph`` whose nodes have an ``xy`` attribute."""
        index = {n: i for i, n in enumerate(g.nodes())}
        xy = np.array([g.nodes[n]["xy"] for n in g.nodes()]).reshape(-1, 2)
        edges = np.array([(index[u], index[v]) for u, v in g.edges() if u != v]).reshape(-1, 2)
        return cls(xy, edges)

    def polylines(self) -> list[np.ndarray]:
        """The graph as polylines between key nodes (degree != 2); loops close on themselves."""
        g = self.to_networkx()
        return [np.array([g.nodes[n]["xy"] for n in ch]) for ch in graph_chains(g)]


def graph_chains(g) -> list[list]:
    """Node paths of a ``networkx.Graph`` between key nodes (degree != 2). A cycle of degree-2
    nodes is returned once, starting and ending at the same node."""
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


# --------------------------------------------------------------------------------------
# Skeleton -> polylines
# --------------------------------------------------------------------------------------

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_chains(sk: np.ndarray):
    """Skeleton -> (nodes {id: xy}, chains [(u, v, coords)]), junction pixel clusters merged."""
    return _skeleton_chains_rc(*np.nonzero(sk))


def _skeleton_chains_rc(rr: np.ndarray, cc: np.ndarray):
    """As :func:`_skeleton_chains` from skeleton pixel coordinates in raster order (sparse: memory
    grows with the skeleton, not the scene)."""
    nx = _deps()[0]
    rr, cc = np.asarray(rr, np.int64), np.asarray(cc, np.int64)
    if len(rr) == 0:
        return {}, []
    width = int(cc.max()) + 3
    keys = (rr + 1) * width + (cc + 1)  # sorted: raster order

    def idx(r, c):
        k = (r + 1) * width + (c + 1)
        j = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        return np.where((keys[j] == k) & (c >= -1) & (c <= width - 3), j, -1)

    pg = nx.Graph()
    pg.add_nodes_from(range(len(rr)))
    for dr, dc in _OFFS:
        nbr = idx(rr + dr, cc + dc)
        ok = nbr > np.arange(len(rr))
        if dr != 0 and dc != 0:  # drop a diagonal if a 4-neighbour path joins the same pixels
            ok &= (idx(rr + dr, cc) < 0) & (idx(rr, cc + dc) < 0)
        pg.add_edges_from(zip(np.nonzero(ok)[0].tolist(), nbr[ok].tolist()))
    xy = np.stack([cc + 0.5, rr + 0.5], 1)
    cluster, nodes = {}, {}
    hubs = pg.subgraph([n for n in pg if pg.degree(n) >= 3])
    for ci, comp in enumerate(nx.connected_components(hubs)):
        comp = list(comp)
        for n in comp:
            cluster[n] = ("j", ci)
        nodes[("j", ci)] = xy[comp].mean(0)
    for n in pg:
        if pg.degree(n) == 1:
            cluster[n] = ("e", n)
            nodes[("e", n)] = xy[n]
    chains, seen = [], set()
    for start in list(cluster):
        for nb in pg.neighbors(start):
            if (nb in cluster and cluster[nb] == cluster[start]) or frozenset((start, nb)) in seen:
                continue
            seen.add(frozenset((start, nb)))
            path = [start, nb]
            while path[-1] not in cluster:
                cur, prev = path[-1], path[-2]
                nxt = [n for n in pg.neighbors(cur)
                       if n != prev and frozenset((cur, n)) not in seen]
                if not nxt:
                    break
                seen.add(frozenset((cur, nxt[0])))
                path.append(nxt[0])
            if path[-1] in cluster:
                u, v = cluster[path[0]], cluster[path[-1]]
                chains.append((u, v, np.concatenate([[nodes[u]], xy[path[1:-1]], [nodes[v]]])))
    rest = [n for n in pg if n not in cluster
            and not any(frozenset((n, m)) in seen for m in pg.neighbors(n))]
    restg = pg.subgraph(rest)
    for comp in nx.connected_components(restg):  # closed loops without junctions or ends
        comp = list(comp)
        if len(comp) < 3:
            continue
        cyc, prev = [comp[0]], None
        while len(cyc) <= len(comp):
            nxt = [n for n in restg.neighbors(cyc[-1]) if n != prev]
            if not nxt or nxt[0] == cyc[0]:
                break
            prev = cyc[-1]
            cyc.append(nxt[0])
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


def _contract_short(nodes, chains, min_len=3.0):
    """Merge nodes joined by a chain shorter than ``min_len`` px (junction artefacts): always the
    first such chain in list order, until none is left. A merge only changes the chains at its
    two nodes, so the rest are not looked at again (noise has thousands of short chains)."""
    chains = dict(enumerate(chains))
    at: dict = {}
    for i, (u, v, _) in chains.items():
        at.setdefault(u, set()).add(i)
        at.setdefault(v, set()).add(i)
    short = [i for i, (u, v, c) in chains.items() if u != v and _chain_len(c) < min_len]
    heapq.heapify(short)  # may hold chains no longer short; checked when popped
    while short:
        i = heapq.heappop(short)
        if i not in chains:
            continue
        u, v, c = chains[i]
        if u == v or _chain_len(c) >= min_len:
            continue
        nodes[u] = (np.asarray(nodes[u]) + np.asarray(nodes[v])) / 2
        del chains[i]
        at[u].discard(i)
        at[v].discard(i)
        at[u] |= at.pop(v)
        del nodes[v]
        for j in at[u]:
            a, b, cc = chains[j]
            a, b, cc = (u if a == v else a), (u if b == v else b), cc.copy()
            if a == u:
                cc[0] = nodes[u]
            if b == u:
                cc[-1] = nodes[u]
            chains[j] = (a, b, cc)
            if a != b and _chain_len(cc) < min_len:
                heapq.heappush(short, j)
    return nodes, [chains[i] for i in sorted(chains)]


def _join_degree2(nodes, chains):
    """Join chains through nodes of degree 2 (not loops), first node in chain order first; a
    joined chain goes to the end of the order."""
    chains = dict(enumerate(chains))
    at: dict = {}
    for i, (u, v, _) in chains.items():
        at.setdefault(u, set()).add(i)
        at.setdefault(v, set()).add(i)

    def rank(n):
        """Where ``n`` first appears in the chains if it can be joined, else None."""
        if len(at[n]) != 2 or any(chains[i][0] == chains[i][1] for i in at[n]):
            return None
        return min((i, 0 if chains[i][0] == n else 1) for i in at[n])

    todo = [(r, n) for r, n in ((rank(n), n) for n in at) if r is not None]
    heapq.heapify(todo)  # may hold stale ranks; checked when popped
    next_id = len(chains)
    while todo:
        r, n2 = heapq.heappop(todo)
        if n2 not in at or rank(n2) != r:
            continue
        i1, i2 = sorted(at.pop(n2))
        a1, b1, c1 = chains.pop(i1)
        a2, b2, c2 = chains.pop(i2)
        if b1 != n2:
            a1, b1, c1 = b1, a1, c1[::-1]
        if a2 != n2:
            a2, b2, c2 = b2, a2, c2[::-1]
        chains[next_id] = (a1, b2, np.concatenate([c1, c2[1:]]))
        at[a1].discard(i1)
        at[b2].discard(i2)
        at[a1].add(next_id)
        at[b2].add(next_id)
        next_id += 1
        nodes.pop(n2, None)
        for n in {a1, b2}:
            if (q := rank(n)) is not None:
                heapq.heappush(todo, (q, n))
    return nodes, list(chains.values())


# A road cut by the image border ends its skeleton about half a road width inside (3.25 px for
# 6 px roads); within this margin an end counts as "at the border": spur pruning keeps it and
# bridge_ends does not join it to another cut end (E2E test, 2026-10-29; was 3 px).
BORDER_MARGIN = 5.0


def _near_border(p, shape, margin=BORDER_MARGIN) -> bool:
    return min(p[0], p[1], shape[1] - p[0], shape[0] - p[1]) <= margin


def _prune_spurs(nodes, chains, spur, shape, passes=2):
    for _ in range(passes):
        deg = _degrees(chains)
        new = []
        for u, v, c in chains:
            ends = [n for n in (u, v) if deg.get(n, 0) == 1]
            if ends and u != v and _chain_len(c) < spur and deg[u] + deg[v] > 2 \
                    and not _near_border(nodes[ends[0]], shape):
                continue
            new.append((u, v, c))
        nodes, chains = _join_degree2(nodes, new)
    return nodes, chains


def bridge_ends(g, gap: float, shape):
    """Join mutually-nearest dangling ends (degree 1, not within ``BORDER_MARGIN`` = 5 px of
    the image border) of the ``networkx`` graph ``g`` (nodes carry ``xy``) closer than ``gap``
    px, unless a path shorter than ``3 * gap`` already joins them.

    **Modifies** ``g`` **in place** (adds edges) and returns it, for chaining; copy it first
    (``g.copy()``) to keep the original."""
    nx, _, cKDTree, _ = _deps()
    if gap <= 0:
        return g
    ends = [n for n in g if g.degree(n) == 1 and not _near_border(g.nodes[n]["xy"], shape)]
    if len(ends) < 2:
        return g
    xy = np.array([g.nodes[n]["xy"] for n in ends])
    d, k = cKDTree(xy).query(xy, k=2, distance_upper_bound=gap)

    def w(u, v, _):
        return float(np.hypot(*(g.nodes[u]["xy"] - g.nodes[v]["xy"])))

    for i in range(len(ends)):
        j = k[i, 1]
        if not np.isfinite(d[i, 1]) or j <= i or k[j, 1] != i:
            continue
        try:
            close = nx.dijkstra_path_length(g, ends[i], ends[j], weight=w) < 3 * gap
        except nx.NetworkXNoPath:
            close = False
        if not close:
            g.add_edge(ends[i], ends[j])
    return g


def graph_from_mask(prob, threshold: float = 0.5, *, spur: float = 10, gap: float = 0,
                    simplify: float = 1.0, min_length: float = 20, min_area: int = 30,
                    scale: float = 1.0) -> VectorGraph:
    """Line graph from a line-probability map ``(H, W)`` (numpy or torch; probabilities, so apply
    ``sigmoid`` to logits first).

    Args:
        threshold: Foreground threshold.
        spur: Dangling branches shorter than this (px) are removed, except at the image border.
        gap: Join mutually-nearest dangling ends closer than this (px); 0 disables. E028 tuned
            25-50 px (1 m pixels) once occlusion breaks lines.
        simplify: Douglas-Peucker tolerance (px).
        min_length: Connected pieces shorter than this (px) are dropped.
        min_area: Foreground blobs smaller than this many pixels are dropped before skeletonizing.
        scale: The head's ``scale``; coordinates are returned in input pixels.
    """
    skeletonize = _deps()[3]
    p = prob.detach().float().cpu().numpy() if hasattr(prob, "detach") else np.asarray(prob)
    if p.ndim != 2:
        raise ValueError(f"expected a (H, W) probability map, got shape {p.shape}")
    m = _remove_small(p > threshold, min_area)
    return _graph_from_skeleton(*np.nonzero(skeletonize(m)), p.shape, spur=spur, gap=gap,
                                simplify=simplify, min_length=min_length, scale=scale)


def _graph_from_skeleton(rr, cc, shape, *, spur, gap, simplify, min_length, scale) -> VectorGraph:
    g = _nx_from_skeleton(rr, cc, shape, spur=spur, simplify=simplify, min_length=min_length)
    bridge_ends(g, gap, shape)
    vg = VectorGraph.from_networkx(g)
    if scale != 1:
        vg.xy = vg.xy / scale
    return vg


def _nx_from_skeleton(rr, cc, shape, *, spur, simplify, min_length):
    """Skeleton pixels -> ``networkx`` graph (nodes carry ``xy``), before gap bridging."""
    nx, shapely = _deps()[:2]
    nodes, chains = _contract_short(*_skeleton_chains_rc(rr, cc))
    nodes, chains = _join_degree2(nodes, chains)
    nodes, chains = _prune_spurs(nodes, chains, spur, shape)
    g = nx.Graph()
    for n, q in nodes.items():
        g.add_node(n, xy=np.asarray(q, float))
    for ci, (u, v, c) in enumerate(chains):
        s = np.asarray(shapely.LineString(c).simplify(simplify).coords) if len(c) > 2 else c
        if u == v and len(s) < 4:
            continue
        s = s.copy()
        s[0], s[-1] = nodes[u], nodes[v]
        prev = u
        for k in range(1, len(s) - 1):
            g.add_node(("c", ci, k), xy=s[k])
            g.add_edge(prev, ("c", ci, k))
            prev = ("c", ci, k)
        if prev != v:
            g.add_edge(prev, v)
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    for comp in list(nx.connected_components(g)):
        sub = g.subgraph(comp)
        xy = g.nodes
        length = sum(float(np.hypot(*(xy[a]["xy"] - xy[b]["xy"]))) for a, b in sub.edges())
        if length < min_length:
            g.remove_nodes_from(comp)
    return g


def graph_from_mask_blocks(prob, threshold: float = 0.5, *, block: int = 2048, halo: int = 256,
                           spur: float = 10, gap: float = 0, simplify: float = 1.0,
                           min_length: float = 20, min_area: int = 30,
                           scale: float = 1.0) -> VectorGraph:
    """:func:`graph_from_mask` for scenes too large to skeletonize at once (``prob`` may be a
    memmap): the large-scene path for a plain line-probability head (the unified road heads
    use ``block_decoder("roads")`` instead). Each window (core + ``halo``) is thresholded and
    skeletonized; its core's skeleton pixels are kept, and one graph is built from the union.
    Equals the global result when thinning's reach (about half the widest blob) is below
    ``halo``. The halo used is at least
    ``min_area - 1`` (as ``RoadBlockDecoder.min_halo``): a blob that reaches from a core to its
    window's edge then has at least ``min_area`` pixels, so a small blob across a seam is seen
    whole (and removed) rather than kept by both windows as possibly large.
    """
    from .blocks import block_grid

    skeletonize = _deps()[3]
    from scipy import ndimage

    if hasattr(prob, "detach"):  # as graph_from_mask (Codex 10-28 #2)
        prob = prob.detach().float().cpu().numpy()
    H, W = prob.shape[-2:]
    if H == 0 or W == 0:
        return VectorGraph(np.zeros((0, 2)), np.zeros((0, 2), np.int64))
    rows, cols = [], []
    halo = max(int(halo), int(min_area) - 1)
    for b in block_grid(H, W, block, halo):
        r0, r1, c0, c1 = b.window
        w = np.asarray(prob[r0:r1, c0:c1], np.float32) > threshold
        lab, n = ndimage.label(w)  # drop small blobs only where the window shows all of them
        if n:
            size = np.bincount(lab.reshape(-1), minlength=n + 1)
            edge = np.zeros(n + 1, bool)
            for side, inner in ((lab[0], r0 > 0), (lab[-1], r0 + w.shape[0] < H),
                                (lab[:, 0], c0 > 0), (lab[:, -1], c0 + w.shape[1] < W)):
                if inner:
                    edge[side] = True
            small = (size < min_area) & ~edge
            small[0] = False
            w &= ~small[lab]
        sk = skeletonize(w)
        t, bt, lf, rt = b.inner
        r, c = np.nonzero(sk[t:bt, lf:rt])
        rows.append(r + b.core[0])
        cols.append(c + b.core[2])
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    o = np.lexsort((cc, rr))
    return _graph_from_skeleton(rr[o], cc[o], (H, W), spur=spur, gap=gap, simplify=simplify,
                                min_length=min_length, scale=scale)
