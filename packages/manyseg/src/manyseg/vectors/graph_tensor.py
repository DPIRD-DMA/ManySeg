"""Sat2Graph-style graph tensor for polygons (experimental; E028 oracle viable, E030 pending).

Dense targets that encode a planar graph: around each vertex (within ``radius`` px) a vertex
score, and for each of ``sectors`` angular sectors an edge probability plus the vector from the
pixel centre to the neighbouring vertex in that sector. Polygons are graph faces: the decoder
finds vertices (peaks), links them through the predicted vectors, joins near dangling ends,
polygonizes the edges together with the image border, and keeps faces that are mostly inside
the ``extent`` mask.

E028 (oracle, OSM footprints at 0.5 m) found it decodes footprints with near-exact corners and
degrades all-or-nothing, and that it is fragile for road networks (one missed vertex breaks two
edges), so it is offered for polygons only. Coordinates follow :mod:`manyseg.vectors.graph`.
"""

from __future__ import annotations

import math

import numpy as np

from .graph import VectorGraph, _deps

SECTORS = 6


def graph_tensor_heads(sectors: int = SECTORS) -> dict[str, dict]:
    """Head configs for ``create_model(heads=...)``: ``vertex`` (1, regression on a Gaussian
    target), ``edge`` (``sectors``, per-sector logits), ``vector`` (``2 * sectors``: dx, dy per
    sector, in pixels / ``spacing``), ``extent`` (1, logits)."""
    return {"vertex": {"type": "regression", "out_channels": 1},
            "edge": {"type": "segmentation", "out_channels": sectors},
            "vector": {"type": "regression", "out_channels": 2 * sectors},
            "extent": {"type": "segmentation", "out_channels": 1}}


def _densify(coords: np.ndarray, spacing: float) -> np.ndarray:
    out = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / spacing)))
        out.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(out)


def vertex_graph(polygons, spacing: float = 20.0, *, merge: float = 1.0, border=None) -> VectorGraph:
    """Polygon rings as one planar graph: rings simplified (Douglas-Peucker, 1 px), densified to
    at most ``spacing`` px per edge, and vertices closer than ``merge`` px merged (adjacent
    buildings share corners). With ``border=(H, W)``, ring segments lying on the image border are
    left out (the decoder adds the border back)."""
    shapely = _deps()[1]
    cKDTree = _deps()[2]
    pts, edges = [], []
    for p in polygons:
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(shapely.Polygon(ring).exterior.simplify(1.0).coords)
            if len(c) < 4:
                continue
            d = _densify(c, spacing)
            base = len(pts)
            pts.extend(d)
            for k in range(len(d) - 1):
                a, b = d[k], d[k + 1]
                if border is not None and _on_border(a, b, border):
                    continue
                edges.append((base + k, base + k + 1))
    if not pts:
        return VectorGraph(np.zeros((0, 2)), np.zeros((0, 2), int))
    pts = np.array(pts)
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(pts).query_pairs(merge):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(pts))])
    uniq, inv = np.unique(roots, return_inverse=True)
    used = np.zeros(len(uniq), bool)
    e = {tuple(sorted((inv[a], inv[b]))) for a, b in edges if inv[a] != inv[b]}
    for a, b in e:
        used[a] = used[b] = True
    # keep vertices on any edge; renumber
    keep = np.nonzero(used)[0]
    remap = -np.ones(len(uniq), int)
    remap[keep] = np.arange(len(keep))
    return VectorGraph(pts[uniq][keep], np.array(sorted((remap[a], remap[b]) for a, b in e)).reshape(-1, 2))


def _on_border(a, b, shape, eps=1e-6) -> bool:
    H, W = shape
    for k, lim in ((0, W), (1, H)):
        for v in (0.0, float(lim)):
            if abs(a[k] - v) < eps and abs(b[k] - v) < eps:
                return True
    return False


def encode_graph_tensor(vg: VectorGraph, shape: tuple[int, int], *, sectors: int = SECTORS,
                        radius: float = 3.0, sigma: float = 1.5) -> dict[str, np.ndarray]:
    """Dense targets ``vertex`` (H, W), ``edge`` (S, H, W) in {0, 1}, ``vector`` (S, 2, H, W)
    in pixels and ``support`` (H, W) bool (pixels within ``radius`` of a vertex). Each pixel
    belongs to its nearest vertex; a neighbour goes to its own sector, else the adjacent sector
    on the nearer side, else the other adjacent one (else it is not represented)."""
    H, W = shape
    V = len(vg.xy)
    nb = np.zeros((V + 1, sectors, 2), np.float32)
    has = np.zeros((V + 1, sectors), bool)
    width = 2 * math.pi / sectors
    adj: list[list[int]] = [[] for _ in range(V)]
    for a, b in vg.edges:
        adj[a].append(b)
        adj[b].append(a)
    for v in range(V):
        for u in adj[v]:
            d = vg.xy[u] - vg.xy[v]
            ang = math.atan2(d[1], d[0]) % (2 * math.pi)
            own = int(ang // width) % sectors
            side = 1 if ang / width - own >= 0.5 else -1
            for k in (own, (own + side) % sectors, (own - side) % sectors):
                if not has[v, k]:
                    has[v, k], nb[v, k] = True, vg.xy[u]
                    break
    best = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), V, np.int64)
    r = int(math.ceil(radius))
    for v, (x, y) in enumerate(vg.xy):
        c0, r0 = int(math.floor(x)) - r, int(math.floor(y)) - r
        c1, r1 = c0 + 2 * r + 1, r0 + 2 * r + 1
        cs, rs = max(c0, 0), max(r0, 0)
        ce, re = min(c1, W), min(r1, H)
        if cs >= ce or rs >= re:
            continue
        yy, xx = np.mgrid[rs:re, cs:ce]
        d2 = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2
        win = best[rs:re, cs:ce]
        upd = (d2 <= radius ** 2) & (d2 < win)
        win[upd] = d2[upd]
        owner[rs:re, cs:ce][upd] = v
    support = owner < V
    vertex = np.where(support, np.exp(-best / (2 * sigma ** 2)), 0).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pix = np.stack([xx + 0.5, yy + 0.5], -1).astype(np.float32)  # (H, W, 2)
    edge = has[owner].transpose(2, 0, 1).astype(np.float32)  # (S, H, W)
    vec = (nb[owner] - pix[:, :, None, :]) * has[owner][..., None]  # (H, W, S, 2)
    return {"vertex": vertex, "edge": edge, "vector": vec.transpose(2, 3, 0, 1).astype(np.float32),
            "support": support}


def _peak_pixels(hm: np.ndarray, threshold: float, kernel: int) -> np.ndarray:
    """(row, col) local maxima above ``threshold`` (max filter), then greedy suppression within
    the kernel radius, strongest first (a plateau otherwise gives several tied peaks)."""
    from scipy import ndimage

    cKDTree = _deps()[2]
    keep = (hm == ndimage.maximum_filter(hm, size=kernel, mode="nearest")) & (hm > threshold)
    pts = np.argwhere(keep)
    if len(pts) < 2:
        return pts
    pts = pts[np.argsort(-hm[keep], kind="stable")]
    tree = cKDTree(pts)
    alive = np.ones(len(pts), bool)
    for i in range(len(pts)):
        if alive[i]:
            for j in tree.query_ball_point(pts[i], (kernel // 2) * 1.5):
                if j > i:
                    alive[j] = False
    return pts[alive]


def _refine(hm: np.ndarray, pix: np.ndarray) -> np.ndarray:
    """Sub-pixel ``(x, y)``: per-axis parabola through the log heatmap (exact for a Gaussian),
    clipped to +-1 px; no offset on an axis whose neighbour is outside the image."""
    if len(pix) == 0:
        return np.zeros((0, 2))
    H, W = hm.shape
    lg = np.log(np.clip(hm, 1e-4, None))
    r, c = pix[:, 0], pix[:, 1]

    def offset(lo, mid, hi):
        den = lo - 2 * mid + hi
        ok = den < -1e-9
        return np.clip(np.where(ok, 0.5 * (lo - hi) / np.where(ok, den, -1), 0), -1, 1)

    inx, iny = (c > 0) & (c < W - 1), (r > 0) & (r < H - 1)
    dx = np.where(inx, offset(lg[r, np.maximum(c - 1, 0)], lg[r, c], lg[r, np.minimum(c + 1, W - 1)]), 0)
    dy = np.where(iny, offset(lg[np.maximum(r - 1, 0), c], lg[r, c], lg[np.minimum(r + 1, H - 1), c]), 0)
    return np.stack([c + 0.5 + dx, r + 0.5 + dy], 1)


def graph_from_tensor(vertex, edge, vector, *, vertex_threshold: float = 0.2,
                      edge_threshold: float = 0.5, link_radius: float = 10.0, gap: float = 25.0,
                      kernel: int = 3, snap_border: bool = True,
                      max_vertices: int | None = 5000,
                      max_edge_length: float | None = 60.0) -> VectorGraph:
    """Decode a graph from ``vertex`` (H, W), ``edge`` (S, H, W) probabilities and ``vector``
    (S, 2, H, W) in pixels (numpy or torch). Defaults are E028's level-2 tuning for footprints.

    ``max_vertices`` keeps only the strongest peaks: on a noisy vertex map (an untrained model,
    a low threshold) the number of peaks, and the cost of polygonizing the edges, is otherwise
    unbounded (E030: one 512² tile took > 10 GB). A 512² WHU tile has about 2,000 GT vertices
    at most. ``max_edge_length`` (px) drops longer edges: true edges are at most the encoding's
    vertex spacing (20 px) long, and long edges from noisy vectors cross each other quadratically
    when polygonized.
    """
    nx, _, cKDTree, _, _ = _deps()
    from .graph import bridge_ends

    vertex, edge, vector = (a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                            for a in (vertex, edge, vector))
    H, W = vertex.shape
    pi = _peak_pixels(vertex, vertex_threshold, kernel)  # strongest first
    if max_vertices is not None:
        pi = pi[:max_vertices]
    g = nx.Graph()
    if len(pi) == 0:
        return VectorGraph(np.zeros((0, 2)), np.zeros((0, 2), int))
    centre = np.stack([pi[:, 1] + 0.5, pi[:, 0] + 0.5], 1)  # vectors are relative to these
    xy = _refine(vertex, pi)
    if snap_border:
        xy[:, 0] = np.where(xy[:, 0] < 1.5, 0.0, np.where(xy[:, 0] > W - 1.5, float(W), xy[:, 0]))
        xy[:, 1] = np.where(xy[:, 1] < 1.5, 0.0, np.where(xy[:, 1] > H - 1.5, float(H), xy[:, 1]))
    for i, q in enumerate(xy):
        g.add_node(i, xy=q)
    tree = cKDTree(xy)
    for i, (r, c) in enumerate(pi):
        for k in range(edge.shape[0]):
            if edge[k, r, c] <= edge_threshold:
                continue
            d, j = tree.query(centre[i] + vector[k, :, r, c], k=2, distance_upper_bound=link_radius)
            for dd, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
                if np.isfinite(dd) and jj != i:
                    if max_edge_length is None or np.hypot(*(xy[jj] - xy[i])) <= max_edge_length:
                        g.add_edge(i, int(jj))
                    break
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    bridge_ends(g, gap, (H, W))
    return VectorGraph.from_networkx(g)


def polygons_from_tensor(vertex, edge, vector, extent, *, extent_threshold: float = 0.5,
                         face_threshold: float = 0.5, simplify: float = 1.5, min_area: float = 16,
                         **graph_kwargs) -> list:
    """Footprint polygons: :func:`graph_from_tensor`, then faces of the graph plus the image
    border that are more than ``face_threshold`` inside ``extent > extent_threshold``
    (probabilities), simplified (px)."""
    _, shapely, _, _, _ = _deps()
    from shapely.ops import polygonize, unary_union

    ext = extent.detach().float().cpu().numpy() if hasattr(extent, "detach") else np.asarray(extent)
    H, W = ext.shape
    g = graph_from_tensor(vertex, edge, vector, **graph_kwargs)
    lines = [shapely.LineString([g.xy[a], g.xy[b]]) for a, b in g.edges]
    corners = [(0, 0), (W, 0), (W, H), (0, H), (0, 0)]
    lines += [shapely.LineString([a, b]) for a, b in zip(corners[:-1], corners[1:])]
    faces = list(polygonize(unary_union(lines)))
    if not faces:
        return []
    inside = (ext > extent_threshold).astype(np.float64)
    out = []
    for f in faces:
        if f.area < min_area:
            continue
        frac = _face_fraction(f, inside)
        if frac > face_threshold:
            q = f.simplify(simplify)
            if q.geom_type == "Polygon" and q.is_valid and q.area >= min_area:
                out.append(shapely.Polygon(q.exterior))
    return out


def _face_fraction(face, inside: np.ndarray) -> float:
    """Fraction of the face's pixels (centres inside the face) that are inside the mask."""
    import shapely

    x0, y0, x1, y1 = face.bounds
    c0, r0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    c1, r1 = min(int(math.ceil(x1)), inside.shape[1]), min(int(math.ceil(y1)), inside.shape[0])
    if c0 >= c1 or r0 >= r1:
        return 0.0
    yy, xx = np.mgrid[r0:r1, c0:c1]
    m = shapely.contains_xy(face, xx.ravel() + 0.5, yy.ravel() + 0.5).reshape(xx.shape)
    n = int(m.sum())
    return float(inside[r0:r1, c0:c1][m].sum() / n) if n else 0.0
