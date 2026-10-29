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

from ..utils._peaks import peak_pixels, refine
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


def _nearest_within(xy: np.ndarray, shape: tuple[int, int], radius: float):
    """Pixels within ``radius`` of any point of ``xy`` ``(V, 2)``: their rows, cols, nearest point
    index and float32 squared distance (from the pixel centre). All points at once; ties go to
    the lower point index, as in a loop over points with a strict "closer" test."""
    H, W = shape
    r = int(math.ceil(radius))
    off = np.arange(-r, r + 1)
    x, y = xy[:, 0], xy[:, 1]
    cc = np.floor(x).astype(np.int64)[:, None, None] + off[None, None, :]
    rr = np.floor(y).astype(np.int64)[:, None, None] + off[None, :, None]
    cc, rr = np.broadcast_arrays(cc, rr)
    d2 = (cc + 0.5 - x[:, None, None]) ** 2 + (rr + 0.5 - y[:, None, None]) ** 2
    vi = np.broadcast_to(np.arange(len(xy))[:, None, None], d2.shape)
    ok = (d2 <= radius ** 2) & (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
    cc, rr, d2, vi = cc[ok], rr[ok], d2[ok], vi[ok]
    pix = rr * W + cc
    order = np.lexsort((vi, d2.astype(np.float32), pix))  # by pixel, then distance, then point
    pix, d2, vi, cc, rr = pix[order], d2[order], vi[order], cc[order], rr[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    return rr[first], cc[first], vi[first], d2[first].astype(np.float32)


def offset_field(points: np.ndarray, shape: tuple[int, int], radius: float = 8.0):
    """Offset targets towards point geometry (vertices, corners, object centres): for each pixel
    within ``radius`` of a point, the vector ``(dx, dy)`` in pixels from the pixel centre to the
    nearest point. Returns ``field`` ``(2, H, W)`` float32 and ``mask`` ``(H, W)`` bool. Train a
    2-channel regression head on it (masked L1) and move detected points with
    :func:`snap_points`: a prediction a few pixels off then lands on the true location."""
    H, W = shape
    field = np.zeros((2, H, W), np.float32)
    mask = np.zeros((H, W), bool)
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return field, mask
    rr, cc, vi, _ = _nearest_within(pts, shape, radius)
    mask[rr, cc] = True
    field[0, rr, cc] = pts[vi, 0] - (cc + 0.5)
    field[1, rr, cc] = pts[vi, 1] - (rr + 0.5)
    return field, mask


def offset_field_from_mask(geometry: np.ndarray, radius: float = 8.0):
    """As :func:`offset_field` for raster geometry (line, boundary or point pixels, bool
    ``(H, W)``): vectors to the centre of the nearest geometry pixel, within ``radius``."""
    from scipy import ndimage

    g = np.asarray(geometry, bool)
    field = np.zeros((2,) + g.shape, np.float32)
    if not g.any():
        return field, np.zeros(g.shape, bool)
    d, (ir, ic) = ndimage.distance_transform_edt(~g, return_indices=True)
    rows, cols = np.indices(g.shape)
    mask = d <= radius
    field[0] = np.where(mask, ic - cols, 0)
    field[1] = np.where(mask, ir - rows, 0)
    return field, mask


def snap_points(points: np.ndarray, field) -> np.ndarray:
    """Move points ``(K, 2)`` ``(x, y)`` by a predicted offset field ``(2, H, W)`` (pixels), read
    at each point's pixel; returns the snapped ``(K, 2)``."""
    f = field.detach().float().cpu().numpy() if hasattr(field, "detach") else np.asarray(field)
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    c = np.clip(np.floor(pts[:, 0]).astype(int), 0, f.shape[2] - 1)
    r = np.clip(np.floor(pts[:, 1]).astype(int), 0, f.shape[1] - 1)
    return np.stack([c + 0.5 + f[0, r, c], r + 0.5 + f[1, r, c]], 1)


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
    vertex = np.zeros((H, W), np.float32)
    edge = np.zeros((sectors, H, W), np.float32)
    vec = np.zeros((sectors, 2, H, W), np.float32)
    support = np.zeros((H, W), bool)
    if V == 0:
        return {"vertex": vertex, "edge": edge, "vector": vec, "support": support}
    rr, cc, vi, best = _nearest_within(vg.xy, (H, W), radius)
    support[rr, cc] = True
    vertex[rr, cc] = np.exp(-best / (2 * sigma ** 2))
    h = has[vi]  # (P, S)
    edge[:, rr, cc] = h.T
    pxy = np.stack([cc + 0.5, rr + 0.5], -1).astype(np.float32)  # (P, 2)
    vec[:, :, rr, cc] = ((nb[vi] - pxy[:, None, :]) * h[..., None]).transpose(1, 2, 0)
    return {"vertex": vertex, "edge": edge, "vector": vec, "support": support}


def graph_from_tensor(vertex, edge, vector, *, vertex_threshold: float = 0.2,
                      edge_threshold: float = 0.5, link_radius: float = 10.0, gap: float = 25.0,
                      kernel: int = 3, snap_border: bool = True,
                      max_vertices: int | None = 5000,
                      max_edge_length: float | None = 60.0, snap=None) -> VectorGraph:
    """Decode a graph from ``vertex`` (H, W), ``edge`` (S, H, W) probabilities and ``vector``
    (S, 2, H, W) in pixels (numpy or torch). Defaults are E028's level-2 tuning for footprints.

    ``max_vertices`` keeps only the strongest peaks: on a noisy vertex map (an untrained model,
    a low threshold) the number of peaks, and the cost of polygonizing the edges, is otherwise
    unbounded (E030: one 512² tile took > 10 GB). A 512² WHU tile has about 2,000 GT vertices
    at most. ``max_edge_length`` (px) drops longer edges: true edges are at most the encoding's
    vertex spacing (20 px) long, and long edges from noisy vectors cross each other quadratically
    when polygonized. ``snap`` (optional ``(2, H, W)`` predicted offset field, pixels, see
    :func:`offset_field`) places each vertex at its peak pixel's centre plus the predicted
    offset, instead of the sub-pixel parabola fit.
    """
    nx, _, cKDTree, _, _ = _deps()
    from .graph import bridge_ends

    vertex, edge, vector = (a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                            for a in (vertex, edge, vector))
    H, W = vertex.shape
    pi = peak_pixels(vertex, vertex_threshold, kernel)  # strongest first
    if max_vertices is not None:
        pi = pi[:max_vertices]
    g = nx.Graph()
    if len(pi) == 0:
        return VectorGraph(np.zeros((0, 2)), np.zeros((0, 2), int))
    centre = np.stack([pi[:, 1] + 0.5, pi[:, 0] + 0.5], 1)  # vectors are relative to these
    xy = refine(vertex, pi) if snap is None else snap_points(centre, snap)  # learned offsets win
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
