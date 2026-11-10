"""Training targets for the unified road heads (:func:`manyseg.vectors.road_heads`) from a road
graph. Coordinates follow
:mod:`manyseg.vectors.graph` (``(x, y)``, pixel corners at integers).
"""

from __future__ import annotations

import math

import numpy as np

from .graph import VectorGraph, _deps


def _gaussians(pts: np.ndarray, sigma: float, shape: tuple[int, int]) -> np.ndarray:
    """Max-composited Gaussians (peak 1) at float ``(x, y)`` points, cut at 3 sigma."""
    out = np.zeros(shape, np.float32)
    r = int(math.ceil(3 * sigma))
    for x, y in np.asarray(pts, np.float64).reshape(-1, 2):
        c0, r0 = int(math.floor(x)) - r, int(math.floor(y)) - r
        cs, rs = np.arange(c0, c0 + 2 * r + 1), np.arange(r0, r0 + 2 * r + 1)
        cm, rm = (cs >= 0) & (cs < shape[1]), (rs >= 0) & (rs < shape[0])
        if not cm.any() or not rm.any():
            continue
        dx = (cs[cm] + 0.5 - x)[None, :]
        dy = (rs[rm] + 0.5 - y)[:, None]
        val = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).astype(np.float32)
        sl = (slice(rs[rm][0], rs[rm][-1] + 1), slice(cs[cm][0], cs[cm][-1] + 1))
        np.maximum(out[sl], val, out=out[sl])
    return out


def _chains(g) -> dict:
    """Edge (frozenset of its nodes) -> id of the degree-2 chain it belongs to."""
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


def _as_networkx(graph, width: float, widths):
    nx = _deps()[0]
    if isinstance(graph, VectorGraph):
        w = np.broadcast_to(np.asarray(width if widths is None else widths, np.float64),
                            (len(graph.edges),))
        g = nx.Graph()
        for i, p in enumerate(graph.xy):
            g.add_node(i, xy=p)
        for (a, b), wk in zip(graph.edges, w):
            g.add_edge(int(a), int(b), width=float(wk))
        return g
    if widths is not None:
        raise ValueError("widths= is for a VectorGraph; give a networkx graph 'width' edge data")
    return graph


def road_targets(graph, shape: tuple[int, int], *, width: float = 6.0, widths=None,
                 crossings=(), line_width: float = 3.0, sigma: float = 2.0,
                 border_margin: float = 3.0, cross_deg: float = 20.0) -> dict[str, np.ndarray]:
    """Targets for :func:`road_heads` from a road graph, float32 numpy:

    * ``mask`` ``(H, W)``: the road surface, each edge buffered by half its width (pixel
      centres inside, rasterio's rule);
    * ``line`` ``(H, W)``: a ``line_width`` px band around the centrelines (touching surfaces,
      e.g. dual carriageways, keep separate centrelines);
    * ``keypoints`` ``(3, H, W)``: Gaussian heatmaps (``sigma`` px, peak 1) of junctions
      (degree >= 3), road ends (degree 1, except within ``border_margin`` px of the raster
      border, where the graph was clipped) and ``crossings`` (grade-separated crossings, e.g.
      overpasses: ``(x, y)`` points the graph does not hold, since they are not nodes);
    * ``direction`` ``(4, H, W)``: two ``(cos 2t, sin 2t)`` modes over the road surface. Mode 1
      is the direction of the first edge (in graph edge order) covering a pixel; mode 2 that of
      an edge of a *different* degree-2 chain crossing it at more than ``cross_deg`` degrees
      (zero elsewhere), so a crossing keeps both roads' directions.

    Args:
        graph: a :class:`~manyseg.vectors.VectorGraph` (road widths from ``widths``, one per
            edge, or ``width``) or a ``networkx.Graph`` with node ``xy`` and optional edge
            ``width`` data (default ``width``), in pixel coordinates.
        shape: ``(H, W)`` of the raster.
        width: road width in pixels where the graph gives none (6 px at 1 m was used on
            City-Scale).
    """
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    shapely = _deps()[1]
    H, W = shape
    g = _as_networkx(graph, width, widths)
    out = {"mask": np.zeros(shape, np.float32), "line": np.zeros(shape, np.float32),
           "keypoints": np.zeros((3, H, W), np.float32),
           "direction": np.zeros((4, H, W), np.float32)}
    cross = np.asarray(crossings, np.float64).reshape(-1, 2)
    out["keypoints"][2] = _gaussians(cross, sigma, shape)
    if g.number_of_edges() == 0:  # no roads (rasterio cannot rasterise an empty list)
        return out

    def seg(u, v):
        return shapely.LineString([g.nodes[u]["xy"], g.nodes[v]["xy"]])

    def w(data):
        return data.get("width", width)

    ident = Affine.identity()
    out["mask"] = rasterize([(seg(u, v).buffer(w(d) / 2), 1) for u, v, d in g.edges(data=True)],
                            out_shape=shape, transform=ident, dtype="uint8").astype(np.float32)
    out["line"] = rasterize([(seg(u, v).buffer(line_width / 2), 1) for u, v in g.edges()],
                            out_shape=shape, transform=ident, dtype="uint8").astype(np.float32)

    def near_border(p):
        return min(p[0], p[1], W - p[0], H - p[1]) <= border_margin

    junction = [g.nodes[n]["xy"] for n in g if g.degree(n) >= 3]
    end = [g.nodes[n]["xy"] for n in g if g.degree(n) == 1 and not near_border(g.nodes[n]["xy"])]
    out["keypoints"][0] = _gaussians(np.array(junction).reshape(-1, 2), sigma, shape)
    out["keypoints"][1] = _gaussians(np.array(end).reshape(-1, 2), sigma, shape)

    d = out["direction"]
    owner = np.full(shape, -1, np.int64)
    ang1 = np.full(shape, np.nan, np.float32)
    cid = _chains(g)
    min_dang = math.radians(cross_deg)
    for u, v, data in g.edges(data=True):
        p, q = g.nodes[u]["xy"], g.nodes[v]["xy"]
        th = math.atan2(q[1] - p[1], q[0] - p[0])
        poly = seg(u, v).buffer(w(data) / 2)
        x0, y0, x1, y1 = poly.bounds  # rasterise inside the edge's bounding box only
        c0, r0 = max(int(math.floor(x0)) - 1, 0), max(int(math.floor(y0)) - 1, 0)
        c1, r1 = min(int(math.ceil(x1)) + 1, W), min(int(math.ceil(y1)) + 1, H)
        if c0 >= c1 or r0 >= r1:
            continue
        m = rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0),
                      transform=Affine(1, 0, c0, 0, 1, r0), dtype="uint8").astype(bool)
        c = cid[frozenset((u, v))]
        ow, a1 = owner[r0:r1, c0:c1], ang1[r0:r1, c0:c1]
        first = m & (ow < 0)
        d[0, r0:r1, c0:c1][first], d[1, r0:r1, c0:c1][first] = math.cos(2 * th), math.sin(2 * th)
        ow[first], a1[first] = c, th
        dang = np.abs(((a1 - th) + math.pi / 2) % math.pi - math.pi / 2)
        second = m & (ow >= 0) & (ow != c) & (dang > min_dang)
        d[2, r0:r1, c0:c1][second], d[3, r0:r1, c0:c1][second] = math.cos(2 * th), math.sin(2 * th)
    return out
