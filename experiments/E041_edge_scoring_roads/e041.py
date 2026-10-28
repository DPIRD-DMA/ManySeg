"""E041 R5: edge-scoring road decoder on the unified heads (no skeleton tracing), oracle.

Heads: road surface mask, 3 px centreline band, junction / end / crossing keypoint heatmaps (E039)
and a frame field (E040's encoding): u = v = road direction on each road's band, u and v = the two
roads' directions where two roads of different chains overlap (unconstrained, not perpendicular).

Decoder (SAM-Road-style candidates + scoring, in our framework):
1. nodes: keypoint peaks (junction, end) + greedy maxima of the smoothed centreline band spaced
   >= ``spacing`` px; no node within ``r_cross`` px of a crossing peak (roads pass over crossings);
2. candidate edges: node pairs within 2.2 x spacing (2.6 x spacing if the segment passes within
   ``r_cross`` of a crossing peak);
3. score = mean road probability along the segment (``edge_src``: centreline band or surface) x
   frame-field agreement (max over u, v of cos^2 to the segment; 0.5 where the field is weak);
4. edges must leave every non-junction node within 25 deg of that node's own road direction(s)
   (frame field there); greedy selection by score >= thr_e with degree caps (end 1, junction 6,
   others 2) and angular suppression (no two edges at a node within 25 deg);
5. components < 20 px dropped; E028 end bridging.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import shapely
from rasterio.features import rasterize
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "E039_unified_roads"), str(HERE.parent / "E040_frame_field"),
                str(HERE.parent / "E028_vector_oracle")]
import e039 as R4  # noqa: E402
import e040 as FF  # noqa: E402

E = R4.E
CLASSES = R4.CLASSES


def road_frame(t, shape=(E.TILE, E.TILE)) -> np.ndarray:
    """(4, H, W) Re/Im c0, Re/Im c2. u = v = road direction on each band; where a second road of
    another chain (angle differs > 20 deg) overlaps, u and v are the two roads' directions."""
    g = t.graph
    cid = R4._chains(g)
    first = np.full(shape, np.nan, np.float32)
    owner = np.full(shape, -1, np.int64)
    second = np.full(shape, np.nan, np.float32)
    for u, v, data in g.edges(data=True):
        p, q = g.nodes[u]["xy"], g.nodes[v]["xy"]
        th = math.atan2(q[1] - p[1], q[0] - p[0])
        m = rasterize([(LineString([p, q]).buffer(data["width"] / 2), 1)], out_shape=shape,
                      transform=E.IDENT, dtype="uint8").astype(bool)
        c = cid[frozenset((u, v))]
        new = m & (owner < 0)
        first[new], owner[new] = th, c
        dang = np.abs(((first - th) + math.pi / 2) % math.pi - math.pi / 2)
        sec = m & (owner >= 0) & (owner != c) & (dang > math.radians(20)) & np.isnan(second)
        second[sec] = th
    on = ~np.isnan(first)
    a = np.where(on, first, 0.0)
    b = np.where(np.isnan(second), a, second)
    u2, v2 = np.exp(2j * a), np.exp(2j * b)
    c0, c2 = u2 * v2, -(u2 + v2)
    out = np.stack([c0.real, c0.imag, c2.real, c2.imag]).astype(np.float32)
    out[:, ~on] = 0
    return out


def encode_r5(t):
    """R5 heads: road surface, centreline band, 3 keypoint classes, frame field (4 ch). R4's
    two-mode direction head is replaced by the frame field (Codex E041 #9)."""
    tg = R4.encode_r4(t)
    tg.arrays.pop("direction")
    tg.arrays["frame"] = road_frame(t)
    return tg


def corrupt_r5(tg, t, cfg: dict, seed: int) -> dict:
    out = E.corrupt(tg, t, cfg, seed)
    f = tg.arrays["frame"].copy()
    if cfg["occ"] > 0:
        f[:, E._disk_mask(E.occlusion_disks(t, cfg["occ"], seed))] = 0
    if cfg["vec"] > 0:
        f = f + E._rng(seed, t, "frame").normal(0, 0.1 * cfg["vec"], f.shape).astype(np.float32)
    out["frame"] = f
    return out


def band_maxima(line: np.ndarray, thr: float, spacing: float) -> np.ndarray:
    """Greedy maxima of the (sigma 1) smoothed centreline band, at least ``spacing`` px apart; (x, y)."""
    sm = ndimage.gaussian_filter(line, 1.0)
    rr, cc = np.nonzero(sm > thr)
    if len(rr) == 0:
        return np.zeros((0, 2))
    order = np.argsort(-sm[rr, cc])
    free = np.ones(sm.shape, bool)
    H, W = sm.shape
    s = int(math.ceil(spacing))
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    disk = xx ** 2 + yy ** 2 < spacing ** 2
    pts = []
    for k in order:
        r, c = rr[k], cc[k]
        if not free[r, c]:
            continue
        pts.append((c + 0.5, r + 0.5))
        r0, r1, c0, c1 = max(r - s, 0), min(r + s + 1, H), max(c - s, 0), min(c + s + 1, W)
        free[r0:r1, c0:c1] &= ~disk[(r0 - r + s):(r1 - r + s), (c0 - c + s):(c1 - c + s)]
    return np.array(pts)


def _sample(arr2d, a, b, n):
    t = np.linspace(0, 1, n)
    xy = a[None] * (1 - t[:, None]) + b[None] * t[:, None]
    H, W = arr2d.shape[-2:]
    c = np.clip(xy[:, 0].astype(int), 0, W - 1)
    r = np.clip(xy[:, 1].astype(int), 0, H - 1)
    return r, c


def edge_score(a, b, prob, frame) -> float:
    n = max(int(np.hypot(*(b - a))), 3)
    r, c = _sample(prob, a, b, n)
    p = float(prob[r, c].mean())
    tu, tv, s = FF.frame_dirs(frame, r, c)
    th = math.atan2(b[1] - a[1], b[0] - a[0])
    agree = np.maximum(np.cos(th - tu) ** 2, np.cos(th - tv) ** 2)
    agree = np.where(s >= 0.2, agree, 0.5)
    return p * float(agree.mean())


def decode_r5(a: dict, p: dict) -> nx.Graph:
    H, W = a["mask"].shape

    def on_road(q):
        if not len(q):
            return q
        c = np.clip(q[:, 0].astype(int), 0, W - 1); r = np.clip(q[:, 1].astype(int), 0, H - 1)
        return q[a["mask"][r, c] > 0.3]

    pk = {c: on_road(E._peaks(a[c], p["thr_k"], 5)) for c in CLASSES}
    # a junction/end peak within r_cross of a crossing peak: the stronger heatmap value wins
    # (Codex E041 #7: a junction node at a crossing would join the two roads)
    if len(pk["crossing"]):
        def val(c, q):
            return a[c][np.clip(q[:, 1].astype(int), 0, H - 1), np.clip(q[:, 0].astype(int), 0, W - 1)]
        keep_x = np.ones(len(pk["crossing"]), bool)
        ctree = cKDTree(pk["crossing"])
        for c in ("junction", "end"):
            if not len(pk[c]):
                continue
            d, j = ctree.query(pk[c])
            near = d <= p["r_cross"]
            stronger_x = val("crossing", pk["crossing"][j]) >= val(c, pk[c])
            pk[c] = pk[c][~(near & stronger_x)]
            keep_x[j[near & ~stronger_x]] = False
        pk["crossing"] = pk["crossing"][keep_x]
    band = band_maxima(a["line"], p["thr_n"], p["spacing"])
    keyn = np.concatenate([pk["junction"], pk["end"]]) if len(pk["junction"]) or len(pk["end"]) else np.zeros((0, 2))
    if len(band) and len(keyn):
        band = band[cKDTree(keyn).query(band)[0] > 6.0]
    if len(band) and len(pk["crossing"]):
        band = band[cKDTree(pk["crossing"]).query(band)[0] > p["r_cross"]]
    nodes = np.concatenate([x for x in (pk["junction"], pk["end"], band) if len(x)]) if (len(keyn) or len(band)) else np.zeros((0, 2))
    kind = np.array(["junction"] * len(pk["junction"]) + ["end"] * len(pk["end"]) + ["band"] * len(band))
    if len(nodes) < 2:
        return nx.Graph()
    cap = {"junction": 6, "end": 1, "band": 2}
    # each non-junction node's own road direction(s) from the frame field (strongest 3x3 pixel);
    # an edge must leave such a node within ``end_deg`` of one of them (smoke-tile fix: zig-zags
    # between touching lanes and wrong pairings across crossings passed the along-segment average)
    fr = a["frame"]
    st = np.hypot(fr[0], fr[1])
    ndir = []
    for (x, y), kd in zip(nodes, kind):
        r, c = int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, H), max(c - 1, 0), min(c + 2, W)
        k = int(np.argmax(st[r0:r1, c0:c1]))
        rr, cc = r0 + k // (c1 - c0), c0 + k % (c1 - c0)
        if kd == "junction" or st[rr, cc] < 0.2:
            ndir.append(None)
        else:
            tu, tv, _ = FF.frame_dirs(fr, np.array([rr]), np.array([cc]))
            ndir.append((float(tu[0]), float(tv[0])))
    cos_end = math.cos(math.radians(p.get("end_deg", 25.0)))

    def leaves_ok(k, th):
        if ndir[k] is None:
            return True
        return max(abs(math.cos(th - ndir[k][0])), abs(math.cos(th - ndir[k][1]))) >= cos_end
    prob = a["line"] if p["edge_src"] == "line" else a["mask"]
    tree = cKDTree(nodes)
    cross_tree = cKDTree(pk["crossing"]) if len(pk["crossing"]) else None
    cand = []
    for i, j in tree.query_pairs(2.6 * p["spacing"]):
        d = float(np.hypot(*(nodes[j] - nodes[i])))
        key = kind[i] != "band" or kind[j] != "band"
        across = False
        if cross_tree is not None:  # point-to-segment distance (Codex E041 #6)
            seg = LineString([nodes[i], nodes[j]])
            across = min(seg.distance(shapely.Point(q)) for q in pk["crossing"]) <= p["r_cross"]
        if d > 2.2 * p["spacing"] and not across:  # greedy maxima can sit up to ~2x spacing apart
            continue
        th = math.atan2(nodes[j][1] - nodes[i][1], nodes[j][0] - nodes[i][0])
        if not (leaves_ok(i, th) and leaves_ok(j, th)):
            continue
        sc = edge_score(nodes[i], nodes[j], prob, a["frame"])
        if sc >= p["thr_e"]:
            cand.append((across, key, sc, i, j))
    # across-crossing edges first (Codex E041 #8), then edges at keypoints (so a junction claims its
    # arms before band nodes link past it; smoke-tile fix), then by score
    cand.sort(reverse=True)
    cand = [(sc, i, j) for _, _, sc, i, j in cand]
    g = nx.Graph()
    for k, q in enumerate(nodes):
        g.add_node(k, xy=np.asarray(q, float))
    dirs: dict = {k: [] for k in range(len(nodes))}
    cos_sup = math.cos(math.radians(25))
    for sc, i, j in cand:
        if g.degree(i) >= cap[kind[i]] or g.degree(j) >= cap[kind[j]]:
            continue
        v = nodes[j] - nodes[i]
        v = v / np.hypot(*v)
        if any(float(v @ w) > cos_sup for w in dirs[i]) or any(float(-v @ w) > cos_sup for w in dirs[j]):
            continue
        g.add_edge(i, j)
        dirs[i].append(v)
        dirs[j].append(-v)
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    for comp in list(nx.connected_components(g)):
        if E.graph_length(g.subgraph(comp)) < 20:
            g.remove_nodes_from(comp)
    return E.bridge_ends(g, p["gap"])
