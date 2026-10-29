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
_KERNEL = 5  # keypoint peak max-filter size


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


def _arr(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, np.float32)


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
    remove_small_objects, skeletonize = _deps()[3:]
    m, kp, d = _arr(mask), _arr(keypoints), _arr(direction)
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
        c: on_mask(_refine(kp[i], _peak_pixels(kp[i], keypoint_threshold, _KERNEL)))
        for i, c in enumerate(CLASSES)
    }
    return _assemble(surf, peaks, d, (H, W), zone_radius, pair_deg, gap, scale)


def _assemble(surf, peaks, d, shape, zone_radius, pair_deg, gap, scale) -> VectorGraph:
    """Surface graph + keypoint peaks -> road graph (the steps after the dense ones)."""
    nx, shapely = _deps()[:2]
    from scipy.sparse.csgraph import minimum_spanning_tree

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
    bridge_ends(g, gap, shape)
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


# --------------------------------------------------------------------------------------
# Block-wise decoding of large scenes (contract: notes/large_scene_vectors_contract.md §2-3)
# --------------------------------------------------------------------------------------


class RoadBlockDecoder:
    """:func:`graph_from_road_heads` block by block (core + halo windows) over stitched fields.

    Usage, in memory or through ManyPred's ``postprocess_blocks``::

        dec = RoadBlockDecoder(threshold=0.5, gap=50)
        graph = dec.combine([dec(window_fields, b, scene) for b in blocks], scene)

    ``window_fields``: ``names`` -> ``(C, h, w)`` arrays cropped to ``block.window``; ``block``
    has ``core`` / ``window`` ``(top, bottom, left, right)`` and ``scene_edges``; ``scene`` has
    ``height`` / ``width``. The graph is in scene pixel coordinates.

    Per block (dense, local): threshold and small-blob removal (blobs touching an inner window
    edge are kept, as the window sees only part of them), skeletonize, keypoint local maxima
    with their sub-pixel position and on-surface test, and the direction field on the pixels
    within ``zone_radius + 9`` px of each on-surface maximum. Only the core's skeleton pixels and
    maxima are returned. ``combine`` (sparse, global): the surface graph from the union of
    skeleton pixels, peak suppression in the whole-scene order, arm rejoining (direction read
    from the sparse samples), short-component removal and gap bridging, all with the scene as
    the border. Every per-block step except skeletonize reaches at most ``zone_radius + 10`` px,
    so the result equals the whole-scene decode exactly when thinning's reach is below the halo.
    That reach is data dependent: about the half-width of a solid blob, but up to ~3x the
    half-width of a blob with holes (a car park); City-Scale road blobs are at most 11 px deep
    and halo 128 gives identical graphs. Each block also thins core + half the halo and
    ``combine`` warns if any core skeleton changed (reach above half the halo). The halo must
    be at least :attr:`min_halo`; the blocks' result size is ~24 bytes per pixel within
    ``zone_radius + 10`` px of a keypoint (direction samples) plus 16 per skeleton pixel.
    """

    def __init__(
        self,
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
        names: dict | None = None,
    ):
        self.params = dict(
            threshold=threshold,
            keypoint_threshold=keypoint_threshold,
            zone_radius=zone_radius,
            pair_deg=pair_deg,
            spur=spur,
            gap=gap,
            simplify=simplify,
            min_length=min_length,
            min_area=min_area,
            scale=scale,
        )
        self.names = {"mask": "mask", "keypoints": "keypoints", "direction": "direction"}
        self.names.update(names or {})

    def __repr__(self) -> str:
        return f"RoadBlockDecoder({self.params}, names={self.names})"

    @property
    def min_halo(self) -> int:
        """Halo the keypoint, direction and small-blob steps need: ``zone_radius + 10`` and
        ``min_area - 1`` (a blob reaching from the core to the window edge then has at least
        ``min_area`` pixels). Skeletonize needs more on wide blobs (see the class note)."""
        p = self.params
        return max(int(math.ceil(p["zone_radius"])) + 10, int(p["min_area"]) - 1)

    def __call__(self, fields, block, scene) -> dict:
        from scipy import ndimage

        skeletonize = _deps()[4]
        p, nm = self.params, self.names
        H, W = int(scene.height), int(scene.width)
        wt, wb, wl, wr = (int(v) for v in block.window)
        ct, cb, cl, cr = (int(v) for v in block.core)
        edges = tuple(bool(e) for e in block.scene_edges)
        for at_edge, margin in zip(edges, (ct - wt, wb - cb, cl - wl, wr - cr)):
            if not at_edge and margin < self.min_halo:
                raise ValueError(
                    f"block {block.index}: halo {margin} px < {self.min_halo} px "
                    "(max(zone_radius + 10, min_area - 1))"
                )
        m = _arr(fields[nm["mask"]])
        m = m[0] if m.ndim == 3 else m
        kp, d = _arr(fields[nm["keypoints"]]), _arr(fields[nm["direction"]])
        cy, cx = slice(ct - wt, cb - wt), slice(cl - wl, cr - wl)  # core in window pixels

        fg = m > p["threshold"]
        lab, n = ndimage.label(fg)  # as remove_small_objects, but only blobs seen whole
        if n and p["min_area"] > 1:
            size = np.bincount(lab.reshape(-1), minlength=n + 1)
            inner = np.zeros(n + 1, bool)
            for side, at_edge in zip((lab[0], lab[-1], lab[:, 0], lab[:, -1]), edges):
                if not at_edge:
                    inner[side] = True
            small = (size < p["min_area"]) & ~inner
            small[0] = False
            fg &= ~small[lab]
        sk = skeletonize(fg)[cy, cx]
        sr, sc = np.nonzero(sk)
        out = {"skeleton": np.stack([sr + ct, sc + cl], 1).astype(np.int64), "peaks": {}}
        # thinning's reach is data dependent: flag a core skeleton that changes with half the halo
        half = [g if e else g // 2 for e, g in zip(edges, (ct - wt, wb - cb, cl - wl, wr - cr))]
        out["unstable"] = False
        if not all(edges):
            st, sb, sl = ct - wt - half[0], cb - wt + half[1], cl - wl - half[2]
            sub = skeletonize(fg[st:sb, sl : cr - wl + half[3]])
            hy, hx = slice(half[0], half[0] + cb - ct), slice(half[2], half[2] + cr - cl)
            out["unstable"] = not np.array_equal(sub[hy, hx], sk)

        R = p["zone_radius"] + 9  # direction samples: pixel centres within R of a peak (below)
        rad = int(math.ceil(R)) + 1
        oy, ox = np.mgrid[-rad : rad + 1, -rad : rad + 1]
        disk = oy**2 + ox**2 <= (R + 1) ** 2  # around the peak's pixel: a superset
        need = np.zeros(m.shape, bool)
        for i, c in enumerate(CLASSES):
            hm = kp[i]
            keep = (hm == ndimage.maximum_filter(hm, size=_KERNEL, mode="nearest")) & (
                hm > p["keypoint_threshold"]
            )
            pix = np.argwhere(keep[cy, cx]) + (ct - wt, cl - wl)  # window pixels, raster order
            xy = _refine_window(hm, pix, (wt, wl), (H, W))
            ix = np.clip(xy[:, 0].astype(int), 0, W - 1) - wl
            iy = np.clip(xy[:, 1].astype(int), 0, H - 1) - wt
            onm = m[iy, ix] > p["threshold"]
            out["peaks"][c] = (pix + (wt, wl), hm[pix[:, 0], pix[:, 1]], xy, onm)
            for x, y in xy[onm]:  # window-sized mask: memory bounded whatever the peak count
                py, px = int(math.floor(y)) - wt, int(math.floor(x)) - wl
                y0, y1 = max(py - rad, 0), min(py + rad + 1, need.shape[0])
                x0, x1 = max(px - rad, 0), min(px + rad + 1, need.shape[1])
                need[y0:y1, x0:x1] |= disk[
                    y0 - py + rad : y1 - py + rad, x0 - px + rad : x1 - px + rad
                ]
        ky, kx = np.nonzero(need)  # raster order; the halo (>= R + 1) holds every sample
        out["direction"] = ((ky + wt).astype(np.int64) * W + kx + wl, d[:, ky, kx].T.copy())
        return out

    def combine(self, results: list, scene) -> VectorGraph:
        import warnings

        cKDTree = _deps()[2]
        p = self.params
        H, W = int(scene.height), int(scene.width)
        bad = sum(bool(x.get("unstable")) for x in results)
        if bad:
            warnings.warn(
                f"{bad} block(s): the core skeleton changed when the halo was halved (road blobs "
                "wider than the halo supports); the graph may differ from a whole-scene decode "
                "near their seams. Use a larger halo.",
                stacklevel=2,
            )
        sk = np.concatenate([x["skeleton"] for x in results] or [np.zeros((0, 2), np.int64)])
        o = np.lexsort((sk[:, 1], sk[:, 0]))  # raster order, as np.nonzero on the scene
        surf = _nx_from_skeleton(
            sk[o, 0],
            sk[o, 1],
            (H, W),
            spur=p["spur"],
            simplify=p["simplify"],
            min_length=p["min_length"],
        )
        peaks = {}
        for c in CLASSES:
            parts = [x["peaks"][c] for x in results]
            pix = np.concatenate([q[0] for q in parts] or [np.zeros((0, 2), np.int64)])
            val = np.concatenate([q[1] for q in parts] or [np.zeros(0, np.float32)])
            xy = np.concatenate([q[2] for q in parts] or [np.zeros((0, 2))])
            onm = np.concatenate([q[3] for q in parts] or [np.zeros(0, bool)])
            o = np.lexsort((pix[:, 1], pix[:, 0]))
            sel = o[_suppress(pix[o], val[o], _KERNEL, cKDTree)]  # as _peak_pixels
            peaks[c] = xy[sel[onm[sel]]].reshape(-1, 2)
        keys = np.concatenate([x["direction"][0] for x in results] or [np.zeros(0, np.int64)])
        vals = np.concatenate(
            [x["direction"][1] for x in results] or [np.zeros((0, 4), np.float32)]
        )
        keys, first = np.unique(keys, return_index=True)  # halos overlap: same pixels twice
        d = _SparseField(keys, vals[first], (vals.shape[1] if vals.ndim == 2 else 4, H, W))
        return _assemble(
            surf, peaks, d, (H, W), p["zone_radius"], p["pair_deg"], p["gap"], p["scale"]
        )


def _refine_window(hm, pix, origin, scene_hw):
    """:func:`~manyseg.vectors.graph_tensor._refine` of window pixels ``pix`` with the scene as
    the image (``origin`` = window's top-left in the scene), so it equals a whole-scene call."""
    if len(pix) == 0:
        return np.zeros((0, 2))
    H, W = scene_hw
    h, w = hm.shape
    lg = np.log(np.clip(hm, 1e-4, None))
    r, c = pix[:, 0], pix[:, 1]
    R, C = r + origin[0], c + origin[1]

    def offset(lo, mid, hi):
        den = lo - 2 * mid + hi
        ok = den < -1e-9
        return np.clip(np.where(ok, 0.5 * (lo - hi) / np.where(ok, den, -1), 0), -1, 1)

    inx, iny = (C > 0) & (C < W - 1), (R > 0) & (R < H - 1)
    dx = np.where(
        inx, offset(lg[r, np.maximum(c - 1, 0)], lg[r, c], lg[r, np.minimum(c + 1, w - 1)]), 0
    )
    dy = np.where(
        iny, offset(lg[np.maximum(r - 1, 0), c], lg[r, c], lg[np.minimum(r + 1, h - 1), c]), 0
    )
    return np.stack([C + 0.5 + dx, R + 0.5 + dy], 1)


def _suppress(pts, vals, kernel, cKDTree):
    """Indices of the peaks :func:`~manyseg.vectors.graph_tensor._peak_pixels` keeps, given its
    local maxima ``pts`` in raster order and their values: strongest first, greedy."""
    if len(pts) < 2:
        return np.arange(len(pts))
    order = np.argsort(-vals, kind="stable")
    q = pts[order]
    tree = cKDTree(q)
    alive = np.ones(len(q), bool)
    for i in range(len(q)):
        if alive[i]:
            for j in tree.query_ball_point(q[i], (kernel // 2) * 1.5):
                if j > i:
                    alive[j] = False
    return order[alive]


class _SparseField:
    """Read-only ``(C, H, W)`` field known on a set of pixels; ``f[k0:k1, y, x]`` as numpy.
    A pixel outside the set raises (the halo or sample radius was too small)."""

    def __init__(self, keys, vals, shape):
        self.keys, self.vals, self.shape = keys, vals, shape

    def __getitem__(self, idx):
        ch, y, x = idx
        k = int(y) * self.shape[2] + int(x)
        j = int(np.searchsorted(self.keys, k))
        if j >= len(self.keys) or self.keys[j] != k:
            raise KeyError(f"direction not sampled at pixel (row {y}, col {x})")
        return self.vals[j][ch]
