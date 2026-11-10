"""Road graphs from the unified road heads (the default road decoder).

Heads: road surface probability, keypoint heatmaps (junction, end, crossing) and a two-mode
``(cos 2t, sin 2t)`` direction field. Decoding: the surface skeleton graph (as
:func:`~manyseg.vectors.graph_from_mask`) is cut out of a disk around every keypoint; inside a
zone the arms that reach it are rejoined by the keypoint's class: to the junction (several
junctions in one zone are linked by a minimum spanning tree), paired straight through a
crossing (most opposite headings first, so an overpass stays two unconnected roads), or to
the road end. On City-Scale (three seeds) this kept ~55% of grade-separated crossings
against ~32% for the plain skeleton, with no loss of APLS or junction F1.
"""

from __future__ import annotations

import math

import numpy as np

from ..utils._peaks import maxima_mask, peak_pixels, refine, suppress
from .graph import VectorGraph, _deps, _nx_from_skeleton, _remove_small, bridge_ends

CLASSES = ("junction", "end", "crossing")
_KERNEL = 5  # keypoint peak max-filter size
# Strongest junctions used per zone. Far above a real one; on noise, zones chain into one holding
# thousands and the spanning tree's dense distance matrix would grow with their square.
_MAX_ZONE_JUNCTIONS = 256


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


def _sigmoid(a: np.ndarray) -> np.ndarray:
    """float32 sigmoid, as ``torch.sigmoid`` on a float32 tensor."""
    import torch

    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(a, np.float32))).numpy()


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
    logits: bool = False,
) -> VectorGraph:
    """Road graph from the unified road heads (numpy or torch).

    Args:
        mask (np.ndarray | torch.Tensor): ``(H, W)`` road-surface probability (logit with
            ``logits=True``).
        keypoints (np.ndarray | torch.Tensor): ``(3, H, W)`` junction, end and crossing
            heatmaps (probabilities; logits with ``logits=True``).
        direction (np.ndarray | torch.Tensor): ``(4, H, W)`` two ``(cos 2t, sin 2t)`` modes (the
            second where two road chains overlap at an angle: crossings and junctions, as
            :func:`road_targets` sets it).
        threshold: Road-surface threshold, as :func:`graph_from_mask`.
        spur: Dangling branches shorter than this (px) are removed, as :func:`graph_from_mask`.
        gap: Mutually-nearest dangling ends closer than this (px) are joined (0 disables), as
            :func:`graph_from_mask` (gap bridging runs once, on the final graph).
        simplify: Douglas-Peucker tolerance (px) of the edges, as :func:`graph_from_mask`.
        min_length: Connected pieces shorter than this (px) are dropped, as
            :func:`graph_from_mask`.
        min_area: Surface blobs smaller than this many pixels are dropped first, as
            :func:`graph_from_mask`.
        keypoint_threshold: Heatmap peak threshold; peaks off the road surface are ignored.
        zone_radius: Radius (px) of the disk around each keypoint where arms are rejoined.
        pair_deg: Arms through a crossing are paired if their headings are at least this far apart.
        scale: The heads' ``scale``; coordinates are returned in input pixels.
        logits: ``mask`` and ``keypoints`` are logits: apply a sigmoid to both first
            (``direction`` is a value and is never transformed).

    A low-level building block for one tile or scene in memory; the recommended entry point is
    ``decode_scene(fields, block_decoder("roads"))``, which gives the same graph whole-scene and
    block by block. The defaults here are the same as ``block_decoder("roads")``: tuned on
    City-Scale (1 m pixels).
    """
    skeletonize = _deps()[3]
    m, kp, d = _arr(mask), _arr(keypoints), _arr(direction)
    if logits:
        m, kp = _sigmoid(m), _sigmoid(kp)
    H, W = m.shape
    fg = _remove_small(m > threshold, min_area)
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
        c: on_mask(refine(kp[i], peak_pixels(kp[i], keypoint_threshold, _KERNEL)))
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

    disks = shapely.buffer(shapely.points(allk), zr, quad_segs=16)  # as Point.buffer
    zone = shapely.union_all(disks)
    zones = list(getattr(zone, "geoms", [zone]))
    ztree = shapely.STRtree(zones)
    dtree = shapely.STRtree(disks)

    def clip(ln):  # against the disks it meets: dense peaks merge into one huge zone polygon
        hit = dtree.query(ln, predicate="intersects")
        if len(hit) == 0:
            return ln
        if shapely.covers(disks[hit], ln).any():
            return shapely.LineString()
        return ln.difference(shapely.union_all(disks[hit]))

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
            J = np.array(inzone["junction"][i][:_MAX_ZONE_JUNCTIONS])
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
# Block-wise decoding of large scenes (the block contract of manyseg.vectors.blocks)
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
    so the result equals the whole-scene decode exactly when the halo is at least
    :attr:`min_halo` **and** covers thinning's reach.
    That reach is data dependent: about the half-width of a solid blob, but up to ~3x the
    half-width of a blob with holes (a car park); City-Scale road blobs are at most 11 px deep
    and halo 128 gives identical graphs. Each block also thins core + half the halo and
    ``combine`` warns if any core skeleton changed (reach above half the halo). The halo must
    be at least :attr:`min_halo` (enforced); the blocks' result size is ~24 bytes per pixel
    within ``zone_radius + 10`` px of a keypoint (direction samples) plus 16 per skeleton pixel.

    ``logits=True``: the ``mask`` and ``keypoints`` fields are logits (a sigmoid is applied in
    each block, as :func:`graph_from_road_heads` ``logits=True``); ``direction`` is a value.
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
        logits: bool = False,
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
            logits=bool(logits),
        )
        self.names = {"mask": "mask", "keypoints": "keypoints", "direction": "direction"}
        self.names.update(names or {})

    def __repr__(self) -> str:
        return f"RoadBlockDecoder({self.params}, names={self.names})"

    @property
    def min_halo(self) -> int:
        """Halo the keypoint, direction and small-blob steps need: ``zone_radius + 10`` and
        ``min_area - 1`` (a blob reaching from the core to the window edge then has at least
        ``min_area`` pixels). Necessary, not sufficient: exactness also needs the halo to cover
        the skeleton's data-dependent reach (about the half-width of the widest road blob, up to
        ~3x for blobs with holes); ``combine`` warns when it did not (see the class note)."""
        p = self.params
        return max(int(math.ceil(p["zone_radius"])) + 10, int(p["min_area"]) - 1)

    def __call__(self, fields, block, scene) -> dict:
        """Dense steps on one block's window: the core's skeleton pixels, keypoint maxima and
        the direction samples around them (see the class note)."""
        from scipy import ndimage

        skeletonize = _deps()[3]
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
        if p["logits"]:
            m, kp = _sigmoid(m), _sigmoid(kp)
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
            keep = maxima_mask(hm, p["keypoint_threshold"], _KERNEL)
            pix = np.argwhere(keep[cy, cx]) + (ct - wt, cl - wl)  # window pixels, raster order
            xy = refine(hm, pix, origin=(wt, wl), shape=(H, W))  # as on the scene
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
        """Sparse, global steps on all blocks' results: the scene's road :class:`VectorGraph`
        (see the class note)."""
        import warnings

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
            sel = o[suppress(pix[o], val[o], _KERNEL)]  # as peak_pixels on the scene
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
