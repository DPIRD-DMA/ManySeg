"""Block decoders: vectors from large stitched scenes, block by block (experimental).

The same decoders run inside ManyPred's block post-processing (``manypred.blocks``), which
calls ``decoder(window_fields, block, scene)`` per block and ``decoder.combine(results,
scene)`` once. ManySeg never imports ManyPred: blocks and scenes are duck-typed (``index``,
``core``, ``window``, ``scene_edges``; ``height``, ``width``, ``transform``, ``crs``), and
:func:`decode_scene` is an in-memory driver with the same calls::

    dec = block_decoder("footprints")
    records = decode_scene(fields, dec, scene=scene)                # or, with ManyPred:
    records = postprocess_blocks(fields, dec, combine=dec.combine, block=4096, halo=256)
    records.write("buildings.gpkg")  # SceneRecords: in the scene's transform and CRS

Dense fields (already stitched) are decoded in windows = core + halo; the cores partition the
scene. Every object is keyed by its *anchor*, the scene pixel index of its first pixel in raster
order, which no block grid changes. An instance that lies inside one core and keeps a margin
from its window edge is vectorised by that block alone; its result is a small record. Every
other instance becomes a *piece* in each block whose core it reaches: its pixels in the window
plus the field crops its vectorisation needs. ``combine`` joins pieces of neighbouring blocks
whose masks agree on the area both windows see (IoU >= ``min_iou``), shares out the pieces' core
pixels (cores partition the scene, so nothing is duplicated or dropped; a block that sees an
instance whole settles which pixels are its own) and vectorises each object once. The output
equals a whole-scene decode provided the window decoder labels an instance the same way as a
global decode: tie-breaks follow scene pixel indices (``origin`` in
:func:`manyseg.instances.instances_from_outputs`) and the decoder's reach stays below the halo.
Where windows disagree beyond that, an object may split at a seam, but no pixel is lost or
claimed twice.

Roofs: faces are joined with :func:`planar_faces` before their planes are fitted, which couples
neighbouring faces. A *cluster* (faces linked by gaps up to the join tolerance) is the unit:
a block resolves a cluster (join, planes, mesh) when every face in it is settled there (complete,
away from the window edge and from any instance the window sees only in part, by more than the
join tolerance plus the largest distance a face strays from its mask) and the cluster's anchor
(its smallest face anchor) lies in its core. The other faces travel with a height crop and are
joined in ``combine`` cluster by cluster, so the result equals
:func:`manyseg.vectors.roofs.roof_faces` on the whole scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .graph import _deps

Box = tuple[int, int, int, int]  # (top, bottom, left, right), half-open, scene pixels


@dataclass(frozen=True)
class Block:
    """One block of a :func:`block_grid` (attribute names as ManyPred's ``Block``)."""

    index: int  # row-major position in the grid
    core: Box  # cores partition the scene
    window: Box  # core grown by the halo on every side, clipped to the scene
    scene_edges: tuple[bool, bool, bool, bool]  # window top / bottom / left / right on the edge

    @property
    def inner(self) -> Box:
        """The core in window coordinates."""
        t, l0 = self.window[0], self.window[2]
        return (self.core[0] - t, self.core[1] - t, self.core[2] - l0, self.core[3] - l0)


@dataclass(frozen=True)
class Scene:
    """Scene size and georeferencing (attribute names as ManyPred's ``Scene``)."""

    height: int
    width: int
    transform: Any = None  # affine pixel (x, y) -> world, rasterio / affine.Affine order
    crs: Any = None


def block_grid(height: int, width: int, block: int = 4096, halo: int = 256,
               offset: tuple[int, int] = (0, 0)) -> list[Block]:
    """Row-major blocks whose cores tile the scene in ``block`` px steps (the last row and
    column are shorter; ``offset`` shortens the first ones instead, for grid-invariance tests);
    windows extend each core by ``halo`` px, clipped to the scene."""
    if block < 1 or halo < 0:
        raise ValueError(f"need block >= 1 and halo >= 0, got {block}, {halo}")

    def cuts(n, o):
        o = o % block
        return sorted({0, n, *range(o if o else block, n, block)})

    rs, cs = cuts(height, offset[0]), cuts(width, offset[1])
    out = []
    for r0, r1 in zip(rs[:-1], rs[1:]):
        for c0, c1 in zip(cs[:-1], cs[1:]):
            w = (max(r0 - halo, 0), min(r1 + halo, height), max(c0 - halo, 0),
                 min(c1 + halo, width))
            edges = (w[0] == 0, w[1] == height, w[2] == 0, w[3] == width)
            out.append(Block(len(out), (r0, r1, c0, c1), w, edges))
    return out


class BlockDecoder(Protocol):
    """Decodes one window (``fields``: name -> (C, h, w) arrays cropped to ``block.window``)
    into a picklable result; ``combine`` merges the results of all blocks in grid order."""

    def __call__(self, fields: dict[str, np.ndarray], block, scene) -> Any:
        """Decode one block's window; the result must be picklable (it may come from a
        worker process)."""
        ...

    def combine(self, results: list, scene) -> Any:
        """Join the per-block results (in grid order) into the scene's output."""
        ...


_UNSET: Any = object()  # "argument not given" (``None`` means pixel coordinates)


class SceneRecords(list):
    """The decoded objects of a scene: a ``list[dict]`` sorted by ``id`` (the anchor), with
    ``stats`` and the scene's ``transform`` and ``crs`` (set by ``combine`` from the scene).

    :meth:`to_geojson` and :meth:`write` put them in world coordinates through that transform
    and name that CRS, as :func:`records_to_geojson` / :func:`write_records` do."""

    def __init__(self, items=(), stats: dict | None = None, *, transform=None, crs=None):
        super().__init__(items)
        self.stats = stats or {}
        self.transform = transform
        self.crs = crs

    def to_geojson(self, *, mesh: bool = False) -> dict:
        """GeoJSON ``FeatureCollection`` through the scene's transform, naming its CRS
        (:func:`records_to_geojson`)."""
        return records_to_geojson(self, mesh=mesh)

    def write(self, path, *, layer: str | None = None, mesh: bool = False) -> Path:
        """Write to ``path`` (``.geojson``, or e.g. ``.gpkg`` with geopandas) in the scene's
        transform and CRS (:func:`write_records`). Returns the path."""
        return write_records(self, path, layer=layer, mesh=mesh)


def _scene_records(records, scene, stats: dict) -> SceneRecords:
    return SceneRecords(records, stats, transform=getattr(scene, "transform", None),
                        crs=getattr(scene, "crs", None))


def decode_scene(fields, decoder, block: int = 4096, halo: int = 256, *, scene=None,
                 offset: tuple[int, int] = (0, 0)):
    """Decode a scene, whole or block by block: the recommended way to run any decoder from
    :func:`block_decoder` (``decode_scene(fields, block_decoder(task))``).

    ``decoder(window_fields, block, scene)`` runs per block in grid order, then
    ``decoder.combine(results, scene)``, as ManyPred's ``postprocess_blocks`` does (no ManyPred
    needed). ``fields``: name -> (H, W) or (C, H, W) arrays (or memmaps) sharing (H, W).
    ``scene``: its size and georeferencing (:class:`Scene`; default ``Scene(H, W)``, pixel
    coordinates). With the defaults (``block=4096``, ``halo=256``) a scene of up to 4096 px a
    side is one block, i.e. a whole-scene decode; a larger one is cut into 4096 px cores with
    256 px of halo. The halo must be at least ``decoder.min_halo`` when there is more than one
    block (checked here, before any block runs); for roads exactness also needs the halo to
    cover the skeleton's data-dependent reach (:class:`~manyseg.vectors.RoadBlockDecoder`
    warns). Returns what ``combine`` returns: :class:`SceneRecords` (instances, roofs), a
    :class:`~manyseg.vectors.VectorGraph` (roads), :class:`~manyseg.points.ScenePoints` or
    :class:`~manyseg.points.SceneCount`."""
    if not len(fields):
        raise ValueError("fields must not be empty")
    shapes = {k: tuple(np.shape(v))[-2:] for k, v in fields.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"fields must share (H, W), got {shapes}")
    H, W = next(iter(shapes.values()))
    scene = scene if scene is not None else Scene(H, W)
    if (scene.height, scene.width) != (H, W):
        raise ValueError(f"scene is {scene.height} x {scene.width}, fields are {H} x {W}")
    grid = block_grid(H, W, block, halo, offset)
    need = getattr(decoder, "min_halo", 0)  # every block decoder here defines it
    if halo < need and len(grid) > 1:
        raise ValueError(f"halo {halo} is below this decoder's minimum {need} (min_halo)")
    results = []
    for b in grid:
        t, bt, lt, rt = b.window
        wf = {k: np.asarray(v[..., t:bt, lt:rt]) for k, v in fields.items()}
        results.append(decoder({k: v if v.ndim == 3 else v[None] for k, v in wf.items()},
                               b, scene))
    return decoder.combine(results, scene)


# ---------------------------------------------------------------- pieces and their merge

@dataclass
class _Piece:
    """One block's view of an instance that crosses its core or nears its window edge."""

    block: int
    bbox: Box  # scene box of ``mask``
    mask: np.ndarray  # bool, the instance's pixels in the window
    core: Box
    window: Box
    crops: dict = field(default_factory=dict)  # name -> ((row, col), array) field crops
    peaks: tuple | None = None  # (pix (N, 2) scene row / col, xy (N, 2)) near the core pixels
    margin: int = -1  # gap to the window edge (not the scene edge) if the window sees it whole


@dataclass
class _Pending:
    """A roof face its owner could not join (its cluster reaches outside what it sees whole)."""

    anchor: int
    face: Any
    crop: tuple  # ((row, col), height crop)


@dataclass
class _BlockResult:
    """What a block decoder returns for one block (picklable)."""

    index: int
    records: list  # finished objects (small)
    pieces: list  # _Piece: instances finished in ``combine``
    pending: list = field(default_factory=list)  # roofs: _Pending faces
    resolved: list = field(default_factory=list)  # roofs: anchors of faces in ``records``
    stats: dict = field(default_factory=dict)


def _inter(a: Box | None, b: Box | None) -> Box | None:
    if a is None or b is None:
        return None
    r = (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))
    return r if r[0] < r[1] and r[2] < r[3] else None


def _sub(m: np.ndarray, box: Box, region: Box) -> np.ndarray:
    return m[region[0] - box[0] : region[1] - box[0], region[2] - box[2] : region[3] - box[2]]


def _grow(box: Box, d: int, limit: Box) -> Box:
    return (max(box[0] - d, limit[0]), min(box[1] + d, limit[1]), max(box[2] - d, limit[2]),
            min(box[3] + d, limit[3]))


def _anchor(mask: np.ndarray, r0: int, c0: int, W: int) -> int:
    fr = int(np.argmax(mask.reshape(-1)))  # first pixel in raster order
    return (r0 + fr // mask.shape[1]) * W + (c0 + fr % mask.shape[1])


def _paste(box: Box, crops, fill=np.nan) -> np.ndarray:
    """Canvas over scene ``box`` from ``(origin, array)`` crops ((C, h, w) or (h, w));
    pixels no crop covers get ``fill``."""
    a0 = crops[0][1]
    lead = a0.shape[:-2]
    out = np.full(lead + (box[1] - box[0], box[3] - box[2]), fill, np.float32)
    for (r, c), a in crops:
        reg = _inter(box, (r, r + a.shape[-2], c, c + a.shape[-1]))
        if reg is not None:
            out[..., reg[0] - box[0] : reg[1] - box[0], reg[2] - box[2] : reg[3] - box[2]] = (
                a[..., reg[0] - r : reg[1] - r, reg[2] - c : reg[3] - c])
    return out


def _merge_pieces(pieces: list, min_iou: float) -> list[tuple[Box, np.ndarray, list]]:
    """Join pieces into objects: (box, mask, pieces) per object, masks disjoint.

    Pieces of different blocks are one object if their masks agree (IoU >= ``min_iou``) on
    the whole area both windows see. The pixels to share out are every piece's *core* pixels
    (cores partition the scene), so nothing is dropped or duplicated. A pixel claimed by no
    *whole view* (a piece whose window sees the instance with a margin, ``margin`` >= 0) goes
    to the object its core block's piece belongs to; a claimed pixel goes to that object too
    if its whole view claims it, else to the claimant with the most central view. When windows
    decode an instance alike (its reach is below the halo) all rules agree; when they do not
    (a window that misses a large neighbour's seeds joins two instances), a whole view from
    another block still separates them.
    """
    shapely = _deps()[1]
    n = len(pieces)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def count(p, region):
        reg = _inter(p.bbox, region)
        return 0 if reg is None else int(_sub(p.mask, p.bbox, reg).sum())

    boxes = [shapely.box(p.bbox[2], p.bbox[0], p.bbox[3], p.bbox[1]) for p in pieces]
    tree = shapely.STRtree(boxes)
    for i, a in enumerate(pieces):
        for j in tree.query(boxes[i], predicate="intersects").tolist():
            b = pieces[j]
            if j <= i or a.block == b.block:
                continue
            shared = _inter(a.window, b.window)  # both windows see all of it
            reg = _inter(_inter(a.bbox, b.bbox), shared)
            if reg is None:
                continue
            inter = int((_sub(a.mask, a.bbox, reg) & _sub(b.mask, b.bbox, reg)).sum())
            if inter and inter / (count(a, shared) + count(b, shared) - inter) >= min_iou:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    members = list(groups.values())
    gid = np.empty(n, np.int64)  # piece -> object
    for g, idx in enumerate(members):
        gid[idx] = g
    view = []  # each object's most central whole view (piece index) or -1
    for idx in members:
        v = max(idx, key=lambda i: (pieces[i].margin, -pieces[i].block))
        view.append(v if pieces[v].margin >= 0 else -1)
    views = sorted((v for v in view if v >= 0), key=lambda v: (pieces[v].margin, -pieces[v].block))
    vtree = shapely.STRtree([boxes[v] for v in views]) if views else None
    def pixels(p, region):  # scene (row, col) of p's mask pixels inside ``region``
        reg = _inter(p.bbox, region)
        if reg is None:
            return np.zeros((0, 2), np.int64)
        return np.argwhere(_sub(p.mask, p.bbox, reg)) + (reg[0], reg[2])

    def hits(p, rc, region=None):  # which of the scene pixels ``rc`` p's mask holds (in region)
        box = p.bbox if region is None else _inter(p.bbox, region)
        if box is None:
            return np.zeros(len(rc), bool)
        ok = ((rc[:, 0] >= box[0]) & (rc[:, 0] < box[1]) & (rc[:, 1] >= box[2])
              & (rc[:, 1] < box[3]))
        ok[ok] = p.mask[rc[ok, 0] - p.bbox[0], rc[ok, 1] - p.bbox[2]]
        return ok

    out = []
    for g, idx in enumerate(members):
        # candidate pixels (sparse: never a dense canvas over a large, thin object's box): its
        # core pixels and those its whole view claims
        rc = [pixels(pieces[i], pieces[i].core) for i in idx]
        if view[g] >= 0:
            rc.append(pixels(pieces[view[g]], pieces[view[g]].bbox))
        rc = np.unique(np.concatenate(rc).reshape(-1, 2), axis=0)
        if not len(rc):
            continue
        box = (int(rc[:, 0].min()), int(rc[:, 0].max()) + 1, int(rc[:, 1].min()),
               int(rc[:, 1].max()) + 1)
        q = shapely.box(box[2], box[0], box[3], box[1])
        owner = np.full(len(rc), -1, np.int64)  # object of each core pixel (core block's view)
        for i in tree.query(q, predicate="intersects").tolist():
            owner[hits(pieces[i], rc, pieces[i].core)] = gid[i]
        if vtree is not None:
            best = np.full(len(rc), -1, np.int64)  # most central claimant
            own = np.zeros(len(rc), bool)  # the core block's object claims it itself
            for k in sorted(vtree.query(q, predicate="intersects").tolist()):  # by priority
                h = hits(pieces[views[k]], rc)
                best[h] = gid[views[k]]
                own |= h & (owner == gid[views[k]])
            owner = np.where((best >= 0) & ~own & (owner >= 0), best, owner)
        rc = rc[owner == g]
        if not len(rc):
            continue
        tight = (int(rc[:, 0].min()), int(rc[:, 0].max()) + 1, int(rc[:, 1].min()),
                 int(rc[:, 1].max()) + 1)
        m = np.zeros((tight[1] - tight[0], tight[3] - tight[2]), bool)
        m[rc[:, 0] - tight[0], rc[:, 1] - tight[2]] = True
        out.append((tight, m, [pieces[i] for i in idx]))
    return out


# ---------------------------------------------------------------- instance decoders

@dataclass
class InstanceBlockDecoder:
    """Instances (boundary + distance heads) -> polygons, block by block.

    ``method="corners"``: :func:`polygons_from_corners`, straight walls between predicted corners
    (needs ``corner`` and ``direction``); ``"outline"``: :func:`polygons_from_labels`, the
    simplified instance outlines. Instance thresholds as
    :func:`manyseg.instances.instances_from_outputs`; polygon parameters as the decoders'.
    ``logits``: ``False`` (default), stitched ``extent``, ``boundary`` and ``corner`` are
    probabilities; ``True``, they are logits and a sigmoid is applied to them (``distance``,
    ``direction`` and roofs' ``height`` are values, never transformed). ``names`` maps the
    field names used here (``extent``, ``boundary``, ``distance``, ``corner``, ``direction``)
    to the caller's.
    Result (:meth:`combine`): :class:`SceneRecords`, dicts with ``id`` (anchor), ``geometry``
    (shapely, scene pixel coordinates), ``pixels`` (instance size) and ``scene_edge`` (touches
    the scene edge, so may be cut).
    """

    method: str = "corners"
    extent_threshold: float = 0.6
    boundary_threshold: float = 0.5
    seed_threshold: float = 0.15
    min_seed: int = 8
    min_size: int = 16
    logits: bool = False
    backend: str = "auto"
    corner_threshold: float = 0.5
    kernel: int = 3
    snap_radius: float = 4.0
    big_dev: float = 4.0
    agree: float = 0.4
    merge_deg: float = 8.0
    simplify: float = 1.0
    min_area: float = 16
    min_iou: float = 0.5
    names: dict = field(default_factory=dict)
    holes: bool = True  # method="corners": keep courtyards, as polygons_from_corners

    def __post_init__(self):
        if self.method not in ("corners", "outline"):
            raise ValueError(f"method must be corners|outline, got {self.method!r}")

    # -- geometry of the decoder's reach (px)
    @property
    def margin(self) -> int:
        """Gap an instance must keep from a window edge (not the scene edge) to be decoded
        there: the corner peaks and directions it reads lie within this of its pixels."""
        if self.method == "outline":
            return 1
        return math.ceil(self.snap_radius + 1.5) + self.kernel + 2

    @property
    def crop_margin(self) -> int:
        """Field context a piece carries around its core pixels."""
        return self.margin

    @property
    def min_halo(self) -> int:
        """Smallest halo for which pieces carry valid context (peaks need ``kernel`` + 3 px
        of window beyond the crop)."""
        if self.method == "outline":
            return self.margin + 1
        return self.crop_margin + self.kernel + 3

    # -- per window
    def _field(self, fields, name) -> np.ndarray:
        a = np.asarray(fields[self.names.get(name, name)], np.float32)
        return a if a.ndim == 3 else a[None]

    def labels(self, fields, origin: tuple[int, int], scene_width: int) -> np.ndarray:
        """(h, w) instance labels of one window (1..K, any order); no size filter (a window
        may see only part of an instance)."""
        import torch

        from ..instances import instances_from_outputs

        t = [torch.from_numpy(self._field(fields, n))[None]
             for n in ("extent", "boundary", "distance")]
        return instances_from_outputs(
            *t, logits=self.logits, extent_threshold=self.extent_threshold,
            boundary_threshold=self.boundary_threshold, seed_threshold=self.seed_threshold,
            min_seed=self.min_seed, min_size=0, backend=self.backend, origin=origin,
            scene_width=scene_width,
        )[0].numpy()

    def _context(self, fields, origin) -> dict:
        """What :meth:`_vectorise` reads, for a window at scene ``origin``."""
        if self.method == "outline":
            return {}
        from .corners import _corner_peaks

        corner = self._field(fields, "corner")[0]
        if self.logits:  # as extent and boundary in instances_from_outputs(logits=True)
            import torch

            corner = torch.sigmoid(torch.from_numpy(corner)).numpy()
        pix, xy = _corner_peaks(corner, self.corner_threshold, self.kernel, origin)
        o = np.argsort(pix[:, 0], kind="stable")
        return {"pix": pix[o], "xy": xy[o], "direction": (origin, self._field(fields, "direction"))}

    def _peaks_near(self, ctx, box: Box):
        pix = ctx["pix"]
        a, b = np.searchsorted(pix[:, 0], [box[0], box[1]])
        sel = np.arange(a, b)
        sel = sel[(pix[sel, 1] >= box[2]) & (pix[sel, 1] < box[3])]
        return pix[sel], ctx["xy"][sel]

    def _vectorise(self, mask, y0: int, x0: int, ctx):
        if self.method == "outline":
            from .polygons import _outline

            q = _outline(mask, y0, x0).simplify(self.simplify)
            return q if q.geom_type == "Polygon" and q.area >= self.min_area else None
        from .corners import _instance_polygon

        d = math.ceil(self.snap_radius) + 2
        box = (y0 - d, y0 + mask.shape[0] + d, x0 - d, x0 + mask.shape[1] + d)
        (o, dr) = ctx["direction"]
        return _instance_polygon(
            mask, y0, x0, self._peaks_near(ctx, box)[1], dr, o, snap_radius=self.snap_radius,
            big_dev=self.big_dev, agree=self.agree, merge_deg=self.merge_deg,
            simplify=self.simplify, min_area=self.min_area, holes=self.holes)

    def _crops(self, fields, ctx, box: Box, window: Box) -> tuple[dict, tuple | None]:
        """Field crops (and corner peaks) over scene ``box`` (inside ``window``) for a piece."""
        if self.method == "outline":
            return {}, None
        (o, dr) = ctx["direction"]
        crop = dr[:, box[0] - o[0] : box[1] - o[0], box[2] - o[1] : box[3] - o[1]]
        return {"direction": ((box[0], box[2]), crop.copy())}, self._peaks_near(ctx, box)

    def _piece_context(self, group: list, box: Box, scene) -> dict:
        """:meth:`_context` for a merged object over scene ``box`` from its pieces' crops."""
        if self.method == "outline":
            return {}
        lim = (0, scene.height, 0, scene.width)
        dbox = _grow(box, self.crop_margin, lim)
        pix = np.concatenate([p.peaks[0] for p in group]).reshape(-1, 2)
        xy = np.concatenate([p.peaks[1] for p in group]).reshape(-1, 2)
        if len(pix):
            _, first = np.unique(pix, axis=0, return_index=True)  # seen by two blocks
            first.sort()
            pix, xy = pix[first], xy[first]
        o = np.argsort(pix[:, 0], kind="stable")
        dr = _paste(dbox, [p.crops["direction"] for p in group], 0.0)
        return {"pix": pix[o], "xy": xy[o], "direction": ((dbox[0], dbox[2]), dr)}

    def _record(self, anchor, geom, n, box: Box, scene) -> dict:
        edge = box[0] == 0 or box[2] == 0 or box[1] == scene.height or box[3] == scene.width
        return {"id": int(anchor), "geometry": geom, "pixels": int(n), "scene_edge": bool(edge)}

    def _split(self, fields, block, scene):
        """Window decode -> (complete objects, pieces, incomplete label ids, labels, ctx).
        A complete object: (anchor, box, mask, inside core); its geometry is not computed."""
        from scipy import ndimage

        wt, wb, wl, wr = block.window
        R, e = self.margin, block.scene_edges
        ct, cb, cl, cr = block.core
        halo = min([ct - wt] * (not e[0]) + [wb - cb] * (not e[1]) + [cl - wl] * (not e[2])
                   + [wr - cr] * (not e[3]) + [1 << 30])
        if halo < self.min_halo:
            raise ValueError(f"halo {halo} is below this decoder's minimum {self.min_halo}")
        if wb > scene.height or wr > scene.width:
            raise ValueError(f"window {block.window} is outside the scene "
                             f"({scene.height} x {scene.width})")
        lab = self.labels(fields, (wt, wl), scene.width)
        ctx = self._context(fields, (wt, wl))
        complete, pieces, partial, outside = [], [], [], []
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab[sl] == k
            box = (wt + sl[0].start, wt + sl[0].stop, wl + sl[1].start, wl + sl[1].stop)
            ok = ((e[0] or box[0] - wt >= R) and (e[1] or wb - box[1] >= R)
                  and (e[2] or box[2] - wl >= R) and (e[3] or wr - box[3] >= R))
            inside = _inter(box, block.core) == box
            gap = min([box[0] - wt] * (not e[0]) + [wb - box[1]] * (not e[1])
                      + [box[2] - wl] * (not e[2]) + [wr - box[3]] * (not e[3]) + [1 << 30])
            if ok:
                complete.append((_anchor(m, box[0], box[2], scene.width), box, m, inside, k))
            else:
                partial.append(k)
            if ok and inside:
                continue
            core = _inter(box, block.core)
            if core is None or not _sub(m, box, core).any():
                if ok:  # another block's core holds all of it; a whole view may still help
                    outside.append((sl, box, m, gap))
                continue
            cm = _sub(m, box, core)
            rr, cc = np.nonzero(cm.any(1))[0], np.nonzero(cm.any(0))[0]
            tight = (core[0] + rr[0], core[0] + rr[-1] + 1, core[2] + cc[0], core[2] + cc[-1] + 1)
            crops, peaks = self._crops(fields, ctx, _grow(tight, self.crop_margin, block.window),
                                       block.window)
            pieces.append(_Piece(block.index, box, m, block.core, block.window, crops, peaks,
                                 gap if ok else -1))
        if pieces and outside:
            # Whole views of instances held by another core that touch (through touching
            # instances) one of this block's pieces: if that core's window joins them with a
            # neighbour it cannot see whole, these views separate them again (_merge_pieces).
            comp = ndimage.label(lab > 0)[0]
            hot = np.zeros(comp.max() + 1, bool)
            for p in pieces:
                sl = (slice(p.bbox[0] - wt, p.bbox[1] - wt), slice(p.bbox[2] - wl, p.bbox[3] - wl))
                hot[comp[sl][p.mask]] = True
            for sl, box, m, gap in outside:
                if hot[comp[sl][m][0]]:
                    pieces.append(_Piece(block.index, box, m, block.core, block.window,
                                         margin=gap))
        return complete, pieces, partial, lab, ctx

    def __call__(self, fields, block, scene) -> _BlockResult:
        """Decode one block's window: records of the objects it settles, pieces of the rest."""
        complete, pieces, _, _, ctx = self._split(fields, block, scene)
        records = []
        for anchor, box, m, inside, _ in complete:
            if not inside:
                continue
            n = int(m.sum())
            if n < self.min_size:
                continue
            g = self._vectorise(m, box[0], box[2], ctx)
            if g is not None:
                records.append(self._record(anchor, g, n, box, scene))
        return _BlockResult(block.index, records, pieces,
                           stats={"records": len(records), "pieces": len(pieces)})

    def _merged(self, results, scene):
        """Objects rebuilt from pieces after ``min_size``: (anchor, box, mask, pieces whose
        field crops reach the object; the crops of all of them together cover its context)."""
        shapely = _deps()[1]
        pieces = [p for res in results for p in res.pieces]
        cropped = [p for p in pieces if p.crops]
        cboxes = []
        for p in cropped:
            (r, c), a = next(iter(p.crops.values()))
            cboxes.append(shapely.box(c, r, c + a.shape[-1], r + a.shape[-2]))
        tree = shapely.STRtree(cboxes)
        lim = (0, scene.height, 0, scene.width)
        out = []
        for box, m, _ in _merge_pieces(pieces, self.min_iou):
            n = int(m.sum())
            if n < self.min_size:
                continue
            gb = _grow(box, self.crop_margin, lim)
            near = tree.query(shapely.box(gb[2], gb[0], gb[3], gb[1]), predicate="intersects")
            out.append((_anchor(m, box[0], box[2], scene.width), box, m,
                        [cropped[i] for i in sorted(near.tolist())]))
        return out, len(pieces)

    def combine(self, results: list, scene) -> SceneRecords:
        """Join pieces across blocks, vectorise each object once: :class:`SceneRecords`."""
        records = [r for res in results for r in res.records]
        merged, n_pieces = self._merged(results, scene)
        for anchor, box, m, g in merged:
            geom = self._vectorise(m, box[0], box[2], self._piece_context(g, box, scene))
            if geom is not None:
                records.append(self._record(anchor, geom, int(m.sum()), box, scene))
        ids = [r["id"] for r in records]
        if len(set(ids)) != len(ids):  # disjoint masks have distinct first pixels
            raise RuntimeError("anchor collision: two objects claim the same first pixel")
        records.sort(key=lambda r: r["id"])
        return _scene_records(records, scene, {"blocks": len(results), "pieces": n_pieces,
                                               "merged_objects": len(merged)})


@dataclass
class RoofBlockDecoder(InstanceBlockDecoder):
    """Roof faces (corners decoder) joined by :func:`planar_faces`, a robust plane per face
    from the ``height`` field (nDSM, metres) and an optional mesh (:mod:`manyseg.vectors.roofs`).
    Result: :class:`SceneRecords` sorted by ``(id, part)`` as
    :func:`manyseg.vectors.roofs.roof_faces`: ``id``, ``part``, ``geometry``, ``plane``
    (a, b, c) and ``mesh`` ((K, 3) ring) unless ``mesh`` is None."""

    extent_threshold: float = 0.4
    boundary_threshold: float = 0.3
    seed_threshold: float = 0.15
    agree: float = 0.6
    tolerance: float = 2.5
    erode: int = 2
    trim: float = 0.2
    mesh: str | None = "plane_intersection"
    holes: bool = False  # faces join and mesh on their outer walls

    def __post_init__(self):
        if self.method != "corners":
            raise ValueError("roofs decode faces with method='corners'")

    @property
    def stray(self) -> float:
        """Bound on how far a face leaves its instance's pixels: corners move at most
        3 x ``snap_radius`` from the outline (``_polygon``), plus simplification and slack."""
        return 3 * self.snap_radius + self.simplify + 2

    @property
    def settle(self) -> int:
        """Gap from a window edge / partly seen instance that makes a face's cluster certain."""
        return math.ceil(self.tolerance + self.stray) + 2

    @property
    def crop_margin(self) -> int:
        # a joined face stays within ~stray + tolerance of its pixels; the plane fit reads
        # its bounding box + erode + 1
        return max(self.margin, math.ceil(self.stray + 2 * self.tolerance) + self.erode + 2)

    def _context(self, fields, origin) -> dict:
        ctx = super()._context(fields, origin)
        ctx["height"] = (origin, self._field(fields, "height")[0])
        return ctx

    def _crops(self, fields, ctx, box, window):
        crops, peaks = super()._crops(fields, ctx, box, window)
        (o, h) = ctx["height"]
        crops["height"] = ((box[0], box[2]),
                           h[box[0] - o[0] : box[1] - o[0], box[2] - o[1] : box[3] - o[1]].copy())
        return crops, peaks

    def _join(self, anchors, faces, height, origin):
        from .roofs import roof_records

        return roof_records(anchors, faces, height, origin, tolerance=self.tolerance,
                            erode=self.erode, trim=self.trim, mesh=self.mesh)

    def _clusters(self, faces: list) -> list[list[int]]:
        """Indices of ``faces`` grouped as :func:`planar_faces` joins them."""
        from .polygons import _face_clusters

        return _face_clusters(faces, self.tolerance)

    def __call__(self, fields, block, scene) -> _BlockResult:
        """Decode one block's window: the face clusters it resolves, pieces of the rest."""
        from scipy import ndimage

        shapely = _deps()[1]
        complete, pieces, partial, lab, ctx = self._split(fields, block, scene)
        wt, wb, wl, wr = block.window
        W = scene.width
        S, e = self.settle, block.scene_edges
        near = None
        if partial:  # distance (px) to the nearest pixel of an instance seen only in part
            near = ndimage.distance_transform_edt(~np.isin(lab, partial))
        anchors, faces, settled, owned = [], [], [], []
        for anchor, box, m, inside, _ in complete:
            if int(m.sum()) < self.min_size:
                continue
            g = self._vectorise(m, box[0], box[2], ctx)
            if g is None:
                continue
            x0, y0, x1, y1 = g.bounds
            ok = ((e[0] or y0 - wt >= S) and (e[1] or wb - y1 >= S)
                  and (e[2] or x0 - wl >= S) and (e[3] or wr - x1 >= S))
            if ok and near is not None:
                rr = (max(int(y0) - wt - 1, 0), min(math.ceil(y1) - wt + 1, wb - wt))
                cc = (max(int(x0) - wl - 1, 0), min(math.ceil(x1) - wl + 1, wr - wl))
                yy, xx = np.mgrid[rr[0] : rr[1], cc[0] : cc[1]]
                px = shapely.box(xx + wl, yy + wt, xx + wl + 1, yy + wt + 1)
                hit = shapely.intersects(g, px)  # pixels the face touches
                ok = not hit.any() or float(near[rr[0] : rr[1], cc[0] : cc[1]][hit].min()) > S
            anchors.append(anchor)
            faces.append(g)
            settled.append(ok)
            owned.append(inside)
        records, resolved, pending = [], [], []
        done = set()
        (ho, h) = ctx["height"]
        for cl in self._clusters(faces):
            a = min(anchors[i] for i in cl)
            ar, ac = divmod(a, W)
            if all(settled[i] for i in cl) and _inter((ar, ar + 1, ac, ac + 1), block.core):
                records += self._join([anchors[i] for i in cl], [faces[i] for i in cl], h, ho)
                resolved += [anchors[i] for i in cl]
                done.update(cl)
        pad = math.ceil(2 * self.tolerance) + self.erode + 2
        for i in range(len(faces)):
            if owned[i] and i not in done:
                x0, y0, x1, y1 = faces[i].bounds
                box = _grow((int(y0), int(math.ceil(y1)), int(x0), int(math.ceil(x1))), pad,
                            block.window)
                crop = h[box[0] - ho[0] : box[1] - ho[0], box[2] - ho[1] : box[3] - ho[1]].copy()
                pending.append(_Pending(anchors[i], faces[i], ((box[0], box[2]), crop)))
        return _BlockResult(block.index, records, pieces, pending, resolved,
                           stats={"faces": len(faces), "resolved": len(resolved),
                                  "pending": len(pending), "pieces": len(pieces)})

    def combine(self, results: list, scene) -> SceneRecords:
        """Join pieces and clusters across blocks: faces with planes (and meshes), sorted by
        ``(id, part)``."""
        records = [rec for res in results for rec in res.records]
        resolved = [a for res in results for a in res.resolved]
        if len(set(resolved)) != len(resolved):
            raise RuntimeError("two blocks resolved the same roof face")
        done = set(resolved)
        anchors, faces, crops = [], [], []
        for res in results:
            for p in res.pending:
                if p.anchor not in done:
                    anchors.append(p.anchor)
                    faces.append(p.face)
                    crops.append([p.crop])
        merged, n_pieces = self._merged(results, scene)
        for anchor, box, m, g in merged:
            if anchor in done:
                continue
            face = self._vectorise(m, box[0], box[2], self._piece_context(g, box, scene))
            if face is not None:
                anchors.append(anchor)
                faces.append(face)
                crops.append([p.crops["height"] for p in g])
        if len(set(anchors)) != len(anchors):
            raise RuntimeError("anchor collision: two roof faces claim the same first pixel")
        lim = (0, scene.height, 0, scene.width)
        nan_planes = 0
        clusters = self._clusters(faces)
        for cl in clusters:
            cc = [c for i in cl for c in crops[i]]

            def height(face, cc=cc):  # crop around one joined face (planes read + erode + 1)
                x0, y0, x1, y1 = face.bounds
                box = _grow((int(y0), math.ceil(y1), int(x0), math.ceil(x1)), self.erode + 2,
                            lim)
                return _paste(box, cc), (box[0], box[2])

            out = self._join([anchors[i] for i in cl], [faces[i] for i in cl], height, None)
            nan_planes += sum(not np.isfinite(o["plane"]).all() for o in out)
            records += out
        records.sort(key=lambda rec: (rec["id"], rec["part"]))
        return _scene_records(records, scene, {
            "blocks": len(results), "pieces": n_pieces, "merged_objects": len(merged),
            "resolved_in_blocks": len(done), "joined_in_combine": len(faces),
            "combine_clusters": len(clusters), "nan_planes": nan_planes})


# ---------------------------------------------------------------- registry and output

_DEFAULTS = {
    # corner decoding tuned on WHU (corner 0.5 / big_dev 4 / agree 0.4 / merge 8), on the
    # instance thresholds tuned there for outlines
    "footprints": {"method": "corners", "extent_threshold": 0.6, "boundary_threshold": 0.5,
                   "seed_threshold": 0.15},
    # outlines on Fields of The World with absolute distance: thresholds 0.4 / 0.3 / 3 px,
    # Douglas-Peucker 0.5 px
    "fields": {"method": "outline", "extent_threshold": 0.4, "boundary_threshold": 0.3,
               "seed_threshold": 3.0, "simplify": 0.5},
}


def block_decoder(task: str, **params) -> BlockDecoder:
    """A block decoder with the tuned defaults for ``task``; ``params`` override them. Run it
    with :func:`decode_scene` (whole scene or block by block) or ManyPred's block
    post-processing. The low-level functions (:func:`polygons_from_corners`,
    :func:`polygons_from_labels`, ...) keep generic defaults; the tuned settings live here.

    * ``"footprints"``: extent, boundary, distance + corner, direction; ``method="corners"``,
      straight walls between corners (courtyards kept as traced outlines,
      ``holes=False`` fills them) or ``"outline"``, instance outlines (``simplify`` 1 px unless
      given).
    * ``"fields"``: extent, boundary, distance; instance outlines, absolute distance
      (``seed_threshold`` in px; for a model trained on normalised distance pass
      ``seed_threshold=0.3, extent_threshold=0.6``).
    * ``"roofs"``: as footprints + ``height`` (nDSM, metres); thresholds tuned on trained roof
      models, :func:`planar_faces` (2.5 px), robust planes and a
      ``"plane_intersection"`` mesh (``mesh="mean_height"`` is the alternative,
      ``mesh=None`` skips it).
    * ``"roads"``: mask, keypoints, direction; :class:`manyseg.vectors.roads.RoadBlockDecoder`.
    * ``"points"``: a point heatmap; :class:`manyseg.points.blocks.PointBlockDecoder`.
    * ``"density"``: a density head's count; :class:`manyseg.points.blocks.DensityBlockDecoder`
      (``sigma=`` or ``value_scale=`` required, as the head was trained with ``density_loss``).
    """
    if task == "roads":
        from .roads import RoadBlockDecoder

        return RoadBlockDecoder(**params)
    if task in ("points", "density"):
        from ..points.blocks import DensityBlockDecoder, PointBlockDecoder

        return (PointBlockDecoder if task == "points" else DensityBlockDecoder)(**params)
    if task == "roofs":
        return RoofBlockDecoder(**params)
    if task not in _DEFAULTS:
        raise ValueError(f"unknown task {task!r}: footprints|fields|roofs|roads|points|density")
    kw = dict(_DEFAULTS[task])
    if task == "footprints" and params.get("method") == "outline":
        kw["simplify"] = 1.0  # tuned on WHU
    kw.update(params)
    return InstanceBlockDecoder(**kw)


def _world(geom, transform):
    from .georef import pixel_to_world

    return geom if transform is None else pixel_to_world(geom, transform)


def _properties(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k in ("geometry", "mesh"):
            continue
        if k == "plane":
            out.update(plane_a=v[0], plane_b=v[1], plane_c=v[2])
        elif isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _geometries(records, transform, mesh: bool) -> list:
    shapely = _deps()[1]
    out = []
    for r in records:
        g = r["geometry"]
        if mesh and r.get("mesh") is not None:
            ring = np.asarray(r["mesh"])
            g = shapely.Polygon(np.r_[ring, ring[:1]])
        out.append(_world(g, transform))
    return out


def records_to_geojson(records, transform=_UNSET, crs=_UNSET, *, mesh: bool = False) -> dict:
    """GeoJSON ``FeatureCollection`` of block-decoder records (or a :class:`VectorGraph`, via
    :func:`to_geojson`). ``transform``: affine pixel -> world (rasterio order); ``None``: pixel
    coordinates; not given: the records' own (:class:`SceneRecords`, or a :class:`VectorGraph`
    from :func:`decode_scene`), else pixel coordinates.
    ``crs`` (not given: the records' own, but none for pixel coordinates) is written as a
    ``name`` member when known.
    ``mesh=True`` writes roof faces as 3-D polygons from their mesh rings (z in metres).
    ``plane`` becomes ``plane_a`` / ``_b`` / ``_c`` (in scene pixel coordinates)."""
    import json

    shapely = _deps()[1]
    from .graph import VectorGraph
    from .polygons import to_geojson

    if transform is _UNSET:
        transform = getattr(records, "transform", None)
    if crs is _UNSET:  # pixel coordinates are not in the scene's CRS: name it only with a transform
        crs = getattr(records, "crs", None) if transform is not None else None
    if isinstance(records, VectorGraph):
        fc = to_geojson(records, transform)
    else:
        geoms = _geometries(records, transform, mesh)
        fc = {"type": "FeatureCollection",
              "features": [{"type": "Feature", "geometry": json.loads(shapely.to_geojson(g)),
                            "properties": _properties(r)} for g, r in zip(geoms, records)]}
    if crs is not None:
        fc["crs"] = {"type": "name", "properties": {"name": str(crs)}}
    return fc


def write_records(records, path, scene=None, *, layer: str | None = None,
                  mesh: bool = False) -> Path:
    """Write records to ``path`` in world coordinates through ``scene.transform`` / ``crs``
    (``scene`` not given: the records' own, as :class:`SceneRecords` and a
    :class:`VectorGraph` from :func:`decode_scene` carry them; pixel coordinates when there is
    no transform): ``.geojson`` / ``.json`` with the standard library, anything else (e.g.
    ``.gpkg``) with geopandas. Returns the path."""
    import json

    path = Path(path)
    src = records if scene is None else scene
    transform = getattr(src, "transform", None)
    crs = getattr(src, "crs", None)
    if path.suffix.lower() in (".geojson", ".json"):
        path.write_text(json.dumps(records_to_geojson(records, transform, crs, mesh=mesh)))
        return path
    try:
        import geopandas as gpd
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError(f"writing {path.suffix} needs geopandas; use .geojson") from e
    from .graph import VectorGraph

    if isinstance(records, VectorGraph):
        shapely = _deps()[1]
        geoms = [_world(shapely.LineString(p), transform) for p in records.polylines()]
        props: list = [{} for _ in geoms]
    else:
        geoms = _geometries(records, transform, mesh)
        props = [_properties(r) for r in records]
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
    gdf.to_file(path, layer=layer or path.stem)
    return path
