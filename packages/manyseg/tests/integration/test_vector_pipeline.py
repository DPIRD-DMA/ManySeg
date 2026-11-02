"""Vector outputs end to end, through ManyPred's real stitching.

Source vectors in a CRS -> per-tile training targets (``to_pixels`` + the task's encoder, one
call per ManyPred window) -> ``manypred.predict_array(..., merge="cut" | "blend")`` stitches
them -> block decoding (``decode_scene`` on a block grid that is *not* aligned to the tiles,
and ``manypred.postprocess_blocks``, what users run on saved rasters) -> GeoJSON and GPKG in
the source CRS -> read back -> compared with the source vectors in map units.

How ManyPred is driven: the "image" is two channels holding each pixel's scene row and column
(exact in float32). ``TileTargets`` is a ``torch.nn.Module`` whose forward reads the window's
top-left from those channels, checks that the patch is exactly that window of the scene,
encodes the targets of that tile from the world vectors (the tile's geotransform is the
scene's shifted by the window offset) and returns them as a dict of perfect predictions
(probabilities, decoded with ``logits=False``). So the windows, batching, cut / blend weights
and mosaicking are ManyPred's own code path (``predict_array`` -> ``Tiler`` -> ``Mosaic``);
nothing here re-implements stitching. An extra ``owner`` output (the window index, merged with
``"cut"``) gives the cut seams as ManyPred draws them.

The perfect per-tile targets are not tile invariant in known ways (measured here, not hidden):
the instance ``distance`` target is normalised per instance *within the tile*; the corner /
direction (and road keypoint / mask) targets see objects clipped by the tile, so a clip edge
gets corners and walls; and ``corner_targets`` simplifies the clipped ring, so a clipped
polygon's walls can differ. Every run first checks (``check_invariance``) that the stitched
targets equal one whole-scene encoding except exactly there; the decode checks then measure
what that does to the vectors, against the source and against a single-pass decode (seam
objects reported apart). Each instance task runs with ``distance="normalized"`` and
``distance="absolute"`` (the default of ``field_targets`` since E052; footprints and roofs
default to normalised); ``seam_study`` (run by hand) measures the seam errors of both.

Thresholds are the oracle numbers of the experiments (perfect targets through the shipped
decoders): see ``THRESHOLDS``.

Needs ManyPred with ``merge=`` (``feature/vector-blocks``); run locally with
``PYTHONPATH=".../ManyPred-wt/blocks/src"``.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("rasterio")
pytest.importorskip("skimage")
if os.environ.get("MANYSEG_REQUIRE_MANYPRED"):  # CI: this module must run, not skip
    import manypred
else:
    manypred = pytest.importorskip(
        "manypred", reason="ManyPred is an optional test dependency (put its src on PYTHONPATH)")
if "merge" not in inspect.signature(manypred.predict_array).parameters or not hasattr(
        manypred, "postprocess_blocks"):
    pytest.skip("the installed manypred has no merge= / postprocess_blocks "
                "(needs ManyPred's feature/vector-blocks)", allow_module_level=True)

import shapely  # noqa: E402
from manypred.windows import plan_windows  # noqa: E402
from manyseg.vectors import (  # noqa: E402
    Scene,
    block_decoder,
    decode_scene,
    field_targets,
    footprint_targets,
    pixel_to_world,
    rasterize_instances,
    road_graph,
    road_targets,
    roof_face_targets,
    to_pixels,
    world_to_pixel,
    write_records,
)
from rasterio.transform import Affine  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

CRS = "EPSG:28992"  # Amersfoort / RD New
PIXEL = 0.5  # m (E028's buildings; roofs use 0.25 m as E031 / E047)
# a non-integer origin, north-up (negative e): world = T * (x, y) with pixel corners at integers
ORIGIN = (155_123.37, 463_456.81)

# Oracle thresholds (perfect targets through the shipped decoders), not tuned to this test:
# * footprints B3 (corners): mean matched IoU >= 0.989 (vec/encoders round trip); mean PoLiS
#   <= 0.074 m (E028 level 0, corner heatmap B1 at 0.5 m, the worse test area);
# * fields B0 (outline): mean IoU >= 0.972 (vec/encoders round trip); PoLiS <= 0.237 m (E028
#   level 0, B0 outline decoding, the worse area);
# * roof faces: 0.991 (quoted for the vec/encoders round trip) is NOT reached by the oracle on
#   these faces even in one pass without tiles (0.972 mean at 0.25 m, 0.964 at 0.5 m): ties of
#   the distance target on the shared edges of touching faces move those edges by up to
#   1 px, and the acute tip of a hip-end face, cut off by the two walls' boundary bands, has no
#   seed of the face's own: it is flooded from a seed nearby, so the face can lose its tip
#   (single-pass face IoU down to 0.75 on the large scenes). Before min_seed (E053) a tip with a
#   few pixels above the seed threshold instead became an extra small face. Roofs use the
#   in-repo round trip's mean (``unit/test_vector_targets.py::test_roof_face_round_trip``:
#   mean > 0.93); its per-face 0.88 is for its three houses, so the per-face floor here is a
#   regression guard below the worst single-pass face measured on these scenes (0.75, an
#   acute tip), and there is no PoLiS bound (E028 has no face numbers). Per face, stitching is
#   held to the single pass (agreement below), exact with absolute distance and merge="cut";
# * roads R4: recall >= 0.999 and precision >= 1.0 of centreline length within 3 px, junctions
#   within 4 px (E028 / E044 oracle level 0; ``unit/test_vector_targets.py``);
# * per object: IoU >= ``min_iou`` (the round-trip tests' per-object floors), so one bad object
#   cannot hide in a mean; one-to-one: no missing, extra, split or merged object;
# * agreement of the stitched decode with the single pass (one whole-scene encoding, decoded
#   in one block; E050b's metric): every object matched with IoU >= 0.99.
THRESHOLDS = {
    "footprints": {"mean_iou": 0.989, "min_iou": 0.93, "polis_m": 0.074},
    "roofs": {"mean_iou": 0.93, "min_iou": 0.70, "polis_m": None},
    "fields": {"mean_iou": 0.972, "min_iou": 0.95, "polis_m": 0.237},
    "roads": {"recall": 0.999, "precision": 1.0, "tol_px": 3.0, "junction_px": 4.0},
}
ROAD_WIDTH = 6.0  # px


# --------------------------------------------------------------------- the ManyPred model

@dataclass
class Grid:
    """The scene and its tiling: shape, ManyPred windows, block grid for decoding."""

    shape: tuple[int, int]
    patch: int
    overlap: int
    block: int = 150
    halo: int = 48
    offset: tuple[int, int] = (37, 61)  # decode_scene's grid is shifted off the tiles
    shift: tuple[float, float] = (0.0, 0.0)  # scene content moved by (x, y) px
    pixel: float = PIXEL  # m

    @property
    def transform(self) -> Affine:
        """Pixel (x, y) -> world: a non-integer origin, north-up."""
        return Affine(self.pixel, 0.0, ORIGIN[0], 0.0, -self.pixel, ORIGIN[1])

    def __str__(self) -> str:
        return (f"{self.shape[0]}x{self.shape[1]}-p{self.patch}-o{self.overlap}"
                f"-s{self.shift[0]:g},{self.shift[1]:g}")

    @property
    def windows(self) -> list[tuple[int, int, int, int]]:
        return plan_windows(np.zeros(self.shape, bool), self.patch, self.overlap, True)

    def seams(self) -> tuple[list[float], list[float]]:
        """x and y of the cut lines between neighbouring windows (the overlap's middle)."""
        H, W = self.shape
        tops = sorted({w[0] for w in self.windows})
        lefts = sorted({w[2] for w in self.windows})
        ys = [(t + b) / 2 for t, b in zip(tops[1:], [t + self.patch for t in tops[:-1]])]
        xs = [(t + b) / 2 for t, b in zip(lefts[1:], [t + self.patch for t in lefts[:-1]])]
        return xs, ys

    def block_edges(self) -> tuple[list[int], list[int]]:
        H, W = self.shape
        oy, ox = (o % self.block for o in self.offset)
        return (list(range(ox or self.block, W, self.block)),
                list(range(oy or self.block, H, self.block)))


def tile_transform(transform, left: int, top: int) -> Affine:
    """The geotransform of the tile whose top-left pixel is scene (row ``top``, col
    ``left``): the scene's, moved by that many pixels."""
    a, b, c, d, e, f = transform[:6]
    return Affine(a, b, c + a * left + b * top, d, e, f + d * left + e * top)


class TileTargets(torch.nn.Module):
    """Perfect predictions: each patch's output is the task's targets encoded on that tile.

    Input: (B, 2, P, P) scene row / column of every pixel. The window is read from the first
    pixel and the whole patch is checked against it, so a patch ManyPred placed anywhere else
    than a planned window fails loudly."""

    def __init__(self, encode, grid: Grid):
        super().__init__()
        self.encode, self.grid = encode, grid
        self.planned = set(grid.windows)
        self.seen: dict = {}

    def forward(self, x):
        outs = []
        P = self.grid.patch
        for b in range(x.shape[0]):
            top, left = int(x[b, 0, 0, 0]), int(x[b, 1, 0, 0])
            win = (top, top + P, left, left + P)
            assert win in self.planned, f"ManyPred asked for an unplanned window {win}"
            ar = torch.arange(P, dtype=torch.float32)
            assert torch.equal(x[b, 0], (top + ar)[:, None].expand(P, P)), win
            assert torch.equal(x[b, 1], (left + ar)[None, :].expand(P, P)), win
            if win not in self.seen:
                tile = tile_transform(self.grid.transform, left, top)
                self.seen[win] = self.encode(tile, (P, P))
            t = {k: v for k, v in self.seen[win].items() if k != "instances"}
            t["owner"] = np.full((1, P, P), self.grid.windows.index(win), np.float32)
            outs.append(t)
        return {k: torch.from_numpy(np.stack([o[k] for o in outs])) for k in outs[0]}


def stitch(encode, grid: Grid, merge: str) -> tuple[dict, TileTargets]:
    """ManyPred's predict_array over the scene; returns the stitched fields and the model."""
    H, W = grid.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    model = TileTargets(encode, grid)
    names = [k for k in encode(grid.transform, (8, 8)) if k != "instances"]
    out = manypred.predict_array(
        model, np.stack([yy, xx]), patch_size=grid.patch, overlap=grid.overlap, batch_size=3,
        device="cpu", mosaic_device="cpu", mosaic_dtype="float32", dtype="float32",
        normalize=False, output="logits", merge={**{k: merge for k in names}, "owner": "cut"},
        progress=False)
    assert set(model.seen) == model.planned, "ManyPred skipped a planned window"
    return out, model


# --------------------------------------------------------------------- synthetic scenes

def _rect(cx, cy, w, h, angle=0.0):
    return shapely.affinity.rotate(shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                                   angle, origin=(cx, cy))


def _scatter(rng, shape, taken, n, make, gap=6.0):
    """Up to ``n`` items from ``make(rng) -> (geometry, item)`` whose geometry lies inside the
    scene and keeps ``gap`` px from every taken geometry (appended to ``taken``)."""
    H, W = shape
    out, tries = [], 0
    scene = shapely.box(0, 0, W, H)
    while len(out) < n and tries < 4000:
        tries += 1
        geom, item = make(rng)
        if not scene.contains(geom) or any(geom.distance(t) < gap for t in taken):
            continue
        taken.append(geom)
        out.append(item)
    return out


def _disjoint(hard, gap=6.0):
    """The hard cases in order, dropping any that comes within ``gap`` px of an earlier one
    (seam positions move with the tiling); the first one is always kept."""
    keep: list = []
    for g in hard:
        if all(g.distance(k) >= gap for k in keep):
            keep.append(g)
    return keep, list(keep)


def _footprint(faces3):
    return shapely.union_all([shapely.Polygon([v[:2] for v in f]) for f in faces3])


def _hot(grid: Grid):
    """Hard-case positions: the first cut seams and the first block edges (px)."""
    xs, ys = grid.seams()
    bx, by = grid.block_edges()
    return xs, ys, bx, by


def footprint_scene(grid: Grid, seed: int = 0):
    """Buildings (pixel coords): a courtyard building on a seam crossing, a rotated one across
    a column seam, an L across a row seam and a block edge, one cut by the scene edge, one on
    a block corner, then a random scatter (sizes 14-45 px, any angle)."""
    H, W = grid.shape
    xs, ys, bx, by = _hot(grid)
    sx, sy = xs[0], ys[0]
    court = _rect(sx, sy, 54, 42).difference(_rect(sx + 3, sy - 2, 18, 14))
    hard = [court,
            _rect(sx, sy + 62, 40, 22, 25),  # rotated, across a column seam
            shapely.union_all([_rect(bx[-1] + 10, ys[-1], 46, 16),  # an L across a row seam
                               _rect(bx[-1] + 26, ys[-1] + 14, 14, 40)]),  # and a block edge
            _rect(W - 8, sy + 40, 34, 24, 10),  # cut by the right scene edge
            _rect(bx[0] - 2, by[0] + 1, 30, 26, -15)]  # on a block corner
    polys, taken = _disjoint(hard)
    rng = np.random.default_rng(seed)

    def make(r):
        g = _rect(r.uniform(10, W - 10), r.uniform(10, H - 10), r.uniform(14, 45),
                  r.uniform(14, 35), r.uniform(0, 90))
        return g, g

    n = int(H * W / 3500)
    polys += _scatter(rng, grid.shape, taken, n, make)
    return polys


def _hip(cx, cy, w, h, angle, eave, rise):
    """Hip roof: two trapezoids and two triangles sharing edges, with 3-D vertices."""
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    r = h / 2
    a, b = (x0 + r, cy), (x1 - r, cy)
    z = eave + rise
    faces = [[(x0, y0, eave), (x1, y0, eave), (*b, z), (*a, z)],
             [(x1, y0, eave), (x1, y1, eave), (*b, z)],
             [(x1, y1, eave), (x0, y1, eave), (*a, z), (*b, z)],
             [(x0, y1, eave), (x0, y0, eave), (*a, z)]]
    return _rotate3(faces, angle, cx, cy)


def _gable(cx, cy, w, h, angle, eave, rise):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    z = eave + rise
    faces = [[(x0, y0, eave), (x1, y0, eave), (x1, cy, z), (x0, cy, z)],
             [(x0, cy, z), (x1, cy, z), (x1, y1, eave), (x0, y1, eave)]]
    return _rotate3(faces, angle, cx, cy)


def _rotate3(faces, angle, cx, cy):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return [[(cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy), z)
             for x, y, z in f] for f in faces]


def _plane(face3):
    """(a, b, c) with z = a x + b y + c (pixel coords) through a face's 3-D vertices."""
    p = np.asarray(face3, float)
    return np.linalg.lstsq(np.c_[p[:, :2], np.ones(len(p))], p[:, 2], rcond=None)[0]


def roof_scene(grid: Grid, seed: int = 0):
    """Roof faces (pixel coords) and their planes: a hip roof on a seam crossing, a gable whose
    ridge crosses a column seam, a hip on a block corner, then a random scatter of houses
    10-16 m long (sizes in metres, so in pixels at the grid's pixel size)."""
    H, W = grid.shape
    k = 1.0 / grid.pixel  # px per m
    xs, ys, bx, by = _hot(grid)
    sx, sy = xs[0], ys[0]
    hard = [_hip(sx, sy, 14 * k, 8.5 * k, 8, 4.0, 3.0),
            _gable(sx + 1 * k, sy + 15 * k, 11 * k, 7.5 * k, 0, 3.0, 2.5),
            _hip(bx[0], by[0], 11 * k, 7 * k, -20, 5.0, 2.0)]
    keep, taken = _disjoint([_footprint(h) for h in hard], gap=8.0)
    houses = [h for h in hard if any(_footprint(h).equals(g) for g in keep)]
    rng = np.random.default_rng(seed)

    def make(r):
        kind = _hip if r.random() < 0.5 else _gable
        w = r.uniform(10, 16) * k
        m = 7 * k
        h = kind(r.uniform(m, W - m), r.uniform(m, H - m), w, r.uniform(0.45, 0.65) * w,
                 r.uniform(0, 180), r.uniform(3, 6), r.uniform(1.5, 3.5))
        return _footprint(h), h

    houses += _scatter(rng, grid.shape, taken, int(H * W / (2250 * k * k)), make, gap=8.0)
    faces, planes, house = [], [], []
    for k, h in enumerate(houses):
        for f in h:
            faces.append(shapely.Polygon([v[:2] for v in f]))
            planes.append(_plane(f))
            house.append(k)
    return faces, np.array(planes), np.array(house)


def ndsm(faces, planes, shape):
    """nDSM (m): each face's plane at the pixel centres it covers, 0 on the ground."""
    lab, src = rasterize_instances(faces, shape)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    P = np.zeros((len(src) + 1, 3))
    P[1:] = planes[src]
    a, b, c = P[lab, 0], P[lab, 1], P[lab, 2]
    return np.where(lab > 0, a * xx + b * yy + c, 0.0).astype(np.float32)


def field_scene(grid: Grid, seed: int = 0, spacing: float = 80.0):
    """Fields (pixel coords): a jittered lattice of quads sharing edges (touching fields),
    extending past the scene (fields cut by its edge), a few shrunk apart by a 3 px track, and
    the field on the first seam crossing with a hole holding another field."""
    H, W = grid.shape
    xs, ys, *_ = _hot(grid)
    rng = np.random.default_rng(seed)
    gx = np.arange(-spacing, W + 2 * spacing, spacing)
    gy = np.arange(-spacing, H + 2 * spacing, spacing)
    jitter = rng.uniform(-0.2, 0.2, (len(gy), len(gx), 2)) * spacing
    pts = np.stack(np.meshgrid(gx + grid.shift[0], gy + grid.shift[1]), -1) + jitter
    scene = shapely.box(0, 0, W, H)
    fields = []
    for i in range(len(gy) - 1):
        for j in range(len(gx) - 1):
            q = shapely.Polygon([pts[i, j], pts[i, j + 1], pts[i + 1, j + 1], pts[i + 1, j]])
            if not q.is_valid:
                continue
            if rng.random() < 0.25:
                q = q.buffer(-1.5, join_style="mitre")
            q = q.intersection(scene)
            # no slivers along the scene edge (a field is at least 20 px wide)
            if q.geom_type == "Polygon" and not q.buffer(-10).is_empty:
                fields.append(q)
    # a 40 x 34 px hole holding another field, across a seam (nearest the first seam crossing
    # where it fits inside one field, 6 px from its edge, so it never cuts the field in two)
    off = np.arange(-15.0, 16.0, 3.0)  # the hole (about 45 px across) still crosses the seam
    cands = [(x + o, y) for x in xs for o in off for y in np.arange(0, H, 4.0)]
    cands += [(x, y + o) for y in ys for o in off for x in np.arange(0, W, 4.0)]
    cands.sort(key=lambda p: np.hypot(p[0] - xs[0], p[1] - ys[0]))
    for cx, cy in cands:
        hole = _rect(cx, cy, 40, 34, 12)
        host = next((k for k, f in enumerate(fields)
                     if f.buffer(-6, join_style="mitre").contains(hole)), None)
        if host is not None:
            break
    assert host is not None, "no room for the holed field"
    fields[host] = fields[host].difference(hole)
    fields.insert(host + 1, hole)  # the hole is another field
    fields = [f for f in fields if f.geom_type == "Polygon"]
    return fields


def road_scene(grid: Grid):
    """Road centrelines (pixel coords) as OSM-style ways sharing junction vertices: a junction
    exactly on a seam crossing, a T junction away from the seams, a dead end, an overpass
    crossing the main road 4 px from the second column seam (no shared vertex), roads running
    off the scene edges (clipped there: ends that are not road ends)."""
    H, W = grid.shape
    xs, ys, *_ = _hot(grid)
    sx, sy = xs[0], ys[0]
    x2 = xs[1] if len(xs) > 1 else W - 60.0
    J = (sx, sy)  # on both seams
    east = (W + 20.0, sy + 24.0)

    def on_main(x):  # main road east of J, a straight segment J -> east
        return sy + (east[1] - sy) * (x - sx) / (east[0] - sx)

    tj = (sx + 0.45 * (x2 - sx), on_main(sx + 0.45 * (x2 - sx)))
    main_w = [(-20.0, sy - 14.0), J]
    main_e = [J, tj, east]
    north = [(sx - 9.0, -20.0), J]
    south = [J, (sx + 6.0, sy + 0.45 * (H - sy)), (sx + 30.0, H - 30.0)]  # dead end inside
    xc = x2 + 4.0
    yc = on_main(xc)
    over = [(xc - 0.25 * (yc + 20.0), -20.0), (xc + 0.25 * (H + 20.0 - yc), H + 20.0)]
    branch = [tj, (tj[0] + 8.0, tj[1] - 0.5 * tj[1]), (tj[0] - 6.0, -20.0)]
    lines = [main_w, main_e, north, south, over, branch]
    return [shapely.LineString(ln) for ln in lines], {
        "seam_junction": J, "t_junction": tj, "dead_end": south[-1], "overpass": (xc, yc)}


# --------------------------------------------------------------------- encoders per task

@dataclass
class Options:
    """Encoder options measured against the defaults (see the module docstring)."""

    distance: str = "normalized"

    @property
    def label(self) -> str:
        return f"distance={self.distance}"


def _instance_fields(t: dict, corners: bool) -> dict:
    """The decoder's fields (float32 (C, h, w)) and ``instances`` (not predicted: kept for the
    invariance checks)."""
    out = {k: t[k][None].astype(np.float32) for k in ("extent", "boundary", "distance")}
    out["instances"] = t["instances"][None]
    if corners:
        out["corner"] = t["corner"][None].astype(np.float32)
        out["direction"] = t["direction"].astype(np.float32)
    return out


DEFAULT = Options()


def encoder(task: str, world, opt: Options = DEFAULT):
    """``encode(tile_transform, shape) -> {name: (C, h, w) float32}`` for one task."""
    if task == "footprints":
        def encode(tf, shape):
            polys = to_pixels(world, tf, shape, crs=CRS, raster_crs=CRS, kind="polygons")
            return _instance_fields(footprint_targets(polys, shape, distance=opt.distance),
                                   True)
    elif task == "roofs":
        def encode(tf, shape):
            polys = to_pixels(world, tf, shape, crs=CRS, raster_crs=CRS, kind="polygons")
            return _instance_fields(roof_face_targets(polys, shape, distance=opt.distance),
                                   True)
    elif task == "fields":
        def encode(tf, shape):
            polys = to_pixels(world, tf, shape, crs=CRS, raster_crs=CRS, kind="polygons")
            return _instance_fields(field_targets(polys, shape, distance=opt.distance), False)
    elif task == "roads":
        def encode(tf, shape):
            lines = to_pixels(world, tf, shape, crs=CRS, raster_crs=CRS, kind="lines")
            g, cr = road_graph(lines, width=ROAD_WIDTH)
            t = road_targets(g, shape, crossings=cr)
            return {"mask": t["mask"][None], "keypoints": t["keypoints"],
                    "direction": t["direction"]}
    else:
        raise ValueError(task)
    return encode


# --------------------------------------------------------------------- comparison

def _hausdorff_px(a, b, pixel: float) -> float:
    return float(shapely.hausdorff_distance(a.boundary, b.boundary, densify=0.1)) / pixel


def polis(a, b) -> float:
    """PoLiS distance (map units, as E028): the mean distance of each polygon's vertices to the
    other's boundary, averaged over both directions (all rings)."""
    def one(p, q):
        v = np.concatenate([np.asarray(r.coords)[:-1] for r in [p.exterior, *p.interiors]])
        return float(np.mean(shapely.distance(shapely.points(v), q.boundary)))

    return 0.5 * (one(a, b) + one(b, a))


@dataclass
class Report:
    task: str
    n_source: int = 0
    n_decoded: int = 0
    missing: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    split: list = field(default_factory=list)
    merged: list = field(default_factory=list)
    ious: dict = field(default_factory=dict)  # source index -> IoU
    hausdorff: dict = field(default_factory=dict)  # source index -> px
    polis: dict = field(default_factory=dict)  # source index -> map units
    vertex_delta: dict = field(default_factory=dict)  # decoded - source vertex count
    seam: set = field(default_factory=set)  # source indices crossing a cut seam
    holes_lost: list = field(default_factory=list)
    invalid: list = field(default_factory=list)

    def summary(self, which=None) -> dict:
        idx = [i for i in self.ious if which is None or (i in self.seam) == which]
        iou = np.array([self.ious[i] for i in idx]) if idx else np.zeros(0)
        hd = np.array([self.hausdorff[i] for i in idx]) if idx else np.zeros(0)
        vd = np.array([self.vertex_delta.get(i, 0) for i in idx]) if idx else np.zeros(0)

        def sel(lst):
            return [i for i in lst if which is None or (i in self.seam) == which]

        return {"n": len(idx), "mean_iou": float(iou.mean()) if len(iou) else float("nan"),
                "min_iou": float(iou.min()) if len(iou) else float("nan"),
                "max_hausdorff_px": float(hd.max()) if len(hd) else float("nan"),
                "mean_polis_m": float(np.mean([self.polis[i] for i in idx])) if idx
                else float("nan"),
                "vertex_delta_abs_mean": float(np.abs(vd).mean()) if len(vd) else float("nan"),
                "missing": len(sel(self.missing)), "split": len(sel(self.split)),
                "merged": len(sel(self.merged))}


def compare_polygons(task: str, source, decoded, seam, *, holes: bool,
                     pixel: float = PIXEL) -> Report:
    """One-to-one (Hungarian on IoU) matching of decoded to source polygons (map units).
    ``holes=False`` (roof faces, decoded as outer rings): compare with source exteriors."""
    src = [s if holes else shapely.Polygon(s.exterior) for s in source]
    rep = Report(task, len(src), len(decoded))
    tree = shapely.STRtree(decoded) if decoded else None
    iou = np.zeros((len(src), len(decoded)))
    for i, s in enumerate(src):
        for j in (tree.query(s, predicate="intersects") if tree is not None else []):
            d = decoded[j]
            iou[i, j] = s.intersection(d).area / s.union(d).area
    rows, cols = linear_sum_assignment(-iou) if iou.size else ([], [])
    matched_d = set()
    for i, j in zip(rows, cols):
        if iou[i, j] < 0.5:
            continue
        matched_d.add(j)
        rep.ious[i] = float(iou[i, j])
        rep.hausdorff[i] = _hausdorff_px(src[i], decoded[j], pixel)
        rep.polis[i] = polis(src[i], decoded[j])
        rep.vertex_delta[i] = (len(decoded[j].exterior.coords) - len(src[i].exterior.coords))
        if holes and len(source[i].interiors) and not len(decoded[j].interiors):
            rep.holes_lost.append(i)
    rep.missing = [i for i in range(len(src)) if i not in rep.ious]
    rep.extra = [j for j in range(len(decoded)) if j not in matched_d]
    # split: a source object covered by more than one decoded polygon (each > 10 % of it)
    for i, s in enumerate(src):
        parts = [j for j in (tree.query(s, predicate="intersects") if tree is not None else [])
                 if s.intersection(decoded[j]).area > 0.1 * s.area]
        if len(parts) > 1:
            rep.split.append(i)
    stree = shapely.STRtree(src)
    for j, d in enumerate(decoded):
        hits = [i for i in stree.query(d, predicate="intersects")
                if d.intersection(src[i]).area > 0.1 * src[i].area]
        if len(hits) > 1:
            rep.merged.append(j)
        if not (d.is_valid and d.geom_type == "Polygon" and d.exterior.is_closed):
            rep.invalid.append(j)
    if seam is not None:
        rep.seam = {i for i, s in enumerate(src) if s.intersects(seam)}
    return rep


def seam_zone(run, world: bool = True):
    """Where stitching acts: with ``merge="cut"`` the seams as ManyPred drew them (lines where
    the ``owner`` window changes), with ``"blend"`` the overlap zones (pixels two or more
    windows cover). World coordinates, or scene pixels with ``world=False``."""
    if run.merge == "cut":
        lines = []
        owner = run.owner
        for r, c in zip(*np.nonzero(owner[:, 1:] != owner[:, :-1])):
            lines.append([(c + 1, r), (c + 1, r + 1)])
        for r, c in zip(*np.nonzero(owner[1:, :] != owner[:-1, :])):
            lines.append([(c, r + 1), (c + 1, r + 1)])
        g = shapely.line_merge(shapely.MultiLineString(lines)) if lines else shapely.Point()
    else:
        boxes = [shapely.box(lft, t, r, b) for t, b, lft, r in run.grid.windows]
        g = shapely.union_all([a.intersection(b) for i, a in enumerate(boxes)
                               for b in boxes[i + 1:] if a.intersection(b).area > 0])
    return pixel_to_world(g, run.grid.transform) if world else g


# --------------------------------------------------------------------- diagnostics

def render(path, grid: Grid, source, decoded, bad=(), *, lines=False, title=""):
    """PNG overlay in scene pixels: source (blue), decoded (red), bad source objects (thick
    orange), ManyPred windows (grey), cut seams (black dashed), decode blocks (green)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(W / 40, H / 40), dpi=110)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    for t, b, lft, r in grid.windows:
        ax.plot([lft, r, r, lft, lft], [t, t, b, b, t], color="0.75", lw=0.5)
    xs, ys = grid.seams()
    for x in xs:
        ax.axvline(x, color="k", ls="--", lw=0.6)
    for y in ys:
        ax.axhline(y, color="k", ls="--", lw=0.6)
    bx, by = grid.block_edges()
    for x in bx:
        ax.axvline(x, color="g", lw=0.8)
    for y in by:
        ax.axhline(y, color="g", lw=0.8)

    def draw(g, **kw):
        g = world_to_pixel(g, grid.transform)
        for p in getattr(g, "geoms", [g]):
            rings = [p] if lines else [p.exterior, *p.interiors]
            for ring in rings:
                xy = np.asarray(ring.coords)
                ax.plot(xy[:, 0], xy[:, 1], **kw)

    for g in source:
        draw(g, color="tab:blue", lw=1.6)
    for g in decoded:
        draw(g, color="tab:red", lw=0.8)
    for g in bad:
        draw(g, color="tab:orange", lw=3, alpha=0.7)
    ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def dump_bad(path, geoms):
    fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": CRS}},
          "features": [{"type": "Feature", "properties": {"i": i},
                        "geometry": json.loads(shapely.to_geojson(g))}
                       for i, g in enumerate(geoms)]}
    path.write_text(json.dumps(fc))
    return path


# --------------------------------------------------------------------- the pipeline

@dataclass
class Run:
    """Everything one pipeline run produced (pixel geometry is scene pixels)."""

    task: str
    grid: Grid
    merge: str
    opt: Options
    source_px: list  # source geometry clipped to the scene
    world: list  # source as given to the encoders (world, unclipped)
    fields: dict  # stitched (without "owner")
    owner: np.ndarray
    whole_fields: dict  # one whole-scene encoding
    model: TileTargets
    records: object  # decode_scene on the offset block grid
    via_manypred: object  # manypred.postprocess_blocks on the stitched fields
    single: object  # whole-scene encoding decoded in one block (the single-pass reference)
    files: dict  # "geojson" / "gpkg" -> path
    extras: dict = field(default_factory=dict)


def _scene_box(grid):
    H, W = grid.shape
    return shapely.box(0, 0, W, H)


def source_px(task, grid, seed=0):
    """(pixel geometry, extras) of the task's synthetic scene, moved by ``grid.shift``."""
    if task == "footprints":
        geoms, extras = footprint_scene(grid, seed), {}
    elif task == "fields":
        geoms, extras = field_scene(grid, seed), {}
    elif task == "roofs":
        faces, planes, house = roof_scene(grid, seed)
        geoms, extras = faces, {"planes": planes, "house": house}
    elif task == "roads":
        geoms, pts = road_scene(grid)
        extras = {"points": pts}
    else:
        raise ValueError(task)
    dx, dy = grid.shift
    if (dx or dy) and task != "fields":
        geoms = [shapely.affinity.translate(g, dx, dy) for g in geoms]
        if task == "roofs":  # z = a x + b y + c moves with the face
            P = extras["planes"].copy()
            P[:, 2] -= P[:, 0] * dx + P[:, 1] * dy
            extras["planes"] = P
        if task == "roads":
            extras["points"] = {k: (v[0] + dx, v[1] + dy) for k, v in extras["points"].items()}
    return _inside(task, grid, geoms, extras)


def _inside(task, grid, geoms, extras):
    """Objects (partly) off the scene would leave slivers no decoder returns: keep buildings
    that lie inside the scene or keep most of themselves (half their area and a 12 px wide
    part) and houses wholly inside (fields are clipped as built)."""
    if task == "roads":
        return geoms, extras
    box = _scene_box(grid)

    def ok(g):
        c = g.intersection(box)
        return box.contains(g) or (c.area >= 0.5 * g.area and not c.buffer(-6).is_empty
                                   and c.geom_type == "Polygon")

    if task == "footprints":
        return [g for g in geoms if ok(g)], extras
    if task == "fields":  # the lattice was built shifted (field_scene)
        return geoms, extras
    # roofs: whole houses inside the scene only (a house's faces are joined)
    house = extras["house"]
    keep = [k for k in np.unique(house)
            if box.contains(shapely.union_all([geoms[i] for i in np.flatnonzero(house == k)]))]
    idx = np.flatnonzero(np.isin(house, keep))
    extras = {**extras, "planes": extras["planes"][idx], "house": house[idx]}
    return [geoms[i] for i in idx], extras


def pipeline(task: str, grid: Grid, merge: str, tmp, opt: Options = DEFAULT, seed: int = 0,
             write: bool = True) -> Run:
    H, W = grid.shape
    geoms, extras = source_px(task, grid, seed)
    world = [pixel_to_world(g, grid.transform) for g in geoms]
    enc = encoder(task, world, opt)
    fields, model = stitch(enc, grid, merge)
    owner = fields.pop("owner")[0]
    whole = enc(grid.transform, (H, W))
    if "instances" in whole:
        extras["instances"] = whole.pop("instances")[0]
    if task == "roofs":
        h = ndsm(geoms, extras["planes"], (H, W))[None]
        fields["height"] = whole["height"] = h
    # the fields decoder's seed threshold is in pixels (absolute distance, E052); the
    # normalised arm keeps E046's 0.3 of the instance maximum
    norm = task == "fields" and opt.distance == "normalized"
    dec = block_decoder(task, **({"extent_threshold": 0.6, "seed_threshold": 0.3} if norm else {}))
    scene = Scene(H, W, grid.transform, CRS)
    records = decode_scene(fields, dec, grid.block, grid.halo, scene=scene, offset=grid.offset)
    via = manypred.postprocess_blocks(
        fields, dec, combine=dec.combine, block=grid.block + 23, halo=grid.halo,
        scene=manypred.blocks.Scene(H, W, grid.transform, CRS))
    single = decode_scene(whole, dec, block=max(H, W), halo=0, scene=scene)
    one_block = decode_scene(fields, dec, block=max(H, W), halo=0, scene=scene)
    box = _scene_box(grid)
    clipped = []
    for g in geoms:
        c = g.intersection(box)
        clipped += [p for p in getattr(c, "geoms", [c]) if not p.is_empty
                    and (p.geom_type == "LineString" or p.area > 0)]
    files = {}
    if write:
        files = {"geojson": write_records(records, tmp / f"{task}.geojson", scene),
                 "gpkg": write_records(records, tmp / f"{task}.gpkg", scene)}
    extras["one_block"] = one_block
    return Run(task, grid, merge, opt, clipped, world, fields, owner, whole, model, records,
               via, single, files, extras)


def agreement(run: Run) -> Report:
    """The tiled decode against the single-pass decode (one whole-scene encoding decoded in
    one block): what stitching changed, objects in the seam zone apart (E050b's metric, on
    perfect targets). ``ious`` holds the IoU of each single-pass polygon with its tiled match."""
    single = [r["geometry"] for r in run.single]
    tiled = [r["geometry"] for r in run.records]
    return compare_polygons(run.task, single, tiled, seam_zone(run, world=False), holes=True,
                            pixel=run.grid.pixel)


def _covered(lines, other, tol: float, step: float) -> float:
    """Fraction of the length of ``lines`` within ``tol`` of ``other`` (sampled every
    ``step``; map units)."""
    target = shapely.MultiLineString([np.asarray(o.coords) for o in other]) if other else None
    pts = [ln.interpolate(s) for ln in lines for s in np.arange(0, ln.length, step)]
    if not pts:
        return 1.0
    if target is None:
        return 0.0
    return float(np.mean(shapely.distance(np.array(pts, dtype=object), target) <= tol))


def _graph_of(lines, snap: float):
    """networkx graph of polylines (map units) joined at end points within ``snap``."""
    g, _ = road_graph(lines, snap=snap)
    return g


def compare_roads(source_lines, decoded_lines, points: dict, grid: Grid) -> dict:
    """Roads in map units: centreline recall / precision within 3 px, junctions (degree >= 3)
    matched one-to-one within 4 px with their degree, the overpass not a junction, the dead
    end an end, connected components and total length."""
    import networkx as nx

    th = THRESHOLDS["roads"]
    px = grid.pixel
    tol, jtol = th["tol_px"] * px, th["junction_px"] * px
    out = {"recall": _covered(source_lines, decoded_lines, tol, px),
           "precision": _covered(decoded_lines, source_lines, tol, px)}
    gs, gd = _graph_of(source_lines, 1e-6), _graph_of(decoded_lines, 1e-6)
    js = [(gs.nodes[n]["xy"], gs.degree(n)) for n in gs if gs.degree(n) >= 3]
    jd = [(gd.nodes[n]["xy"], gd.degree(n)) for n in gd if gd.degree(n) >= 3]
    d = np.array([[np.hypot(*(a - b)) for b, _ in jd] for a, _ in js]).reshape(len(js), len(jd))
    rows, cols = linear_sum_assignment(d) if d.size else ([], [])
    pairs = [(i, j) for i, j in zip(rows, cols) if d[i, j] <= jtol]
    out["junctions"] = {"source": len(js), "decoded": len(jd), "matched": len(pairs),
                        "max_offset_px": max([d[i, j] / px for i, j in pairs], default=0.0),
                        "degree_ok": all(js[i][1] == jd[j][1] for i, j in pairs)}
    world = {k: pixel_to_world(np.asarray(v, float), grid.transform) for k, v in points.items()}
    jxy = np.array([p for p, _ in jd]).reshape(-1, 2)
    near = np.hypot(*(jxy - world["overpass"]).T) if len(jxy) else np.zeros(0)
    out["overpass_junction"] = bool((near < 10 * px).any())
    ends = np.array([gd.nodes[n]["xy"] for n in gd if gd.degree(n) == 1]).reshape(-1, 2)
    out["dead_end_px"] = (float(np.hypot(*(ends - world["dead_end"]).T).min()) / px
                          if len(ends) else float("inf"))
    out["components"] = (nx.number_connected_components(gs), nx.number_connected_components(gd))
    out["length_ratio"] = (sum(ln.length for ln in decoded_lines)
                           / sum(ln.length for ln in source_lines))
    return out


# --------------------------------------------------------------------- checks

def _same_records(a, b) -> bool:
    """Block decoders give identical output (ids, geometry, planes)."""
    if hasattr(a, "xy"):  # VectorGraph
        return np.array_equal(a.xy, b.xy) and np.array_equal(a.edges, b.edges)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x["id"] != y["id"] or not x["geometry"].equals_exact(y["geometry"], 0.0):
            return False
        if "plane" in x and not np.array_equal(x["plane"], y["plane"]):
            return False
    return True


def read_files(run: Run):
    """(GeoJSON geometries, GPKG GeoDataFrame); checks the CRS of both."""
    import geopandas as gpd
    from shapely.geometry import shape

    fc = json.loads(run.files["geojson"].read_text())
    assert fc["crs"]["properties"]["name"] == CRS
    gj = [shape(f["geometry"]) for f in fc["features"]]
    gdf = gpd.read_file(run.files["gpkg"])
    assert gdf.crs is not None and gdf.crs.to_epsg() == 28992, gdf.crs
    assert len(gdf) == len(gj)
    for a, b in zip(gj, gdf.geometry):
        assert a.equals_exact(b, 1e-6), "GeoJSON and GPKG disagree"
    # the attributes too: the same columns and values in both files
    props = [f["properties"] for f in fc["features"]]
    for col in gdf.columns.drop("geometry"):
        got = [p.get(col) for p in props]
        want = gdf[col].tolist()
        assert all(x == y or (isinstance(x, float) and np.isclose(x, y))
                   for x, y in zip(got, want)), f"GeoJSON and GPKG disagree on {col!r}"
    assert {k for p in props for k in p} == set(gdf.columns.drop("geometry"))
    return gj, gdf


class ChecksFailed(AssertionError):
    """Failed checks of one run; ``kinds`` names each failed check."""

    def __init__(self, msg, kinds):
        super().__init__(msg)
        self.kinds = set(kinds)


class Failures(list):
    """Failed checks of one run as (kind, message), written out with diagnostics before
    failing the test."""

    def check(self, ok, msg, kind="other"):
        if not ok:
            self.append((kind, msg))

    def finish(self, tmp, run: Run, source_w, decoded_w, bad, *, lines=False):
        if not self:
            return
        png = render(tmp / f"{run.task}_diagnostic.png", run.grid, source_w, decoded_w, bad,
                     lines=lines, title=f"{run.task} {run.merge} {run.opt.label} {run.grid}")
        gj = dump_bad(tmp / f"{run.task}_bad.geojson", bad)
        raise ChecksFailed("\n".join(f"[{k}] {m}" for k, m in self)
                           + f"\ndiagnostics: {png}, {gj}", [k for k, _ in self])


def check_polygons(run: Run, tmp, *, agree_iou: float = 0.99):
    th = THRESHOLDS[run.task]
    fails = Failures()
    # block decoding: the offset grid, ManyPred's grid and one block give the same records
    fails.check(_same_records(run.records, run.via_manypred),
                "decode_scene and manypred.postprocess_blocks differ", "blocks")
    fails.check(_same_records(run.records, run.extras["one_block"]),
                "block decoding differs from a one-block decode of the stitched fields",
                "blocks")
    ids = [r["id"] for r in run.records]
    fails.check(len(set(ids)) == len(ids), "duplicate ids")
    gj, gdf = read_files(run)
    fails.check(sorted(gdf["id"]) == sorted(ids), "ids not written")
    decoded = list(gdf.geometry)
    pix = [pixel_to_world(r["geometry"], run.grid.transform) for r in run.records]
    fails.check(all(a.equals_exact(b, 1e-6) for a, b in zip(decoded, pix)),
                "written geometry is not the records' through the scene transform")
    source = [pixel_to_world(g, run.grid.transform) for g in run.source_px]
    seams = seam_zone(run)
    holes = run.task != "roofs"  # roof faces join and mesh on their outer rings
    rep = compare_polygons(run.task, source, decoded, seams, holes=holes, pixel=run.grid.pixel)
    s = rep.summary()
    fails.check(not rep.missing, f"missing source objects {rep.missing}", "missing")
    fails.check(not rep.extra, f"extra decoded objects {rep.extra}", "extra")
    fails.check(not rep.split, f"split source objects {rep.split}", "split")
    fails.check(not rep.merged, f"decoded objects merging several sources {rep.merged}",
                "merged")
    fails.check(not rep.invalid, f"invalid / unclosed polygons {rep.invalid}")
    fails.check(not rep.holes_lost, f"holes lost {rep.holes_lost}", "holes")
    if holes:  # the holed field keeps its hole
        fails.check(sum(len(g.interiors) > 0 for g in decoded)
                    >= sum(len(g.interiors) > 0 for g in source), "a hole was filled", "holes")
    fails.check(s["mean_iou"] >= th["mean_iou"],
                f"mean IoU {s['mean_iou']:.4f} < {th['mean_iou']} (seam objects "
                f"{rep.summary(True)['mean_iou']:.4f}, others "
                f"{rep.summary(False)['mean_iou']:.4f})", "mean_iou")
    low = [i for i, v in rep.ious.items() if th["min_iou"] is not None and v < th["min_iou"]]
    fails.check(not low, f"objects under IoU {th['min_iou']}: "
                         f"{[(i, round(rep.ious[i], 3)) for i in low]}", "min_iou")
    fails.check(th["polis_m"] is None or s["mean_polis_m"] <= th["polis_m"],
                f"mean PoLiS {s['mean_polis_m']:.3f} m > {th['polis_m']} m (worst Hausdorff "
                f"{s['max_hausdorff_px']:.2f} px)", "polis")
    far = (sorted(rep.polis, key=rep.polis.get)[-3:]
           if th["polis_m"] is not None and s["mean_polis_m"] > th["polis_m"] else [])
    # stitching vs one whole-scene encoding decoded in one pass (E050b's agreement)
    agr = agreement(run)
    changed = [i for i, v in agr.ious.items() if v < agree_iou]
    fails.check(not (agr.missing or agr.extra or agr.split or agr.merged or changed),
                f"stitched decode differs from the single pass: changed {changed} "
                f"(seam: {sorted(set(changed) & agr.seam)}), missing {agr.missing}, "
                f"extra {agr.extra}, split {agr.split}, merged {agr.merged}", "agreement")
    if run.task == "roofs":
        _check_planes(run, gdf, rep, decoded, source, fails)
    bad = ([source[i] for i in rep.missing + rep.split + low + far]
           + [decoded[j] for j in rep.extra + rep.merged + rep.invalid])
    fails.finish(tmp, run, source, decoded, bad)
    return rep, agr


def _check_planes(run, gdf, rep, decoded, source, fails, tol_m: float = 0.05):
    """Fitted roof planes (scene pixel coordinates, metres) against the source planes at each
    matched face's centroid: the nDSM is exact, so only edge pixels the fit trims can move
    them; ``tol_m`` 5 cm."""
    planes = run.extras["planes"]
    cost = np.zeros((len(source), len(decoded)))
    for i, s in enumerate(source):
        for j, d in enumerate(decoded):
            if s.intersects(d):
                cost[i, j] = -s.intersection(d).area
    worst = 0.0
    for i, j in zip(*linear_sum_assignment(cost)):
        if cost[i, j] == 0:
            continue
        c = world_to_pixel(decoded[j].centroid, run.grid.transform)
        a, b, k = planes[i]  # houses lie inside the scene: source faces are not clipped
        z_src = a * c.x + b * c.y + k
        z_dec = gdf.plane_a[j] * c.x + gdf.plane_b[j] * c.y + gdf.plane_c[j]
        worst = max(worst, abs(z_src - z_dec))
    fails.check(worst <= tol_m, f"roof plane height error {worst:.3f} m > {tol_m} m")


def check_roads(run: Run, tmp):
    th = THRESHOLDS["roads"]
    fails = Failures()
    fails.check(_same_records(run.records, run.via_manypred),
                "decode_scene and manypred.postprocess_blocks differ")
    fails.check(_same_records(run.records, run.extras["one_block"]),
                "block decoding differs from a one-block decode of the stitched fields")
    gj, gdf = read_files(run)
    decoded = [g for g in gdf.geometry if g.geom_type == "LineString"]
    tf = run.grid.transform
    want = [shapely.LineString(pixel_to_world(p, tf)) for p in run.records.polylines()]
    fails.check(len(decoded) == len(want) and all(
        a.equals_exact(b, 1e-6) for a, b in zip(decoded, want)), "written lines differ")
    source = [pixel_to_world(g, run.grid.transform) for g in run.source_px]
    m = compare_roads(source, decoded, run.extras["points"], run.grid)
    fails.check(m["recall"] >= th["recall"], f"recall {m['recall']:.4f} < {th['recall']}")
    fails.check(m["precision"] >= th["precision"],
                f"precision {m['precision']:.4f} < {th['precision']}", "precision")
    j = m["junctions"]
    fails.check(j["source"] == j["decoded"] == j["matched"] and j["degree_ok"],
                f"junctions {j}")
    fails.check(not m["overpass_junction"], "the overpass became a junction")
    fails.check(m["dead_end_px"] <= th["junction_px"], f"dead end off by {m['dead_end_px']} px")
    fails.check(m["components"][0] == m["components"][1], f"components {m['components']}",
                "components")
    fails.check(abs(m["length_ratio"] - 1) <= 0.01, f"length ratio {m['length_ratio']:.4f}",
                "length")
    single = [shapely.LineString(pixel_to_world(p, tf)) for p in run.single.polylines()]
    px = run.grid.pixel
    agree = (_covered(decoded, single, 0.5 * px, px), _covered(single, decoded, 0.5 * px, px))
    fails.check(min(agree) == 1.0, f"stitched graph differs from the single pass {agree}")
    bad = []
    fails.finish(tmp, run, source, decoded, bad, lines=True)
    return m


# --------------------------------------------------------------------- tile invariance

def _interior_edges(grid: Grid) -> np.ndarray:
    """(H, W) bool: the pixels on either side of every window edge inside the scene."""
    H, W = grid.shape
    e = np.zeros((H, W), bool)
    for t, b, lft, r in grid.windows:
        for x in (lft, r):
            if 0 < x < W:
                e[t:b, x - 1 : x + 1] = True
        for y in (t, b):
            if 0 < y < H:
                e[y - 1 : y + 1, lft:r] = True
    return e


def _clipped(run: Run) -> list:
    """Per window: ``(mask, dist)``, the (H, W) pixels inside the window of the instances it
    clips (with pixels outside it or on its edge pixels), and each pixel's distance to the
    window's interior edges (inf where it has none). From the whole-scene instance map: where
    that tile's encoding can see an instance differently."""
    lab = run.extras["instances"]
    H, W = lab.shape
    out = []
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    for t, b, lft, r in run.grid.windows:
        inside = np.zeros((H, W), bool)
        inside[t:b, lft:r] = True
        # a polygon can cross an edge without a pixel centre beyond it (a corner 0.01 px
        # past the edge is clipped too): count instances reaching the edge's last pixel
        core = np.zeros((H, W), bool)
        core[t + (t > 0) : b - (b < H), lft + (lft > 0) : r - (r < W)] = True
        ids = np.intersect1d(np.unique(lab[inside]), np.unique(lab[~core]))
        ids = ids[ids > 0]
        dist = np.full((H, W), np.inf)
        for edge, limit, d in ((t, 0, yy - t), (b, H, b - yy), (lft, 0, xx - lft),
                               (r, W, r - xx)):
            if edge != limit:
                dist = np.minimum(dist, d)
        out.append((inside & np.isin(lab, ids), dist))
    return out


def _supplying(run: Run) -> list:
    """Per window, the (H, W) pixels whose stitched value it contributes to: its owner pixels
    with ``merge="cut"``, all it covers with ``"blend"``."""
    H, W = run.grid.shape
    out = []
    for i, (t, b, lft, r) in enumerate(run.grid.windows):
        if run.merge == "cut":
            m = run.owner == i
        else:
            m = np.zeros((H, W), bool)
            m[t:b, lft:r] = True
        out.append(m)
    return out


def invariance(run: Run, band_px: float = 7.0) -> dict:
    """Where the stitched targets differ from one whole-scene encoding (``diffs``), and the
    zones where they may: ``band`` (pixels within ``band_px`` of an interior window edge: an
    object clipped by the tile gets corners, walls, keypoints and a border there; a Gaussian
    reaches 3 sigma = 6 px), ``corners`` (direction where two walls' bands overlap at a
    corner: the wall drawn last wins and clipping changes the ring's first vertex) and, for
    instance tasks, over the windows that supply a pixel and clip its instance: ``clipped``
    (those pixels) and ``far`` (those whose whole-scene distance reaches that window's edge,
    the only ones where an absolute distance can change)."""
    from scipy import ndimage

    edges = _interior_edges(run.grid)
    out = {"band": ndimage.distance_transform_edt(~edges) <= band_px}
    if "corner" in run.whole_fields:
        out["corners"] = ndimage.binary_dilation(run.whole_fields["corner"][0] >= 0.5,
                                                 iterations=2)
    if "instances" in run.extras:
        d_whole = run.whole_fields["distance"][0]
        clipped = np.zeros(run.grid.shape, bool)
        far = np.zeros(run.grid.shape, bool)
        for (mask, dist), supply in zip(_clipped(run), _supplying(run)):
            m = mask & supply
            clipped |= m
            # absolute distance = 1 + distance to the nearest edge pixel; the tile misses the
            # edge pixels beyond its edge and has none along it, all >= dist - 1 px away
            far |= m & (d_whole + 1 >= dist)
        out["clipped"], out["far"] = clipped, far
    out["diffs"] = {k: np.abs(v - run.whole_fields[k]).max(0) > 1e-5
                    for k, v in run.fields.items()}
    return out


def check_invariance(run: Run):
    """Stitched targets vs one whole-scene encoding: equal except where :func:`invariance`
    allows. With ``merge="cut"`` and overlap / 2 above the band, the band is never stitched
    in (each pixel comes from the window it is deepest in)."""
    from scipy.ndimage import binary_dilation

    inv = invariance(run)
    diffs, band = inv["diffs"], inv["band"]
    cut_drops_band = run.merge == "cut" and run.grid.overlap / 2 > 7.0
    allowed_any = np.zeros_like(band) if cut_drops_band else band
    for k, d in diffs.items():
        allowed = allowed_any.copy()
        if k == "direction" and "corners" in inv:
            # at corners (and blended around them), and on the walls of a polygon the
            # supplying tile clips: corner_targets simplifies the clipped ring
            # (Douglas-Peucker, 1.5 px), which can keep other vertices than the whole ring
            allowed |= inv["corners"] | binary_dilation(inv["clipped"], iterations=2)
        if k == "distance":
            # normalised: the clipped instance's maximum differs, so all of it may;
            # absolute: only where its nearest edge can lie beyond the tile
            allowed |= inv["clipped"] if run.opt.distance == "normalized" else inv["far"]
        bad = d & ~allowed
        assert not bad.any(), (
            f"{run.task} {run.merge} {run.opt.label}: '{k}' differs from the whole-scene "
            f"encoding at {int(bad.sum())} unexpected pixels, e.g. (row, col) "
            f"{tuple(int(v) for v in np.argwhere(bad)[0])}")
    return {k: int(d.sum()) for k, d in diffs.items()}


# --------------------------------------------------------------------- tests

FAST = Grid((290, 410), 128, 32)
TASKS = ["footprints", "fields", "roofs", "roads"]

# Measured seam errors (``seam_study``, 700 x 1000 px, tiles 256 / 32-128 and 512 / 256): the
# default per-tile normalised distance makes the stitched distance jump at seams. Fields: 3-34
# of 32-121 seam-zone fields changed with cut, 8-30 with blend, plus splits / merges / missing /
# extra pieces (agreement with the single pass down to 0.51); distance="absolute": 0 changed
# with cut, at most 1 (0.999) with blend, mean IoU with the source 0.986-0.988 (default
# 0.951-0.991). Roof faces (0.25 m): cut exact for both; blend changes 2-39 faces with the
# default, 0-9 with absolute (worst 0.925; touching faces' shared edges follow distance ties).
# Footprints: no change with cut; with blend at most one building (0.997). A tried option,
# corner_targets(tile_edges="ignore") (no corners or walls on clip edges, left out of the
# corner loss), changed nothing measurable and was dropped (the corners decoder merges a clip
# vertex on a straight wall, and cut never stitches in a clip edge once overlap / 2 > 7 px), and
# cost a little IoU on buildings cut by the scene edge (no corners there).
NORMALISED_SEAMS = ("normalised distance on fields: seam errors (the stitched distance jumps "
                    "where tiles normalise an instance differently: fields split, merged or "
                    "reshaped at seams) and, in the single pass too, seeds lost in narrow parts "
                    "(seed threshold 0.3 of the instance maximum: extra pieces); on the large "
                    "scenes the holed field loses its hole and the block decode differs from a "
                    "one-block decode and from postprocess_blocks (the distance jumps let "
                    "the watershed reach past the halo); distance='absolute' removes all of it")
NORMALISED_ROOFS = ("per-tile normalised distance: roof faces in the overlap reshaped under "
                    "blend; distance='absolute' removes it")


# Failures that are not stitching: the single pass (no tiles) fails the same way.
# Fixed by min_seed (E053; block_decoder default 8): seed fragments of 1-6 px where boundary bands
# meet (the 45-degree tip of a hip-end face, a 1 px seed on the 3 px threshold's edge at a
# field corner) grew into extra 13-25 px objects; they are now flooded by the object around them.
NARROW_STRIP = ("decoder limit, also in the single pass: with a 3 px seed threshold the narrow "
                "strip of the holed field (under ~8 px wide) gets broken seeds, so the field "
                "splits along it and loses its hole")

# (task, merge, distance, str(grid)) that fail for these reasons
_ORACLE = {
    **{("fields", m, "absolute", "700x1000-p512-o256-s-31.5,23.25"): NARROW_STRIP
       for m in ("cut", "blend")},
}


def _expected_failure(task, distance, merge, grid):
    """Parametrisations that fail for a measured, explained reason (xfail, strict, so they
    flip when the defaults or the decoders change)."""
    if (task, merge, distance, str(grid)) in _ORACLE:
        return _ORACLE[task, merge, distance, str(grid)]
    if distance != "normalized":
        return None
    if task == "fields":
        return NORMALISED_SEAMS
    if task == "roofs" and merge == "blend" and grid.shape != (290, 410):
        return NORMALISED_ROOFS
    return None


def grid_for(task: str, grid: Grid) -> Grid:
    """Roofs at 0.25 m (E031 / E047 tiles; hip ends of 3-4 m houses are too small at 0.5 m
    for the roof oracle's IoU), every other task at 0.5 m (E028)."""
    return replace(grid, pixel=0.25) if task == "roofs" else grid


class ExpectedFailure(AssertionError):
    """A check failed in the way an xfail's reason says (any other failure fails the test)."""


# what the failure message of each expected failure must say
# the only checks each expected failure may fail; any other failed check fails the test
_ALLOWED = {NORMALISED_SEAMS: {"agreement", "extra", "split", "merged", "missing", "mean_iou",
                               "min_iou", "polis", "holes", "blocks"},
            NORMALISED_ROOFS: {"agreement", "extra", "split", "merged", "missing", "mean_iou"},
            NARROW_STRIP: {"extra", "split", "holes", "min_iou", "mean_iou", "polis"}}


def _params(grids):
    out = []
    for base in grids:
        for task in TASKS:
            grid = grid_for(task, base)
            for merge in ("cut", "blend"):
                for distance in (("normalized",) if task == "roads"
                                 else ("normalized", "absolute")):
                    marks = []
                    reason = _expected_failure(task, distance, merge, grid)
                    if reason:
                        marks.append(pytest.mark.xfail(strict=True, reason=reason,
                                                       raises=ExpectedFailure))
                    out.append(pytest.param(task, merge, distance, grid, marks=marks,
                                            id=f"{task}-{merge}-{distance}-{grid}"))
    return out


def _run_checks(task, merge, distance, grid, tmp_path):
    run = pipeline(task, grid, merge, tmp_path, Options(distance=distance))
    check_invariance(run)
    reason = _expected_failure(task, distance, merge, grid)
    try:
        if task == "roads":
            check_roads(run, tmp_path)
        else:
            check_polygons(run, tmp_path)
    except ChecksFailed as e:
        if reason and e.kinds <= _ALLOWED[reason]:
            raise ExpectedFailure(str(e)) from None
        raise


@pytest.mark.parametrize("task, merge, distance, grid", _params([FAST]))
def test_pipeline(task, merge, distance, grid, tmp_path):
    """CRS vectors -> per-tile targets -> ManyPred (cut / blend) -> block decoding -> GeoJSON /
    GPKG -> the source vectors again, on a small scene (CPU, seconds)."""
    _run_checks(task, merge, distance, grid, tmp_path)


SLOW = [Grid((700, 1000), 256, 32, block=333, halo=64, offset=(101, 157)),
        Grid((700, 1000), 256, 128, block=333, halo=64, offset=(101, 157), shift=(17.3, -9.6)),
        Grid((700, 1000), 512, 256, block=400, halo=64, offset=(0, 211), shift=(-31.5, 23.25))]


@pytest.mark.slow
@pytest.mark.parametrize("task, merge, distance, grid", _params(SLOW))
def test_pipeline_large(task, merge, distance, grid, tmp_path):
    """The same on a larger scene over tile size, overlap and scene offset."""
    _run_checks(task, merge, distance, grid, tmp_path)


# --------------------------------------------------------------------- seam study (manual)

def seam_study(shape=(700, 1000), configs=((256, 32), (256, 64), (256, 128), (512, 256)),
               tasks=("footprints", "fields", "roofs"), tmp=None):
    """Seam errors of each encoder option over tile overlap and merge: per run, the objects
    of the single pass in the seam zone and how many stitching changed (IoU < 0.999), splits /
    merges / missing / extra, and the mean IoU with the source in and out of the seam zone.
    Run by hand: ``python test_vector_pipeline.py`` (minutes on a CPU)."""
    import tempfile
    from pathlib import Path

    opts = {"normalized": Options(), "absolute": Options(distance="absolute")}
    tmp = Path(tmp or tempfile.mkdtemp())
    rows = []
    for task in tasks:
        for (patch, overlap), merge, name in [(c, m, o) for c in configs
                                              for m in ("cut", "blend") for o in opts]:
            grid = grid_for(task, Grid(shape, patch, overlap, block=333, halo=64,
                                       offset=(101, 157)))
            run = pipeline(task, grid, merge, tmp, opts[name], write=False)
            a = agreement(run)
            src = compare_polygons(task, run.source_px, [r["geometry"] for r in run.records],
                                   seam_zone(run, world=False), holes=task != "roofs",
                                   pixel=grid.pixel)
            row = {"task": task, "patch": patch, "overlap": overlap, "merge": merge,
                   "option": name, "objects": len(run.single), "in_seam_zone": len(a.seam),
                   "changed_seam": sum(1 for i, v in a.ious.items() if i in a.seam and v < 0.999),
                   "changed_other": sum(1 for i, v in a.ious.items()
                                        if i not in a.seam and v < 0.999),
                   "split": len(a.split), "merged": len(a.merged), "missing": len(a.missing),
                   "extra": len(a.extra), "min_agreement": a.summary()["min_iou"],
                   "iou_seam": src.summary(True)["mean_iou"],
                   "iou_other": src.summary(False)["mean_iou"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    seam_study()
