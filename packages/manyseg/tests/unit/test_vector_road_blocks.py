"""RoadBlockDecoder: a block-wise decode (core + halo) equals graph_from_road_heads on the scene."""

import json
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")
from manyseg.vectors.roads import (  # noqa: E402
    RoadBlockDecoder,
    graph_from_road_heads,
)

FIXTURES = Path(os.environ.get("MANYSEG_E051_FIELDS", "e051_fields"))


@dataclass(frozen=True)
class _Block:  # duck-typed as manypred.blocks.Block
    index: int
    core: tuple
    window: tuple
    scene_edges: tuple


@dataclass(frozen=True)
class _Scene:
    height: int
    width: int
    transform: object = None
    crs: object = None


def _grid(H, W, block, halo, offset=(0, 0)):
    def cuts(n, o):
        return sorted({0, n, *range(o % block or block, n, block)})

    rs, cs = cuts(H, offset[0]), cuts(W, offset[1])
    out = []
    for t, b in zip(rs[:-1], rs[1:]):
        for left, right in zip(cs[:-1], cs[1:]):
            w = (max(t - halo, 0), min(b + halo, H), max(left - halo, 0), min(right + halo, W))
            edges = (w[0] == 0, w[1] == H, w[2] == 0, w[3] == W)
            out.append(_Block(len(out), (t, b, left, right), w, edges))
    return out


def _decode(fields, dec, block, halo, offset=(0, 0), results=None, strict=True):
    """In-memory driver as ManyPred's postprocess_blocks (with a pickle round trip, as workers).
    ``strict``: the halo-reach warning is an error."""
    H, W = next(iter(fields.values())).shape[-2:]
    scene = _Scene(H, W)
    dec = pickle.loads(pickle.dumps(dec))
    res = [] if results is None else results
    for b in _grid(H, W, block, halo, offset):
        wf = {k: np.asarray(v[:, b.window[0] : b.window[1], b.window[2] : b.window[3]])
              for k, v in fields.items()}
        res.append(pickle.loads(pickle.dumps(dec(wf, b, scene))))
    with warnings.catch_warnings():
        if strict:
            warnings.simplefilter("error")
        return dec.combine(res, scene)


def _same(a, b):
    return np.array_equal(a.xy, b.xy) and np.array_equal(a.edges, b.edges)


def _seg_dist(xx, yy, p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    v = q - p
    t = np.clip(((xx - p[0]) * v[0] + (yy - p[1]) * v[1]) / max(v @ v, 1e-9), 0, 1)
    return np.hypot(xx - p[0] - t * v[0], yy - p[1] - t * v[1]), math.atan2(v[1], v[0])


def _scene(seed=0, H=384, W=448, quantize=True):
    """Roads of width 4-9 px: through roads crossing the 128-px seams and corners, a junction on
    a block corner, an overpass on a seam, dead ends, a gap across a seam, roads leaving the
    scene, a short spur on a seam; keypoint heatmaps and the two-mode direction field to match.
    Noise and 2-decimal quantisation make plateaus (tied peaks) and ragged skeletons."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    roads = [  # polyline, half width
        ([(0, 128), (W, 128)], 3.5),  # along a seam row
        ([(128, 0), (128, H)], 3.0),  # along a seam column; junction on the corner (128, 128)
        ([(20, 300), (420, 250)], 4.0),
        ([(256, 20), (262, 370)], 2.5),  # crosses the previous road near the x = 256 seam
        ([(300, 40), (330, 120), (440, 160)], 3.0),  # leaves the scene on the right
        ([(60, 200), (60, 236)], 2.5),  # a dead end and, past the y = 256 seam, ...
        ([(60, 270), (65, 360)], 2.5),  # ... a spur pruned at its junction
        ([(300, 185), (370, 185)], 2.5),  # stops at a gap that spans the x = 384 seam ...
        ([(398, 185), (440, 185)], 2.5),  # ... and continues
        ([(200, 60), (215, 128)], 2.0),  # short spur ending on the seam row
        ([(380, 300), (447.5, 383.5)], 3.0),  # runs into the scene corner
    ]
    for _ in range(4):
        a, b = rng.uniform((0, 0), (W, H)), rng.uniform((0, 0), (W, H))
        roads.append(([tuple(a), tuple(b)], rng.uniform(2, 4.5)))
    mask = np.zeros((H, W), np.float32)
    d = np.zeros((4, H, W), np.float32)
    best = np.full((H, W), np.inf)
    for pts, hw in roads:
        for p, q in zip(pts[:-1], pts[1:]):
            dist, th = _seg_dist(xx, yy, p, q)
            on = dist <= hw
            mask[on] = np.maximum(mask[on], 0.9)
            first = on & (best < np.inf)  # second mode where two roads overlap
            d[2][first], d[3][first] = math.cos(2 * th), math.sin(2 * th)
            near = dist < best
            best[near] = dist[near]
            d[0][near], d[1][near] = math.cos(2 * th), math.sin(2 * th)
    kp = np.zeros((3, H, W), np.float32)
    keys = {
        0: [(128, 128), (60, 128), (300, 128), (128, 300), (420.5, 128)],  # junctions, on seams
        1: [(60, 200), (60, 270), (200, 60), (20, 300), (370, 185), (398, 185)],  # ends
        2: [(259.5, 270), (256.4, 128)],  # crossings near / on the x = 256 seam
    }
    for c, pts in keys.items():
        for x, y in pts:
            kp[c] = np.maximum(kp[c], 0.8 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 8))
    mask += rng.normal(0, 0.08, mask.shape).astype(np.float32)
    kp += rng.normal(0, 0.03, kp.shape).astype(np.float32)
    d += rng.normal(0, 0.05, d.shape).astype(np.float32)
    mask[rng.random(mask.shape) < 0.002] = 0.9  # speckle: small-blob removal
    if quantize:
        mask, kp = np.round(mask, 2), np.round(kp, 2)
    # tied junction peaks across the x = 256 seam, the later one in raster order in the right
    # block: suppression keeps (126, 256), as a whole-scene decode, not the left block's first
    kp[0, 126, 256] = kp[0, 127, 255] = 0.75
    # a strong peak off the road suppresses a weaker one on it, then is dropped itself
    kp[0, 133, 350], kp[0, 130, 350] = 0.9, 0.7
    return {"mask": np.clip(mask, 0, 1)[None], "keypoints": np.clip(kp, 0, 1), "direction": d}


def _whole(f, **p):
    return graph_from_road_heads(f["mask"][0], f["keypoints"], f["direction"], **p)


GRIDS = [(128, 32, (0, 0)), (96, 29, (0, 0)), (160, 40, (37, 91)), (100, 30, (64, 0)),
         (512, 32, (0, 0))]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blocks_equal_whole_scene(seed):
    f = _scene(seed)
    p = dict(threshold=0.5, gap=50)
    ref = _whole(f, **p)
    assert len(ref.junctions()) >= 3 and len(ref.edges) > 30  # the scene is not trivial
    for block, halo, off in GRIDS:
        got = _decode(f, RoadBlockDecoder(**p), block, halo, off)
        assert _same(got, ref), (seed, block, halo, off, len(got.edges), len(ref.edges))


def test_scene_exercises_bridging_and_crossings():
    f = _scene(0)
    a, b = _whole(f, threshold=0.5, gap=50), _whole(f, threshold=0.5, gap=0)
    assert len(a.edges) > len(b.edges)  # the gap across the seam is bridged
    got = _decode(f, RoadBlockDecoder(threshold=0.5, gap=0), 128, 32)
    assert _same(got, b)
    kp0 = {**f, "keypoints": np.zeros_like(f["keypoints"])}  # no keypoints: plain skeleton
    assert _same(_decode(kp0, RoadBlockDecoder(threshold=0.5), 128, 32),
                 _whole(kp0, threshold=0.5))


def test_parameters_and_names():
    f = _scene(1, quantize=False)
    p = dict(threshold=0.4, keypoint_threshold=0.3, zone_radius=12.0, pair_deg=140.0, spur=15,
             gap=30, simplify=1.0, min_length=10, min_area=60)
    ref = _whole(f, **p)
    g = {"m": f["mask"], "kp": f["keypoints"], "dir": f["direction"]}
    dec = RoadBlockDecoder(**p, names={"mask": "m", "keypoints": "kp", "direction": "dir"})
    for block, halo in ((128, 59), (96, 64)):
        assert _same(_decode(g, dec, block, halo), ref), (block, halo)


def test_logits():
    """logits=True: sigmoid on mask and keypoints (not direction), whole scene and per block."""
    import torch

    f = _scene(0, quantize=False)
    ref = _whole(f, threshold=0.5, gap=50)
    lg = {k: (v if k == "direction" else
              torch.logit(torch.from_numpy(np.clip(v, 1e-4, 1 - 1e-4).astype(np.float32))).numpy())
          for k, v in f.items()}
    back = {k: (v if k == "direction" else torch.sigmoid(torch.from_numpy(v)).numpy())
            for k, v in lg.items()}  # what logits=True decodes
    exp = _whole(back, threshold=0.5, gap=50)
    assert len(exp.edges) > 0.9 * len(ref.edges)
    assert _same(_whole(lg, threshold=0.5, gap=50, logits=True), exp)
    dec = RoadBlockDecoder(threshold=0.5, gap=50, logits=True)
    assert dec.params["logits"] is True and "logits" in repr(dec)
    assert _same(_decode(lg, dec, 128, 32), exp)
    # raw logits read as probabilities: everything above logit 0.5 is road, a different graph
    assert not _same(_whole(lg, threshold=0.5, gap=50), exp)


def test_empty_scene_and_empty_blocks():
    z = {"mask": np.zeros((1, 200, 300), np.float32), "keypoints": np.zeros((3, 200, 300)),
         "direction": np.zeros((4, 200, 300))}
    g = _decode(z, RoadBlockDecoder(), 64, 29)
    assert g.xy.shape == (0, 2) and g.edges.shape == (0, 2)
    assert len(RoadBlockDecoder().combine([], _Scene(10, 10)).edges) == 0
    z["mask"][0, 90:96, 10:290] = 0.9  # one road; most blocks see nothing
    z["keypoints"] = z["keypoints"].astype(np.float32)
    assert _same(_decode(z, RoadBlockDecoder(), 64, 29), _whole(z))


def test_small_halo_raises():
    f = _scene(0)
    with pytest.raises(ValueError, match="halo"):
        _decode(f, RoadBlockDecoder(), 128, 10)


def _blank(H, W):
    return {"mask": np.zeros((1, H, W), np.float32), "keypoints": np.zeros((3, H, W), np.float32),
            "direction": np.zeros((4, H, W), np.float32)}


def test_small_blob_across_a_seam():
    """Codex 10-29 #1: a 50-px, 1-px-wide blob centred on a seam is seen cut by both windows;
    with min_area 100 the halo must be >= 99 so neither window keeps it as 'maybe large'."""
    f = _blank(400, 400)
    f["mask"][0, 100:300, 60] = 0.9  # a road, kept
    f["mask"][0, 200, 103:153] = 0.9  # 50 px across the x = 128 seam, removed
    ref = _whole(f, min_area=100, gap=0)
    assert len(ref.edges) == 1
    with pytest.raises(ValueError, match="min_area"):
        _decode(f, RoadBlockDecoder(min_area=100, gap=0), 128, 40)
    assert _same(_decode(f, RoadBlockDecoder(min_area=100, gap=0), 128, 99), ref)


def _car_park(H=320, W=320):
    rng = np.random.default_rng(1)
    f = _blank(H, W)
    m = f["mask"][0]
    m[150:156, :] = 0.9
    m[:, 100:105] = 0.9
    m[130:170, 135:200] = 0.9  # 40 x 65 px across the seams at 160
    m[rng.random(m.shape) < 0.01] = 0  # parked cars: holes make thinning reach far
    return f


def test_wide_blob_warns_when_halo_too_small():
    f = _car_park()
    p = dict(threshold=0.5, min_area=1)
    ref = _whole(f, **p)
    with pytest.warns(UserWarning, match="halo"):
        _decode(f, RoadBlockDecoder(**p), 160, 24, strict=False)
    assert _same(_decode(f, RoadBlockDecoder(**p), 160, 100), ref)


def test_dense_peaks_memory_is_bounded():
    """Codex 10-29 #3: an untrained keypoint head (a local maximum every few px) costs at most
    the window's pixels in direction samples, and still decodes as the whole scene."""
    rng = np.random.default_rng(0)
    f = _blank(260, 300)
    for r in (60, 120, 125, 180):
        f["mask"][0, r : r + 6, 10:290] = 0.9
    f["mask"][0, 20:240, 100:106] = f["mask"][0, 20:240, 200:208] = 0.9
    f["keypoints"][:] = rng.random(f["keypoints"].shape).astype(np.float32)
    f["direction"][:] = rng.normal(0, 1, f["direction"].shape).astype(np.float32)
    res = []
    got = _decode(f, RoadBlockDecoder(gap=0), 128, 29, results=res)
    for r, b in zip(res, _grid(260, 300, 128, 29)):
        area = (b.window[1] - b.window[0]) * (b.window[3] - b.window[2])
        keys = r["direction"][0]
        assert len(keys) <= area and np.all(np.diff(keys) > 0)
    assert _same(got, _whole(f, gap=0))


def _fixture(name):
    meta = json.loads((FIXTURES / "meta.json").read_text())["_decoder_params"]
    r0, r4 = meta["roads_r0"], meta["roads_r4"]
    p = dict(threshold=r0["thr"], keypoint_threshold=r4["thr_k"], zone_radius=r4["zr"],
             pair_deg=r4["pair_deg"], spur=r0["spur"], gap=r0["gap"],
             simplify=r0["simplify"], min_length=r0["min_len"])
    f = {k: np.load(FIXTURES / f"roads_{name}_{k}.npy", mmap_mode="r")
         for k in ("mask", "kp", "direction")}
    return f, p


@pytest.mark.skipif(not (FIXTURES / "meta.json").exists(), reason="needs the E051 field cache")
@pytest.mark.parametrize("name", ["r8", "r9"])
@pytest.mark.parametrize("block,halo", [(512, 128), (1024, 256)])
def test_real_fields_equal_whole_scene(name, block, halo):
    """City-Scale road model outputs (2048 x 2048, seamless), E051 tuned settings."""
    f, p = _fixture(name)
    ref = graph_from_road_heads(np.asarray(f["mask"][0]), np.asarray(f["kp"]),
                                np.asarray(f["direction"]), **p)
    got = _decode(f, RoadBlockDecoder(**p, names={"keypoints": "kp"}), block, halo)
    assert len(ref.junctions()) > 100
    assert _same(got, ref)
