"""Block-wise point and count decoding (manyseg.points.blocks) against whole-scene decodes:
exact equality with find_peaks for peaks on seams and corners, plateaus and ties, scene edges,
several grids; halo checks; pickling."""

import pickle
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from manyseg.points import (
    DensityBlockDecoder,
    PointBlockDecoder,
    ScenePoints,
    count_from_density,
    find_peaks,
    gaussian_heatmap,
    points_from_xy,
    points_to_xy,
)
from manyseg.points.postprocess import _gaussian_blur


# A minimal stand-in for the contract's Block / Scene (ManyPred's or manyseg.vectors.blocks')
@dataclass(frozen=True)
class Block:
    index: int
    core: tuple
    window: tuple
    scene_edges: tuple


@dataclass(frozen=True)
class Scene:
    height: int
    width: int
    transform: object = None
    crs: object = None


def grid(h, w, block, halo, offset=(0, 0)):
    """Cores partition the scene (first row / column shortened by ``offset``); windows = core +
    halo, clipped."""
    def cuts(n, o):
        return sorted({0, n, *range(o % block or block, n, block)})

    rs, cs = cuts(h, offset[0]), cuts(w, offset[1])
    out = []
    for t, b in zip(rs, rs[1:]):
        for le, r in zip(cs, cs[1:]):
            win = (max(t - halo, 0), min(b + halo, h), max(le - halo, 0), min(r + halo, w))
            out.append(Block(len(out), (t, b, le, r), win,
                             (win[0] == 0, win[1] == h, win[2] == 0, win[3] == w)))
    return out


def decode(fields, dec, block, halo, offset=(0, 0), scene=None):
    h, w = next(iter(fields.values())).shape[-2:]
    scene = scene or Scene(h, w)
    res = []
    for bl in grid(h, w, block, halo, offset):
        t, b, le, r = bl.window
        res.append(dec(({k: v[:, t:b, le:r] for k, v in fields.items()}), bl, scene))
    return dec.combine(res, scene)


def reference(hm, **kw):
    """Whole-scene find_peaks, in the decoder's order (score descending, ties raster order)."""
    ((p, s),) = find_peaks(torch.from_numpy(hm)[None], **kw)
    p, s = p.numpy(), s.numpy()
    o = np.lexsort((p[:, 1], p[:, 0], -s.astype(np.float64)))
    return p[o], s[o]


def scene_heatmap(h=97, w=113, block=24, seed=0, quantise=False):
    """Gaussian blobs + noise; extra peaks on seams (rows / columns at multiples of ``block``,
    on both sides), seam corners and the scene edges."""
    rng = np.random.default_rng(seed)
    pts = [rng.uniform(0, [h, w]) for _ in range(40)]
    seams_r, seams_c = range(block, h, block), range(block, w, block)
    pts += [(r + d, rng.uniform(0, w)) for r in seams_r for d in (-1, 0)]
    pts += [(rng.uniform(0, h), c + d) for c in seams_c for d in (-1, 0)]
    pts += [(r + dr, c + dc) for r in seams_r for c in seams_c for dr in (-1, 0) for dc in (-1, 0)]
    pts += [(0, 5), (h - 1, 7), (11, 0), (13, w - 1), (0, 0), (h - 1, w - 1)]
    p = torch.tensor(np.round(np.array(pts, float)), dtype=torch.float32)
    hm = gaussian_heatmap([p], (h, w), sigma=1.5)[0].numpy()
    hm = hm * rng.uniform(0.5, 1.0) + rng.uniform(0, 0.3, hm.shape).astype(np.float32)
    if quantise:  # many exact ties and multi-pixel plateaus, some across seams
        hm = np.round(hm * 6) / 6
        hm[:, 20:30, 20:30] = 0.9  # a flat square across the 24-px seams: every pixel a peak
    return hm.astype(np.float32)


@pytest.mark.parametrize("kw", [
    {"threshold": 0.4},
    {"threshold": 0.3, "kernel": 5},
    {"threshold": 0.2, "kernel": 7, "smooth_sigma": 1.0},
    {"threshold": 0.25, "kernel": 3, "smooth_sigma": 1.7},
])
@pytest.mark.parametrize("quantise", [False, True])
def test_block_decode_equals_whole_scene_find_peaks(kw, quantise):
    hm = scene_heatmap(quantise=quantise)
    ref_p, ref_s = reference(hm, **kw)
    assert len(ref_p) > 30
    dec = PointBlockDecoder(**kw)
    for block, offset in [(24, (0, 0)), (24, (7, 13)), (17, (0, 5)), (40, (3, 0)), (200, (0, 0)),
                          (1, (0, 0))]:
        for halo in (dec.min_halo, dec.min_halo + 3):
            if block == 1 and halo != dec.min_halo:
                continue
            out = decode({"heatmap": hm}, dec, block, halo, offset)
            assert np.array_equal(out.pixels, ref_p), (block, offset, halo)
            assert np.array_equal(out.scores, ref_s)
            assert np.array_equal(out.yx, ref_p.astype(np.float64))
    if quantise and not kw.get("smooth_sigma"):  # the flat square across the seams: its
        # interior (every neighbour inside the square) is all peaks, each kept once
        sq = (out.pixels >= 23) & (out.pixels < 27)
        assert (sq[:, 0] & sq[:, 1]).sum() == 16


def test_too_small_halo_raises_and_would_be_wrong():
    hm = scene_heatmap()
    dec = PointBlockDecoder(threshold=0.2, kernel=5, smooth_sigma=1.0)
    assert dec.min_halo == 2 + 3
    with pytest.raises(ValueError, match="halo too small"):
        decode({"heatmap": hm}, dec, 24, dec.min_halo - 1)

    class Unchecked(PointBlockDecoder):  # what the check prevents: seam duplicates / losses
        @property
        def min_halo(self):
            return 0

    ref_p, _ = reference(hm, threshold=0.2, kernel=5, smooth_sigma=1.0)
    out = decode({"heatmap": hm}, Unchecked(threshold=0.2, kernel=5, smooth_sigma=1.0), 24, 1)
    assert not np.array_equal(out.pixels, ref_p)
    # a scene edge needs no halo: one block, window = scene
    one = decode({"heatmap": hm}, dec, 1000, 0)
    assert np.array_equal(one.pixels, ref_p)


def test_halo_check_reads_scene_edges_without_scene_size():
    hm = scene_heatmap()
    dec = PointBlockDecoder(threshold=0.3)
    res = [dec({"heatmap": hm[:, b.window[0]:b.window[1], b.window[2]:b.window[3]]}, b, None)
           for b in grid(*hm.shape[1:], 30, 1)]
    assert np.array_equal(dec.combine(res, None).pixels, reference(hm, threshold=0.3)[0])
    bad = Block(0, (0, 30, 0, 30), (0, 30, 0, 30), (True, False, True, True))
    with pytest.raises(ValueError, match="bottom"):
        dec({"heatmap": hm[:, :30, :30]}, bad, None)


def test_offsets_mask_logits_and_max_points():
    hm = np.clip(scene_heatmap(seed=3), 0.01, 0.99)
    h, w = hm.shape[1:]
    rng = np.random.default_rng(1)
    off = rng.uniform(-0.5, 0.5, (2, h, w)).astype(np.float32)
    fg = (rng.uniform(0, 1, (1, h, w)) > 0.3).astype(np.float32)
    logit = np.log(hm / (1 - hm)).astype(np.float32)
    # logits=True applies to the mask field too: logit 0.3 -> p 0.57 (> 0.5), -3 -> p 0.05
    fields = {"hm_logit": logit, "offset": off, "fg": np.where(fg > 0.5, 0.3, -3.0)}
    dec = PointBlockDecoder("hm_logit", threshold=0.3, kernel=3, offsets="offset", mask="fg",
                            logits=True, max_points=25)
    out = decode(fields, dec, 20, dec.min_halo, (5, 9))
    prob = torch.sigmoid(torch.from_numpy(logit))
    ((p, s),) = find_peaks(prob[None], 0.3, mask=torch.from_numpy(fg > 0.5))
    p, s = p.numpy(), s.numpy()
    o = np.lexsort((p[:, 1], p[:, 0], -s.astype(np.float64)))[:25]
    assert len(out) == 25 and np.array_equal(out.pixels, p[o]) and np.array_equal(out.scores, s[o])
    assert np.allclose(out.yx, p[o] + off[:, p[o][:, 0], p[o][:, 1]].T)
    assert np.allclose(out.xy, out.yx[:, ::-1] + 0.5)
    raw = PointBlockDecoder("hm_logit", threshold=0.3, offsets="offset", mask="fg", logits=False)
    fields["hm_logit"] = prob.numpy()  # without logits=True the mask is compared raw: 0.3 < 0.5
    assert len(decode(fields, raw, 20, raw.min_halo, (5, 9))) == 0


def test_decode_scene_checks_point_halo_early_and_empty_fields():
    from manyseg.vectors import decode_scene

    hm = scene_heatmap()
    dec = PointBlockDecoder(threshold=0.3, kernel=5)
    with pytest.raises(ValueError, match="min_halo"):  # decode_scene's own check, not a block's
        decode_scene({"heatmap": hm}, dec, 24, dec.min_halo - 1)
    with pytest.raises(ValueError, match="must not be empty"):
        decode_scene({}, dec)
    one = decode_scene({"heatmap": hm}, dec)  # default block 4096: the whole scene, one block
    assert np.array_equal(one.pixels, reference(hm, threshold=0.3, kernel=5)[0])
    assert DensityBlockDecoder(value_scale=1.0).min_halo == 0


def test_float16_fields_and_scene_transform():
    hm = scene_heatmap().astype(np.float16)
    dec = PointBlockDecoder(threshold=0.4)
    t = (0.5, 0.0, 1000.0, 0.0, -0.5, 2000.0)
    out = decode({"heatmap": hm}, dec, 32, 1, scene=Scene(*hm.shape[1:], t, "EPSG:28350"))
    ref_p, ref_s = reference(hm.astype(np.float32), threshold=0.4)
    assert np.array_equal(out.pixels, ref_p) and np.array_equal(out.scores, ref_s)
    assert out.transform == t and out.crs == "EPSG:28350"
    pytest.importorskip("shapely")
    fc = out.to_geojson()  # the scene's CRS travels with the points, as in records_to_geojson
    assert fc["crs"] == {"type": "name", "properties": {"name": "EPSG:28350"}}
    assert len(fc["features"]) == len(out.pixels)


def test_empty_and_pickling():
    hm = np.zeros((1, 40, 50), np.float32)
    dec = PointBlockDecoder()
    out = decode({"heatmap": hm}, dec, 16, 1)
    assert out.yx.shape == (0, 2) and out.scores.shape == (0,) and out.pixels.shape == (0, 2)
    assert len(dec.combine([], Scene(40, 50))) == 0
    dec = PointBlockDecoder(threshold=0.3, kernel=5, smooth_sigma=1.2, offsets="offset")
    dec2 = pickle.loads(pickle.dumps(dec))
    assert dec2 == dec and dec2.min_halo == dec.min_halo
    hm = scene_heatmap()
    fields = {"heatmap": hm, "offset": np.zeros((2, *hm.shape[1:]), np.float32)}
    bl = grid(*hm.shape[1:], 30, dec.min_halo)
    t, b, le, r = bl[4].window
    part = dec({k: v[:, t:b, le:r] for k, v in fields.items()}, bl[4], Scene(*hm.shape[1:]))
    part2 = pickle.loads(pickle.dumps(part))
    assert isinstance(part2, ScenePoints) and np.array_equal(part2.yx, part.yx)
    out = pickle.loads(pickle.dumps(decode(fields, dec2, 30, dec.min_halo)))
    assert np.array_equal(out.pixels, reference(hm, threshold=0.3, kernel=5, smooth_sigma=1.2)[0])
    dd = pickle.loads(pickle.dumps(DensityBlockDecoder(value_scale=2.0)))
    assert dd == DensityBlockDecoder(value_scale=2.0)


def test_overlapping_cores_and_bad_fields_raise():
    hm = scene_heatmap()
    dec = PointBlockDecoder(threshold=0.3)
    s = Scene(*hm.shape[1:])
    whole = Block(0, (0, s.height, 0, s.width), (0, s.height, 0, s.width), (True,) * 4)
    r = dec({"heatmap": hm}, whole, s)
    with pytest.raises(ValueError, match="overlap"):
        dec.combine([r, r], s)
    with pytest.raises(ValueError, match="expected"):
        dec({"heatmap": hm[:, :10]}, whole, s)
    with pytest.raises(KeyError):
        dec({"hm": hm}, whole, s)
    with pytest.raises(ValueError):
        PointBlockDecoder(kernel=4)


def test_gaussian_blur_is_crop_invariant_bitwise():
    x = torch.rand(1, 120, 140, generator=torch.Generator().manual_seed(0))
    for sigma in (0.6, 1.0, 2.5):
        r = max(1, int(np.ceil(3 * sigma)))
        full = _gaussian_blur(x, sigma)
        crop = _gaussian_blur(x[:, 17:101, 23:131], sigma)
        assert torch.equal(crop[:, r:-r, r:-r], full[:, 17 + r:101 - r, 23 + r:131 - r])


def test_density_block_count():
    rng = np.random.default_rng(0)
    pts = torch.from_numpy(rng.uniform(0, [89, 109], (60, 2))).float()  # inside the image
    from manyseg.points import density_map

    d = density_map([pts], (90, 110), sigma=3.0)[0].numpy() * 4.0
    fg = (rng.uniform(0, 1, (1, 90, 110)) > 0.2).astype(np.float32)
    dec = DensityBlockDecoder(value_scale=4.0)
    out = decode({"density": d}, dec, 25, 0, (4, 11))
    assert out.total == pytest.approx(60.0, rel=1e-5)
    assert out.total == pytest.approx(float(count_from_density(torch.from_numpy(d)[None],
                                                               value_scale=4.0)), rel=1e-5)
    assert len(out.counts) == len(out.cores) == len(grid(90, 110, 25, 0, (4, 11)))
    m = decode({"density": d, "fg": fg}, DensityBlockDecoder(value_scale=4.0, mask="fg"), 25, 0)
    ref = count_from_density(torch.from_numpy(d)[None], value_scale=4.0,
                             mask=torch.from_numpy(fg > 0.5))
    assert m.total == pytest.approx(float(ref), rel=1e-5)


def test_coordinate_conversions():
    yx = torch.tensor([[2, 3], [0, 0]])
    xy = points_to_xy(yx)
    assert xy.tolist() == [[3.5, 2.5], [0.5, 0.5]]  # centre of pixel (row 2, col 3)
    back = points_from_xy(xy)
    assert back.dtype == torch.float32 and back.tolist() == [[2, 3], [0, 0]]
    hm = gaussian_heatmap([points_from_xy([[3.5, 2.5]])], (5, 6))
    assert hm.shape == (1, 1, 5, 6) and hm[0, 0, 2, 3] == 1.0
    assert points_to_xy(np.zeros((0, 2))).shape == (0, 2)


def test_blocks_from_vectors_block_grid():
    """Blocks from manyseg.vectors.block_grid (offset grid) decode like the whole scene."""
    pytest.importorskip("scipy")
    from manyseg.vectors.blocks import block_grid

    hm = scene_heatmap()
    dec = PointBlockDecoder(threshold=0.3, kernel=5)
    s = Scene(*hm.shape[1:])
    res = [dec({"heatmap": hm[:, b.window[0]:b.window[1], b.window[2]:b.window[3]]}, b, s)
           for b in block_grid(*hm.shape[1:], 24, dec.min_halo, offset=(5, 7))]
    assert np.array_equal(dec.combine(res, s).pixels, reference(hm, threshold=0.3, kernel=5)[0])


def test_block_decoder_registry_points_and_density():
    """manyseg.vectors.block_decoder("points" | "density", **params) builds the point decoders,
    and decode_scene runs them like the whole scene."""
    pytest.importorskip("scipy")
    from manyseg.vectors import block_decoder, decode_scene

    dec = block_decoder("points", threshold=0.3, kernel=5, max_points=50)
    assert type(dec) is PointBlockDecoder and dec.kernel == 5 and dec.max_points == 50
    assert type(block_decoder("density", sigma=2.0)) is DensityBlockDecoder
    assert block_decoder("density", value_scale=4.0, mask="fg").value_scale == 4.0
    with pytest.raises(TypeError):
        block_decoder("points", min_area=3)  # not a PointBlockDecoder parameter

    hm = scene_heatmap()
    out = decode_scene({"heatmap": hm[0]}, block_decoder("points", threshold=0.3, kernel=5),
                       24, 2, offset=(5, 7))  # 2-D field, as ManyPred-free callers pass it
    assert np.array_equal(out.pixels, reference(hm, threshold=0.3, kernel=5)[0])
    d = np.abs(hm).astype(np.float32)
    cnt = decode_scene({"density": d}, block_decoder("density", value_scale=2.0), 24, 0)
    assert cnt.total == pytest.approx(float(d.sum(dtype=np.float64)) / 2.0, rel=1e-6)


@pytest.mark.parametrize("scale", [0, -1.0, float("inf"), float("nan")])
def test_density_decoder_rejects_a_bad_value_scale(scale):
    with pytest.raises(ValueError, match="value_scale"):
        DensityBlockDecoder(value_scale=scale)


def test_density_counts_nan_pixels_as_zero():
    """ManyPred's stitched outputs hold NaN at nodata: the total must stay a number."""
    from manyseg.vectors import block_decoder, decode_scene

    density = np.ones((1, 32, 32), np.float32)
    density[0, :8, :8] = np.nan
    dec = block_decoder("density", value_scale=1.0)
    assert decode_scene({"density": density}, dec, 16, 0).total == 32 * 32 - 64
    mask = np.ones((1, 32, 32), np.float32)
    mask[0, :8, :8] = np.nan
    mask[0, 16:, :] = 0
    dec = block_decoder("density", value_scale=1.0, mask="valid")
    got = decode_scene({"density": density, "valid": mask}, dec, 16, 0)
    assert got.total == 32 * 16 - 64
