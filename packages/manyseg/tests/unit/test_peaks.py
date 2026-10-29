"""manyseg.utils._peaks: the shared heatmap peak core, and that every caller decodes exactly as
before it was shared (frozen copies of the old helpers below are the reference)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("scipy")
pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")

from manyseg.utils._peaks import maxima_mask, peak_pixels, refine, suppress  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# --- frozen copies of the helpers before the refactor (do not edit) ----------------------------

def _old_peak_pixels(hm, threshold, kernel):  # vectors.graph_tensor._peak_pixels
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


def _old_offset(lo, mid, hi):
    den = lo - 2 * mid + hi
    ok = den < -1e-9
    return np.clip(np.where(ok, 0.5 * (lo - hi) / np.where(ok, den, -1), 0), -1, 1)


def _old_refine(hm, pix):  # vectors.graph_tensor._refine
    if len(pix) == 0:
        return np.zeros((0, 2))
    H, W = hm.shape
    lg = np.log(np.clip(hm, 1e-4, None))
    r, c = pix[:, 0], pix[:, 1]
    inx, iny = (c > 0) & (c < W - 1), (r > 0) & (r < H - 1)
    dx = np.where(inx, _old_offset(
        lg[r, np.maximum(c - 1, 0)], lg[r, c], lg[r, np.minimum(c + 1, W - 1)]), 0)
    dy = np.where(iny, _old_offset(
        lg[np.maximum(r - 1, 0), c], lg[r, c], lg[np.minimum(r + 1, H - 1), c]), 0)
    return np.stack([c + 0.5 + dx, r + 0.5 + dy], 1)


def _old_refine_window(hm, pix, origin, scene_hw):  # vectors.roads._refine_window
    if len(pix) == 0:
        return np.zeros((0, 2))
    H, W = scene_hw
    h, w = hm.shape
    lg = np.log(np.clip(hm, 1e-4, None))
    r, c = pix[:, 0], pix[:, 1]
    R, C = r + origin[0], c + origin[1]
    inx, iny = (C > 0) & (C < W - 1), (R > 0) & (R < H - 1)
    dx = np.where(inx, _old_offset(
        lg[r, np.maximum(c - 1, 0)], lg[r, c], lg[r, np.minimum(c + 1, w - 1)]), 0)
    dy = np.where(iny, _old_offset(
        lg[np.maximum(r - 1, 0), c], lg[r, c], lg[np.minimum(r + 1, h - 1), c]), 0)
    return np.stack([C + 0.5 + dx, R + 0.5 + dy], 1)


def _old_suppress(pts, vals, kernel):  # vectors.roads._suppress
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


def _old_block_mask(hm, threshold, kernel):  # RoadBlockDecoder's inline maxima
    return (hm == ndimage.maximum_filter(hm, size=kernel, mode="nearest")) & (hm > threshold)


def _old_pool_mask(hm, threshold, kernel):  # points.find_peaks's inline maxima
    pooled = F.max_pool2d(hm[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    return (hm == pooled) & (hm > threshold)


# --- heatmaps --------------------------------------------------------------------------------

def _gauss(H, W, centres, sigma=1.5, amp=None):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    hm = np.zeros((H, W))
    for k, (x, y) in enumerate(centres):
        a = 1.0 if amp is None else amp[k]
        hm = np.maximum(hm, a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
    return hm


def heatmaps():
    """Random, smoothed-and-quantised (plateaus and ties), Gaussians at sub-pixel spots incl.
    on borders and corners, flat plateaus, constants, zeros, values exactly at the threshold,
    float64, tiny images."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(4):
        out.append(rng.random((37, 53)).astype(np.float32))
        smooth = ndimage.gaussian_filter(rng.random((80, 90)), 1.5)
        out.append(np.round(smooth, 2).astype(np.float32))
        out.append(np.round(rng.random((40, 40)), 1).astype(np.float32))  # many exact ties
        c = rng.uniform((0, 0), (64, 48), (12, 2))
        c[:3] = [(0.2, 0.3), (63.9, 10.5), (31.7, 47.8)]  # touching the border
        out.append(_gauss(48, 64, c, rng.uniform(1, 2), rng.uniform(0.3, 1, 12)).astype(np.float32))
    plateau = np.zeros((30, 30), np.float32)
    plateau[5:9, 5:12] = 0.8  # one flat-topped blob
    plateau[0:3, 26:30] = 0.6  # on the corner
    plateau[20, 20] = plateau[20, 22] = 0.7  # two equal peaks 2 px apart
    plateau[25, 3] = 0.5  # exactly at the threshold used below
    out += [plateau, np.full((16, 16), 0.9, np.float32), np.zeros((16, 16), np.float32),
            _gauss(33, 29, [(10.3, 12.8), (20.6, 5.1)]), np.float32([[0.9]]),
            np.float32([[0.2, 0.9, 0.9, 0.3]]), rng.random((5, 3)).astype(np.float32)]
    return out


HMS = heatmaps()
CASES = [(i, thr, k) for i in range(len(HMS)) for thr in (0.0, 0.3, 0.5) for k in (3, 4, 5, 7)]


# --- the core equals the old helpers ----------------------------------------------------------

@pytest.mark.parametrize("i,thr,kernel", CASES)
def test_peak_pixels_and_refine_equal_old(i, thr, kernel):
    hm = HMS[i]
    got, ref = peak_pixels(hm, thr, kernel), _old_peak_pixels(hm, thr, kernel)
    assert got.dtype == ref.dtype and got.shape == ref.shape and np.array_equal(got, ref)
    a, b = refine(hm, got), _old_refine(hm, ref)
    assert a.dtype == b.dtype and np.array_equal(a, b)
    keep = maxima_mask(hm, thr, kernel)
    assert np.array_equal(keep, _old_block_mask(hm, thr, kernel))
    pts = np.argwhere(keep)
    assert np.array_equal(suppress(pts, hm[keep], kernel), _old_suppress(pts, hm[keep], kernel))


@pytest.mark.parametrize("i", range(len(HMS)))
@pytest.mark.parametrize("thr,kernel", [(0.0, 3), (0.3, 5), (0.5, 7)])
def test_torch_mask_equals_find_peaks_old_and_numpy(i, thr, kernel):
    hm = torch.from_numpy(HMS[i])[None]
    got = maxima_mask(hm, thr, kernel)
    assert got.dtype == torch.bool and torch.equal(got, _old_pool_mask(hm, thr, kernel))
    # finite values, odd kernel: the two backends agree
    assert np.array_equal(got[0].numpy(), maxima_mask(HMS[i], thr, kernel))


def test_torch_mask_with_masked_out_pixels_and_dtypes():
    hm = torch.from_numpy(HMS[1]).repeat(2, 1, 1)
    hm[1, :, :40] = float("-inf")  # find_peaks's mask
    for t in (hm, hm.double(), hm.half()):
        assert torch.equal(maxima_mask(t, 0.3, 5), _old_pool_mask(t, 0.3, 5))
    assert not maxima_mask(hm, 0.3, 5)[1, :, :40].any()
    with pytest.raises(ValueError, match="odd"):
        maxima_mask(hm, 0.3, 4)


@pytest.mark.parametrize("i", range(len(HMS)))
def test_refine_window_equals_old_and_whole_image(i):
    hm = HMS[i].astype(np.float32)
    H, W = hm.shape
    pix = np.argwhere(maxima_mask(hm, 0.0, 3))
    whole = refine(hm, pix)
    for t, left in [(0, 0), (min(3, H - 1), min(5, W - 1)), (H // 2, W // 3)]:
        b, r = min(H, t + 20), min(W, left + 25)
        win = hm[t:b, left:r]
        # window pixels away from the window's inner edges (as the block decoder's core)
        inner = ((pix[:, 0] > t) | (t == 0)) & ((pix[:, 0] < b - 1) | (b == H)) & \
                ((pix[:, 1] > left) | (left == 0)) & ((pix[:, 1] < r - 1) | (r == W)) & \
                (pix[:, 0] >= t) & (pix[:, 0] < b) & (pix[:, 1] >= left) & (pix[:, 1] < r)
        wp = pix[inner] - (t, left)
        got = refine(win, wp, origin=(t, left), shape=(H, W))
        assert np.array_equal(got, _old_refine_window(win, wp, (t, left), (H, W)))
        assert np.array_equal(got, whole[inner])


# --- behaviour of the core ---------------------------------------------------------------------

def test_threshold_is_strict():
    hm = np.zeros((9, 9), np.float32)
    hm[4, 4] = 0.5
    assert len(peak_pixels(hm, 0.5, 3)) == 0 and len(peak_pixels(hm, 0.49, 3)) == 1
    assert not maxima_mask(torch.from_numpy(hm)[None], 0.5, 3).any()


def test_plateau_mask_keeps_all_suppression_keeps_first_raster_pixel():
    hm = np.zeros((12, 12), np.float32)
    hm[3:5, 4:7] = 0.8
    assert maxima_mask(hm, 0.1, 3).sum() == 6  # every plateau pixel (find_peaks)
    # vectors: suppression from the first raster pixel; a plateau wider than the radius
    # (1.5 px for kernel 3) keeps one pixel per radius, as before the refactor
    assert np.array_equal(peak_pixels(hm, 0.1, 3), [[3, 4], [3, 6]])
    assert np.array_equal(peak_pixels(hm, 0.1, 5), [[3, 4]])
    # two equal peaks farther apart than the radius both survive, raster order
    hm = np.zeros((12, 12), np.float32)
    hm[2, 2] = hm[2, 5] = 0.7
    assert np.array_equal(peak_pixels(hm, 0.1, 3), [[2, 2], [2, 5]])  # 3 px > radius 1.5
    assert np.array_equal(peak_pixels(hm, 0.1, 5), [[2, 2]])  # radius 3, inclusive
    hm[2, 2] = 0.69  # now the stronger one wins
    assert np.array_equal(peak_pixels(hm, 0.1, 5), [[2, 5]])


def test_suppression_strongest_first_and_radius():
    pts = np.array([[0, 0], [0, 3], [0, 6], [10, 10]])
    vals = np.array([0.5, 0.9, 0.5, 0.1])
    assert list(suppress(pts, vals, 5)) == [1, 3]  # radius 3 (inclusive): (0, 3) removes both
    assert list(suppress(pts, vals, 3)) == [1, 0, 2, 3]  # radius 1.5
    assert list(suppress(pts[:1], vals[:1], 5)) == [0] and len(suppress(pts[:0], vals[:0], 5)) == 0


def test_borders_use_nearest_and_refine_skips_border_axes():
    hm = _gauss(20, 20, [(0.5, 7.3)]).astype(np.float32)  # peak in column 0
    (r, c), = peak_pixels(hm, 0.5, 3)
    assert (r, c) == (7, 0)
    x, y = refine(hm, np.array([[r, c]]))[0]
    assert x == 0.5 and abs(y - 7.3) < 1e-3  # no x offset on the border column
    # the window border rule uses the scene, not the window
    win = hm[5:10, 0:4]
    assert np.array_equal(refine(win, np.array([[2, 0]]), origin=(5, 0), shape=(20, 20)),
                          refine(hm, np.array([[7, 0]])))


def test_refine_subpixel_exact_for_gaussian_and_clipped():
    hm = _gauss(32, 32, [(12.3, 17.8)], sigma=2.0)
    x, y = refine(hm, peak_pixels(hm, 0.5, 3))[0]
    assert abs(x - 12.3) < 1e-6 and abs(y - 17.8) < 1e-6
    ramp = np.tile(np.exp([-3, -1, 0, 0.5]), (3, 1))  # parabola vertex 1.5 px right: clipped
    x, y = refine(ramp, np.array([[1, 2]]))[0]
    assert x == 3.5 and y == 1.5  # rows equal: zero curvature, no y offset
    flat = np.full((3, 3), 0.5, np.float32)  # zero curvature: no offset
    assert np.array_equal(refine(flat, np.array([[1, 1]])), [[1.5, 1.5]])
    assert refine(flat, np.zeros((0, 2), np.int64)).shape == (0, 2)


# --- every caller decodes as before (old helpers patched in = reference) ------------------------

def test_find_peaks_equals_old(monkeypatch):
    from manyseg.points import find_peaks
    from manyseg.points import postprocess as pp

    rng = np.random.default_rng(1)
    hm = torch.from_numpy(np.stack([HMS[1][:37, :53], HMS[0], HMS[3][:37, :53]]))
    mask = torch.from_numpy(rng.random(hm.shape) > 0.3)
    calls = [dict(threshold=0.3), dict(threshold=0.5, kernel=5, max_points=7),
             dict(threshold=0.2, kernel=3, smooth_sigma=1.5, mask=mask),
             dict(threshold=0.1, scale=0.5), dict(threshold=0.0, kernel=7)]
    ties = torch.from_numpy(np.stack([HMS[2], HMS[6]]))  # 0.1 steps: maxima at the threshold
    def run():
        return [find_peaks(hm, **c) for c in calls] + [find_peaks(ties, 0.5),
                                                       find_peaks(ties, 0.6, kernel=5)]

    new = run()
    monkeypatch.setattr(pp, "maxima_mask", _old_pool_mask)
    old = run()
    for a, b in zip(new, old):
        for (pa, sa), (pb, sb) in zip(a, b):
            assert pa.dtype == pb.dtype and torch.equal(pa, pb) and torch.equal(sa, sb)


def test_polygons_from_corners_equals_old(monkeypatch):
    from manyseg.vectors import corner_targets, polygons_from_corners
    from manyseg.vectors import corners as C
    from shapely.geometry import Polygon, box

    polys = [box(20, 20, 120, 80), box(0, 150, 60, 230),
             Polygon([(150, 150), (250, 150), (250, 200), (200, 200), (200, 250), (150, 250)])]
    t = corner_targets(polys, (300, 300))
    lab = np.zeros((300, 300), np.int32)
    lab[20:80, 20:120] = 1
    lab[150:230, 0:60] = 2
    lab[150:200, 150:250] = 3
    lab[200:250, 150:200] = 3
    rng = np.random.default_rng(2)
    noisy = np.round(np.clip(t["corner"] + rng.normal(0, 0.1, t["corner"].shape), 0, 1), 2)
    runs = [(t["corner"], {}), (noisy.astype(np.float32), dict(corner_threshold=0.5, kernel=5)),
            (torch.from_numpy(noisy), dict(kernel=3))]
    new = [[p.wkb for p in polygons_from_corners(lab, c, t["direction"], **k)] for c, k in runs]
    monkeypatch.setattr(C, "peak_pixels", _old_peak_pixels)
    def old_refine(hm, pix, origin=(0, 0)):  # whole-scene decodes call with origin (0, 0)
        assert tuple(origin) == (0, 0)
        return _old_refine(hm, pix)

    monkeypatch.setattr(C, "refine", old_refine)
    old = [[p.wkb for p in polygons_from_corners(lab, c, t["direction"], **k)] for c, k in runs]
    assert new == old and len(new[0]) == 3


def test_graph_from_tensor_equals_old(monkeypatch):
    from manyseg.vectors import encode_graph_tensor, graph_from_tensor, vertex_graph
    from manyseg.vectors import graph_tensor as G
    from shapely.geometry import box

    vg = vertex_graph([box(30, 30, 90, 70), box(120, 40, 180, 130), box(0, 150, 50, 200)],
                      border=(200, 200))
    t = encode_graph_tensor(vg, (200, 200))
    rng = np.random.default_rng(3)
    v = np.round(np.clip(t["vertex"] + rng.normal(0, 0.05, t["vertex"].shape), 0, 1), 2)
    runs = [(t["vertex"], {}), (v.astype(np.float32), dict(kernel=5, vertex_threshold=0.3))]
    new = [graph_from_tensor(x, t["edge"], t["vector"], **k) for x, k in runs]
    monkeypatch.setattr(G, "peak_pixels", _old_peak_pixels)
    monkeypatch.setattr(G, "refine", _old_refine)
    old = [graph_from_tensor(x, t["edge"], t["vector"], **k) for x, k in runs]
    for a, b in zip(new, old):
        assert np.array_equal(a.xy, b.xy) and np.array_equal(a.edges, b.edges)
    assert len(new[0].xy) > 10


@pytest.mark.parametrize("seed", [0, 1])
def test_road_decoders_equal_old(monkeypatch, seed):
    from manyseg.vectors import roads as R
    from test_vector_road_blocks import _decode, _same, _scene, _whole

    f = _scene(seed)
    p = dict(threshold=0.5, gap=50, keypoint_threshold=0.3)
    new = [_whole(f, **p), _decode(f, R.RoadBlockDecoder(**p), 128, 32),
           _decode(f, R.RoadBlockDecoder(**p), 160, 40, offset=(37, 91))]
    monkeypatch.setattr(R, "peak_pixels", _old_peak_pixels)
    monkeypatch.setattr(R, "maxima_mask", _old_block_mask)
    monkeypatch.setattr(R, "suppress", _old_suppress)
    monkeypatch.setattr(R, "refine", lambda hm, pix, origin=(0, 0), shape=None:
                        _old_refine(hm, pix) if shape is None
                        else _old_refine_window(hm, pix, origin, shape))
    old = [_whole(f, **p), _decode(f, R.RoadBlockDecoder(**p), 128, 32),
           _decode(f, R.RoadBlockDecoder(**p), 160, 40, offset=(37, 91))]
    for a, b in zip(new, old):
        assert _same(a, b)
    assert len(new[0].junctions()) > 3
