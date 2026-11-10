"""Regression tests for the fourth sweep's instance, point and vector fixes: points near a tile
edge counted by exactly one tile, the roof plane's pixel convention, non-finite distances in
instance decoding, DensityBlockDecoder's missing-field message and NaN policy, and
polygon_metrics' empty-case conventions."""

import math

import numpy as np
import pytest
import torch

shapely = pytest.importorskip("shapely")

from manyseg.instances import instances_from_outputs  # noqa: E402
from manyseg.points import (  # noqa: E402
    count_from_density,
    density_map,
    gaussian_heatmap,
    points_from_xy,
)
from manyseg.vectors import Scene, block_decoder, decode_scene, to_pixels  # noqa: E402
from manyseg.vectors.metrics import polygon_metrics  # noqa: E402


def _tile_counts(point, shape=(256, 256), px=0.5, ox=391000.0, oy=6466000.0):
    """Density counts of one world point in two tiles side by side (A left of B)."""
    h, w = shape
    counts = []
    for x0 in (ox, ox + w * px):
        xy = to_pixels([point], (px, 0, x0, 0, -px, oy), shape, kind="points")
        counts.append(float(density_map([points_from_xy(xy)], shape).sum()) if len(xy) else 0.0)
    return counts


def test_point_just_inside_tile_edge_counted_once():
    ox, oy = 391000.0, 6466000.0
    a, b = _tile_counts(shapely.Point(ox + 128 - 2e-6, oy - 50.25))
    assert a + b == pytest.approx(1.0) and a == pytest.approx(1.0)


@pytest.mark.parametrize("dx", [-1e-3, -1e-6, -2e-6, -1e-9, 0.0, 1e-9, 1e-6])
def test_points_near_shared_edge_sum_to_one(dx):
    ox, oy = 391000.0, 6466000.0
    a, b = _tile_counts(shapely.Point(ox + 128 + dx, oy - 50.25))
    assert a + b == pytest.approx(1.0)
    assert {round(a), round(b)} == {0, 1}


def test_points_from_xy_keeps_float64():
    xy = np.array([[255.999996, 100.5]])
    yx = points_from_xy(xy)
    assert yx.dtype == torch.float64
    assert float(yx[0, 1]) < 255.5  # not rounded onto the far edge
    # the heatmap accepts it and renders in float32
    hm = gaussian_heatmap([yx], (256, 256))
    assert hm.dtype == torch.float32
    assert float(hm.max()) == pytest.approx(math.exp(-0.25 / 8), rel=1e-4)  # half a pixel off


def test_density_map_inside_test_in_input_precision():
    edge = torch.tensor([[10.0, 255.5 - 4e-6]], dtype=torch.float64)
    assert float(density_map([edge], (20, 256)).sum()) == pytest.approx(1.0)
    assert float(density_map([edge.float()], (20, 256)).sum()) == 0.0  # float32: on the edge
    on = torch.tensor([[10.0, 255.5]], dtype=torch.float64)
    assert float(density_map([on], (20, 256)).sum()) == 0.0
    assert density_map([edge], (20, 256)).dtype == torch.float32
    ints = torch.tensor([[3, 4], [20, 4]])  # integer points still work
    assert float(density_map([ints], (20, 8)).sum()) == pytest.approx(1.0)


def test_roof_plane_pixel_centres_at_half():
    from manyseg.vectors.roofs import roof_planes

    rr, cc = np.mgrid[0:60, 0:60]
    height = (0.15 * cc + 0.05 * rr + 3.0).astype(np.float32)
    a, b, c = roof_planes([shapely.box(5, 5, 50, 50)], height)[0]
    assert a * (20 + 0.5) + b * (10 + 0.5) + c == pytest.approx(float(height[10, 20]), abs=1e-4)


def _two_seedless():
    ext = torch.zeros(1, 1, 12, 12)
    ext[..., 2:6, 2:6] = 1
    ext[..., 7:10, 7:10] = 1
    bnd = torch.ones(1, 1, 12, 12)  # all boundary: each object needs its fallback seed
    dist = torch.rand(1, 1, 12, 12, generator=torch.Generator().manual_seed(0))
    return ext, bnd, dist


@pytest.mark.parametrize("backend", ["torch", "skimage"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_distance_counts_as_zero(backend, bad):
    if backend == "skimage":
        pytest.importorskip("skimage")
    ext, bnd, dist = _two_seedless()
    d = dist.clone()
    d[0, 0, 3, 3] = bad
    zero = dist.clone()
    zero[0, 0, 3, 3] = 0.0
    out = instances_from_outputs(ext, bnd, d, logits=False, backend=backend)
    assert out.unique().tolist() == [0, 1, 2]
    assert torch.equal(out, instances_from_outputs(ext, bnd, zero, logits=False, backend=backend))
    assert out[0, 3, 3] > 0  # the pixel stays part of its instance


@pytest.mark.parametrize("backend", ["torch", "skimage"])
def test_nan_distance_with_seeds(backend):
    if backend == "skimage":
        pytest.importorskip("skimage")
    ext = torch.zeros(1, 1, 16, 16)
    ext[..., 2:14, 2:14] = 1
    bnd = torch.zeros(1, 1, 16, 16)
    dist = torch.full((1, 1, 16, 16), 0.5)
    dist[..., 5:9, 5:9] = float("nan")
    out = instances_from_outputs(ext, bnd, dist, logits=False, backend=backend)
    assert out.unique().tolist() == [0, 1] and int((out > 0).sum()) == 144


def test_density_decoder_missing_field_message():
    dec = block_decoder("density", sigma=2)
    with pytest.raises(KeyError, match=r"field 'density' missing \(have \['heat'\]\)"):
        decode_scene({"heat": np.zeros((1, 8, 8), np.float32)}, dec, scene=Scene(8, 8))
    with pytest.raises(KeyError, match=r"field 'heatmap' missing"):
        decode_scene({"heat": np.zeros((1, 8, 8), np.float32)}, block_decoder("points"),
                     scene=Scene(8, 8))


def test_density_decoder_sums_channels_like_count_from_density():
    d = np.random.default_rng(0).uniform(0, 1, (2, 16, 20)).astype(np.float32)
    total = decode_scene({"density": d}, block_decoder("density", value_scale=1.0),
                         block=7, halo=0, scene=Scene(16, 20)).total
    ref = float(count_from_density(torch.from_numpy(d)[None], value_scale=1.0)[0])
    assert total == pytest.approx(ref, rel=1e-5)


def test_density_nan_policy_differs():
    d = np.ones((1, 8, 8), np.float32)
    d[0, 2, 3] = np.nan
    total = decode_scene({"density": d}, block_decoder("density", value_scale=1.0),
                         scene=Scene(8, 8)).total
    assert total == pytest.approx(63.0)  # NaN pixel counts as 0
    assert math.isnan(float(count_from_density(torch.from_numpy(d)[None], value_scale=1.0)[0]))


def test_polygon_metrics_empty_conventions():
    sq = shapely.box(0, 0, 4, 4)
    e = polygon_metrics([], [])
    assert e["det_f1"] == 1.0 and e["ciou"] == 0.0 and e["miou"] == 0.0
    assert math.isnan(e["n_ratio"]) and math.isnan(e["polis"])
    for gt, pred in (([sq], []), ([], [sq]), ([sq], [shapely.box(10, 10, 12, 12)])):
        m = polygon_metrics(gt, pred)
        assert m["det_f1"] == 0.0 and m["ciou"] == 0.0 and m["miou"] == 0.0
        assert math.isnan(m["n_ratio"]) and math.isnan(m["polis"])
        assert m["sum_n_pred"] == 0 and m["sum_n_gt"] == 0
