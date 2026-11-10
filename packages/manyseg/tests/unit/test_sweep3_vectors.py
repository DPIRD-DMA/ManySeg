"""Regression tests for the third sweep's instance, point and vector fixes: road block decoding
on a reduced-resolution grid, per-image seed fragments, CRS only with a transform when
writing, tile-local density targets, threshold calibration masks and targets, input checks of
graph_from_mask_blocks and polygon_metrics, and ScenePoints.write."""

import json

import numpy as np
import pytest
import torch

shapely = pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")

from manyseg.instances import instances_from_outputs  # noqa: E402
from manyseg.points import (  # noqa: E402
    ScenePoints,
    calibrate_threshold,
    count_from_density,
    density_loss,
    density_map,
    find_peaks,
    gaussian_heatmap,
)
from manyseg.vectors import (  # noqa: E402
    Scene,
    SceneRecords,
    VectorGraph,
    block_decoder,
    decode_scene,
    graph_from_mask_blocks,
    graph_from_road_heads,
    records_to_geojson,
    road_graph,
    road_targets,
    write_records,
)
from manyseg.vectors.metrics import polygon_metrics  # noqa: E402

CRS = "EPSG:32633"


# --- V1: road block decoding of a half-resolution head gives world coordinates ------------------


def test_road_block_decoder_on_a_half_resolution_grid_gives_the_world_line():
    # input raster 200 x 200 at 1 m from (1000, 2000); the heads are at scale 0.5: 100 x 100, 2 m
    g, cr = road_graph([shapely.LineString([(10.0, 50.0), (90.0, 50.0)])])  # head-grid pixels
    t = road_targets(g, (100, 100), crossings=cr)
    fields = {k: t[k] for k in ("mask", "keypoints", "direction")}
    head_tf = (2.0, 0.0, 1000.0, 0.0, -2.0, 2000.0)
    vg = decode_scene(fields, block_decoder("roads"), scene=Scene(100, 100, head_tf, CRS))
    coords = np.array([c for f in vg.to_geojson()["features"]
                       for c in f["geometry"]["coordinates"]])
    # the true line: world y = 2000 - 2 * 50 = 1900, x from 1020 to 1180
    np.testing.assert_allclose(coords[:, 1], 1900.0, atol=2.0)
    assert coords[:, 0].min() == pytest.approx(1020.0, abs=12.0)
    assert coords[:, 0].max() == pytest.approx(1180.0, abs=12.0)
    # block decoders work on the scene grid: no scale (it used to rescale the coordinates)
    with pytest.raises(TypeError):
        block_decoder("roads", scale=0.5)
    # graph_from_road_heads keeps its scale: head-grid coordinates divided by it
    whole = graph_from_road_heads(*fields.values())
    half = graph_from_road_heads(*fields.values(), scale=0.5)
    np.testing.assert_allclose(half.xy, whole.xy / 0.5)
    np.testing.assert_allclose(vg.xy, whole.xy)


# --- V2: small seed fragments are counted per image ---------------------------------------------


def test_drop_small_seeds_is_per_image():
    H = W = 24
    ext = torch.zeros(1, 1, H, W)
    ext[0, 0, 2:22, 2:22] = 1
    dist = torch.zeros(1, 1, H, W)
    dist[0, 0, 8:18, 8:18] = 1.0  # main seed, 100 px
    dist[0, 0, 3:5, 3:5] = 1.0  # 4 px fragment
    bnd = torch.zeros(1, 1, H, W)
    kw = dict(logits=False, min_seed=8, backend="torch")
    one = instances_from_outputs(ext, bnd, dist, **kw)
    two = instances_from_outputs(ext.repeat(2, 1, 1, 1), bnd.repeat(2, 1, 1, 1),
                                 dist.repeat(2, 1, 1, 1), **kw)
    assert int(one.max()) == 1
    assert torch.equal(two[0], one[0]) and torch.equal(two[1], one[0])


# --- V3: no CRS on pixel coordinates when writing -----------------------------------------------


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg"])
def test_write_drops_the_crs_without_a_transform(tmp_path, suffix):
    recs = SceneRecords([{"id": 1, "geometry": shapely.box(0, 0, 2, 2)}], crs=CRS)
    vg = VectorGraph(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0, 1]]), crs=CRS)
    if suffix == ".geojson":
        for obj in (recs, vg):
            assert "crs" not in obj.to_geojson()
            assert "crs" not in json.loads(obj.write(tmp_path / f"a{suffix}").read_text())
            assert "crs" not in json.loads(write_records(obj, tmp_path / f"b{suffix}")
                                           .read_text())
        p = write_records([{"id": 1, "geometry": shapely.box(0, 0, 1, 1)}],
                          tmp_path / f"c{suffix}", scene=Scene(10, 10, crs=CRS))
        assert "crs" not in json.loads(p.read_text())
        return
    gpd = pytest.importorskip("geopandas")
    for i, obj in enumerate((recs, vg)):
        assert gpd.read_file(obj.write(tmp_path / f"a{i}{suffix}")).crs is None
    p = write_records([{"id": 1, "geometry": shapely.box(0, 0, 1, 1)}],
                      tmp_path / f"c{suffix}", scene=Scene(10, 10, crs=CRS))
    assert gpd.read_file(p).crs is None
    # with a transform the CRS is still written
    geo = SceneRecords(recs, transform=(1.0, 0.0, 500000.0, 0.0, -1.0, 0.0), crs=CRS)
    assert gpd.read_file(geo.write(tmp_path / f"d{suffix}")).crs == CRS


# --- V4: points outside the tile add nothing to the density target ------------------------------


def test_density_map_ignores_points_outside_the_image():
    inside = torch.tensor([[0.0, 0.0], [10.0, 10.0], [-0.4, 19.4]])  # two on the border
    outside = torch.tensor([[-0.6, 5.0], [-6.7, 10.0], [5.0, 19.5], [30.0, 30.0]])
    d = density_map([torch.cat([inside, outside])], (20, 20), sigma=1.0)
    torch.testing.assert_close(d, density_map([inside], (20, 20), sigma=1.0))
    assert float(d.sum()) == pytest.approx(3.0, rel=1e-5)  # inside points keep full mass
    # per-point sigmas are filtered with their points
    sig = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    ref = density_map([inside], (20, 20), sigma=[sig[:3]])
    torch.testing.assert_close(density_map([torch.cat([inside, outside])], (20, 20), [sig]), ref)
    # the loss target and the count decoders agree: sum == count inside
    target = d * (2 * np.pi)
    assert float(density_loss(target, [torch.cat([inside, outside])], sigma=1.0)) < 1e-12
    assert float(count_from_density(target, sigma=1.0)[0]) == pytest.approx(3.0, rel=1e-5)
    cnt = decode_scene({"density": target[0].numpy()}, block_decoder("density", sigma=1.0),
                       block=8, halo=0)
    assert cnt.total == pytest.approx(3.0, rel=1e-5)
    # NaN points still show up as NaN
    assert density_map([torch.tensor([[float("nan"), 3.0]])], (8, 8), 1.0).isnan().any()


# --- V7: calibrate_threshold masks and targets --------------------------------------------------


def test_calibrate_threshold_masks_and_targets():
    pts = [torch.tensor([[5.0, 5.0]]), torch.tensor([[10.0, 10.0], [3.0, 20.0]])]
    hm = gaussian_heatmap(pts, (32, 32))
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    ref = calibrate_threshold(hm, pts, metric="f1", mask=mask)
    # a 2-D heatmap with a 2-D mask, as find_peaks
    one = calibrate_threshold(hm[0, 0], pts[:1], metric="f1", mask=mask[0])
    assert one[1] == calibrate_threshold(hm[:1], pts[:1], metric="f1", mask=mask[:1])[1]
    assert len(find_peaks(hm[0, 0], 0.5, mask=mask[0])) == 1
    # a list of heatmaps takes a list of masks, one per batch
    assert calibrate_threshold([hm[:1], hm[1:]], pts, metric="f1",
                               mask=[mask[:1], mask[1:]]) == ref
    with pytest.raises(ValueError, match="one mask per batch"):
        calibrate_threshold([hm[:1], hm[1:]], pts, metric="f1", mask=[mask])
    with pytest.raises(ValueError, match="sequence of masks"):
        calibrate_threshold([hm[:1], hm[1:]], pts, metric="f1", mask=mask)
    # a bare (N, 2) tensor is not one entry per image
    with pytest.raises(ValueError, match="one entry per image"):
        calibrate_threshold(hm[1:], pts[1], metric="f1")
    assert calibrate_threshold(hm[1:], [pts[1]], metric="f1")[0] == ref[0]


# --- V8: graph_from_mask_blocks checks its input shape ------------------------------------------


def test_graph_from_mask_blocks_needs_a_2d_map():
    with pytest.raises(ValueError, match=r"expected a \(H, W\) probability map"):
        graph_from_mask_blocks(np.zeros((1, 8, 8)))
    with pytest.raises(ValueError, match=r"expected a \(H, W\) probability map"):
        graph_from_mask_blocks(torch.zeros(1, 8, 8))
    assert len(graph_from_mask_blocks(np.zeros((8, 8))).edges) == 0


# --- V9: polygon_metrics refuses MultiPolygons clearly ------------------------------------------


def test_polygon_metrics_refuses_multipolygons():
    P = [shapely.box(0, 0, 10, 10), shapely.box(20, 0, 30, 10)]
    mp = shapely.MultiPolygon(P)
    with pytest.raises(ValueError, match=r"gt\[0\] is a MultiPolygon.*explode"):
        polygon_metrics([mp], P)
    with pytest.raises(ValueError, match=r"pred\[1\] is a MultiPolygon"):
        polygon_metrics(P, [P[0], mp])
    with pytest.raises(ValueError, match="LineString"):
        polygon_metrics(P, [shapely.LineString([(0, 0), (1, 1)])])
    assert polygon_metrics(list(mp.geoms), P)["det_f1"] == 1.0


# --- D5: ScenePoints.write ----------------------------------------------------------------------


def _points(transform=None, crs=CRS):
    return ScenePoints(yx=np.array([[2.0, 3.0], [5.0, 1.0]]),
                       scores=np.array([0.9, 0.6], np.float32),
                       pixels=np.array([[2, 3], [5, 1]]), transform=transform, crs=crs)


def test_scene_points_write_geojson(tmp_path):
    tf = (2.0, 0.0, 1000.0, 0.0, -2.0, 2000.0)
    sp = _points(tf)
    fc = json.loads(sp.write(tmp_path / "p.geojson").read_text())
    assert fc == json.loads(json.dumps(sp.to_geojson()))
    assert fc["crs"]["properties"]["name"] == CRS
    assert [f["properties"]["score"] for f in fc["features"]] == pytest.approx([0.9, 0.6])
    assert fc["features"][0]["geometry"]["coordinates"] == pytest.approx([1007.0, 1995.0])
    assert json.loads(write_records(sp, tmp_path / "q.geojson").read_text()) == fc
    assert records_to_geojson(sp) == sp.to_geojson()
    plain = json.loads(_points().write(tmp_path / "r.geojson").read_text())
    assert "crs" not in plain  # pixel coordinates: no CRS
    assert plain["features"][0]["geometry"]["coordinates"] == pytest.approx([3.5, 2.5])


def test_scene_points_write_gpkg(tmp_path):
    gpd = pytest.importorskip("geopandas")
    tf = (2.0, 0.0, 1000.0, 0.0, -2.0, 2000.0)
    df = gpd.read_file(_points(tf).write(tmp_path / "p.gpkg"))
    assert df.crs == CRS
    assert df["score"].tolist() == pytest.approx([0.9, 0.6])
    assert (df.geometry.x[0], df.geometry.y[0]) == pytest.approx((1007.0, 1995.0))
    assert gpd.read_file(_points().write(tmp_path / "q.gpkg")).crs is None


def test_calibrate_threshold_accepts_batched_point_tensor():
    # a (B, N, 2) tensor is one point set per image, the same as the list of them
    pts = torch.tensor([[[5.0, 5.0], [20.0, 12.0]], [[10.0, 10.0], [3.0, 20.0]]])
    hm = gaussian_heatmap(list(pts), (32, 32))
    assert calibrate_threshold(hm, pts, metric="f1") == calibrate_threshold(
        hm, list(pts), metric="f1")
