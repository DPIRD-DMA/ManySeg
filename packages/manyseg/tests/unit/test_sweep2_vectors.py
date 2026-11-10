"""Regression tests for the second sweep's vector and point fixes: road ``min_length``, Z kept
through georeferencing and augmentation, ``fill_holes`` around another instance, dihedral
exactness off ties, density decoders sharing ``density_loss``'s scale, ``line_radius`` and a
georeferenced road graph from ``decode_scene``."""

import json
import math

import numpy as np
import pytest
import torch

shapely = pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")
pytest.importorskip("rasterio")

from manyseg.points import (  # noqa: E402
    DensityBlockDecoder,
    count_from_density,
    density_loss,
    density_map,
)
from manyseg.vectors import (  # noqa: E402
    Scene,
    SceneRecords,
    block_decoder,
    corner_targets,
    decode_scene,
    dihedral,
    dihedral_geometry,
    dihedral_shape,
    field_targets,
    footprint_targets,
    graph_from_road_heads,
    pixel_to_world,
    rasterize_instances,
    records_to_geojson,
    road_graph,
    road_targets,
    to_geojson,
    world_to_pixel,
    write_records,
)
from manyseg.vectors.blocks import _world  # noqa: E402

T = (10.0, 0.0, 500_000.0, 0.0, -10.0, 6_000_000.0)


# --- V1: the road-heads decoder honours min_length ----------------------------------------------


def _short_road():
    g, _ = road_graph([shapely.LineString([(100.0, 100.0), (116.0, 100.0)])])  # 16 px
    return road_targets(g, (256, 256))


@pytest.mark.parametrize("min_length", [0, 5])
def test_road_heads_keep_pieces_above_min_length(min_length):
    t = _short_road()
    vg = graph_from_road_heads(t["mask"], t["keypoints"], t["direction"], min_length=min_length,
                               spur=0)
    assert len(vg.edges) > 0  # used to be dropped by a hard-coded 20 px
    fields = {k: t[k] for k in ("mask", "keypoints", "direction")}
    blk = decode_scene(fields, block_decoder("roads", min_length=min_length, spur=0))
    assert len(blk.edges) == len(vg.edges)


def test_road_heads_drop_pieces_below_min_length():
    t = _short_road()
    vg = graph_from_road_heads(t["mask"], t["keypoints"], t["direction"], min_length=40, spur=0)
    assert len(vg.edges) == 0


# --- V2: Z coordinates survive georeferencing and dihedral transforms --------------------------


def test_pixel_to_world_keeps_z():
    p3 = shapely.Polygon([(0, 0, 5), (10, 0, 6), (10, 10, 7)])
    w = pixel_to_world(p3, T)
    assert w.has_z
    np.testing.assert_allclose(shapely.get_coordinates(w, include_z=True)[:3],
                               [[500000, 6000000, 5], [500100, 6000000, 6], [500100, 5999900, 7]])
    back = world_to_pixel(w, T)
    np.testing.assert_allclose(shapely.get_coordinates(back, include_z=True),
                               shapely.get_coordinates(p3, include_z=True))
    assert not pixel_to_world(shapely.Point(1, 2), T).has_z  # 2-D stays 2-D
    xyz = pixel_to_world(np.array([[1.0, 2.0, 3.0]]), T)
    np.testing.assert_allclose(xyz, [[500010, 5999980, 3]])


def test_geojson_writers_keep_z_and_agree():
    p3 = shapely.Polygon([(0, 0, 5), (10, 0, 6), (10, 10, 7)])
    c = to_geojson([p3], T)["features"][0]["geometry"]["coordinates"][0][1]
    assert c == [500100.0, 6000000.0, 6.0]
    recs = SceneRecords([{"id": 1, "geometry": p3}], transform=T, crs="EPSG:32750")
    assert records_to_geojson(recs)["features"][0]["geometry"]["coordinates"][0][1] == c
    p2 = shapely.Polygon([(0.5, 0.25), (10, 0), (3, 7)])
    from affine import Affine

    for tr in (T, Affine(*T)):
        assert _world(p2, tr).equals_exact(pixel_to_world(p2, T), 1e-9)


def test_dihedral_geometry_keeps_z():
    ln = shapely.LineString([(1, 2, 10), (5, 6, 20)])
    for d in range(8):
        out = dihedral_geometry(ln, d, (32, 40))
        assert out.has_z
        np.testing.assert_array_equal(shapely.get_coordinates(out, include_z=True)[:, 2],
                                      [10, 20])


# --- V3: fill_holes keeps a hole that holds another instance ------------------------------------


def _ring_and_island():
    outer = shapely.Polygon([(2, 2), (30, 2), (30, 30), (2, 30)],
                            [[(8, 8), (24, 8), (24, 24), (8, 24)]])
    inner = shapely.box(12, 12, 20, 20)
    return outer, inner


def test_fill_holes_keeps_the_background_around_another_instance():
    outer, inner = _ring_and_island()
    plain, _ = rasterize_instances([outer, inner], (32, 32))
    filled, _ = rasterize_instances([outer, inner], (32, 32), fill_holes=True)
    np.testing.assert_array_equal(filled, plain)
    assert (filled[8:24, 8:24] == 0).sum() == 16 * 16 - 8 * 8
    # a hole with only background is still filled
    alone, _ = rasterize_instances([outer], (32, 32), fill_holes=True)
    assert (alone[2:30, 2:30] == 1).all()


def test_field_targets_keep_a_field_inside_another_surrounded():
    outer, inner = _ring_and_island()
    t = field_targets([outer, inner], (32, 32), min_area=1)
    lab = t["instances"]
    assert len(np.unique(lab)) == 3
    assert (lab[8:24, 8:24] == 0).sum() == 16 * 16 - 8 * 8
    assert t["extent"][10, 10] == 0  # the ring between the fields is background


# --- V4: dihedral transforms are exact for geometry off pixel-centre ties ----------------------


def test_dihedral_targets_exact_without_ties():
    H, W = 48, 64
    polys = [shapely.Polygon([(10.2, 12.3), (40.1, 12.3), (40.1, 30.4), (25.3, 30.4),
                              (25.3, 40.2), (10.2, 40.2)]),
             shapely.Polygon([(45.3, 5.2), (60.1, 9.7), (55.4, 25.8), (44.1, 20.1)])]
    t = footprint_targets(polys, (H, W))
    for d in range(8):
        tp = footprint_targets([dihedral_geometry(p, d, (H, W)) for p in polys],
                               dihedral_shape((H, W), d))
        for k in t:
            np.testing.assert_allclose(dihedral(t[k], d, direction=k == "direction"), tp[k],
                                       atol=1e-5, err_msg=f"{k} d={d}")


def test_augment_docstring_names_the_tie_exception():
    import manyseg.vectors.augment as aug

    assert "except at ties" in aug.__doc__


# --- V5: density decoders take density_loss's sigma / value_scale ------------------------------


def _scaled_density(sigma=3.0):
    pts = [torch.tensor([[10.0, 10.0], [30.0, 40.0], [50.0, 20.0]])]
    target = density_map(pts, (64, 64), sigma) * (2 * math.pi * sigma**2)  # density_loss default
    assert float(density_loss(target, pts, sigma=sigma)) == pytest.approx(0.0, abs=1e-10)
    return pts, target


def test_count_from_density_composes_with_density_loss():
    _, target = _scaled_density()
    assert float(count_from_density(target, sigma=3.0)[0]) == pytest.approx(3.0, abs=1e-3)
    vs = 2 * math.pi * 9
    assert float(count_from_density(target, value_scale=vs)[0]) == pytest.approx(3.0, abs=1e-3)
    cnt = decode_scene({"density": target[0].numpy()}, block_decoder("density", sigma=3.0),
                       block=32, halo=0)
    assert cnt.total == pytest.approx(3.0, abs=1e-3)
    assert DensityBlockDecoder(sigma=3.0).count_scale == pytest.approx(vs)


def test_density_decoders_need_exactly_one_scale():
    d = torch.ones(1, 1, 4, 4)
    for kw in ({}, {"sigma": 3.0, "value_scale": 2.0}):
        with pytest.raises(ValueError, match="exactly one of sigma"):
            count_from_density(d, **kw)
        with pytest.raises(ValueError, match="exactly one of sigma"):
            DensityBlockDecoder(**kw)
        with pytest.raises(ValueError, match="exactly one of sigma"):
            block_decoder("density", **kw)


def test_per_point_sigma_error_matches_density_loss():
    pred = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError) as loss_err:
        density_loss(pred, [torch.zeros(1, 2)], sigma=[torch.tensor([2.0])])
    with pytest.raises(ValueError) as count_err:
        count_from_density(pred, sigma=torch.tensor([2.0]))
    with pytest.raises(ValueError) as dec_err:
        DensityBlockDecoder(sigma=[2.0, 3.0])
    assert str(loss_err.value) == str(count_err.value) == str(dec_err.value)


# --- V6: corner_targets' band is line_radius (per side) ----------------------------------------


def test_corner_targets_line_radius():
    box = shapely.box(10.2, 10.2, 50.2, 40.2)
    for r in (1.0, 2.0):
        t = corner_targets([box], (64, 64), line_radius=r)
        rows = np.nonzero(t["direction_mask"][:25, 30])[0]
        # pixel centres within r of the wall at y = 10.2
        want = [i for i in range(25) if abs(i + 0.5 - 10.2) <= r]
        assert rows.tolist() == want
    with pytest.raises(TypeError):
        corner_targets([box], (64, 64), line_width=1.0)


# --- V7: a georeferenced road graph from decode_scene -------------------------------------------


def _road_fields():
    g, _ = road_graph([shapely.LineString([(0, 50.0), (100, 50.0)])])
    t = road_targets(g, (100, 100))
    return {k: t[k] for k in ("mask", "keypoints", "direction")}


def test_decode_scene_roads_carry_transform_and_crs(tmp_path):
    fields = _road_fields()
    scene = Scene(100, 100, T, "EPSG:32750")
    vg = decode_scene(fields, block_decoder("roads"), scene=scene)
    assert vg.transform == T and vg.crs == "EPSG:32750"
    fc = records_to_geojson(vg)
    assert fc["crs"]["properties"]["name"] == "EPSG:32750"
    xs = [c[0] for f in fc["features"] for c in f["geometry"]["coordinates"]]
    assert min(xs) > 500_000 - 1  # world, not pixel coordinates
    assert vg.to_geojson() == fc
    path = write_records(vg, tmp_path / "roads.geojson")
    assert json.loads(path.read_text()) == fc
    # pixel coordinates without georeferencing; an explicit transform=None still wins
    plain = decode_scene(fields, block_decoder("roads"))
    assert plain.transform is None and plain.crs is None
    assert "crs" not in records_to_geojson(plain)
    assert "crs" not in records_to_geojson(vg, transform=None)
    np.testing.assert_allclose(plain.xy, vg.xy)
