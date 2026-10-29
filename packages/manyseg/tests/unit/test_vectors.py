"""manyseg.vectors: line graphs and polygons from dense outputs, GeoJSON export."""

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")

from manyseg.vectors import VectorGraph, graph_from_mask, polygons_from_labels, to_geojson  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402


def road_grid(size=400, lines=(100, 200, 300), width=6):
    m = np.zeros((size, size), np.float32)
    for c in lines:
        m[:, c - width // 2:c + width // 2] = 1
        m[c - width // 2:c + width // 2, :] = 1
    return m


def test_graph_from_mask_grid():
    g = graph_from_mask(road_grid())
    j = g.junctions()
    assert len(j) == 9  # 3 x 3 crossings
    expect = np.array([(x, y) for y in (100, 200, 300) for x in (100, 200, 300)], float)
    d = np.hypot(*(j[:, None] - expect[None]).transpose(2, 0, 1)).min(0)
    assert d.max() < 2.5
    assert g.length == pytest.approx(6 * 400, rel=0.03)


def test_graph_from_mask_torch_scale_and_empty():
    m = road_grid()
    g = graph_from_mask(torch.from_numpy(m), scale=0.5)
    assert np.abs(g.junctions()[:, 0].min() - 200) < 5  # head pixel 100 -> input pixel 200
    assert len(graph_from_mask(np.zeros((64, 64))).edges) == 0
    with pytest.raises(ValueError):
        graph_from_mask(np.zeros((1, 64, 64)))


def test_spur_pruning_and_gap_bridging():
    m = np.zeros((200, 300), np.float32)
    m[97:103, 20:140] = 1
    m[97:103, 160:280] = 1  # a 20 px gap
    m[60:97, 80:83] = 1  # a 37 px side branch
    g = graph_from_mask(m, spur=10)
    assert len(g.junctions()) == 1
    assert len(graph_from_mask(m, spur=50).junctions()) == 0
    bridged = graph_from_mask(m, spur=50, gap=30)
    nx = pytest.importorskip("networkx")
    assert nx.number_connected_components(bridged.to_networkx()) == 1
    assert nx.number_connected_components(graph_from_mask(m, spur=50).to_networkx()) == 2


def test_loop_and_polylines():
    m = np.zeros((200, 200), np.float32)
    m[50:56, 50:150] = m[144:150, 50:150] = 1
    m[50:150, 50:56] = m[50:150, 144:150] = 1
    g = graph_from_mask(m)
    (pl,) = g.polylines()
    assert np.allclose(pl[0], pl[-1])
    assert g.length == pytest.approx(4 * 94, rel=0.05)


def test_polygons_from_labels_exact_and_holes():
    lab = np.zeros((100, 120), np.int32)
    lab[10:30, 20:60] = 1
    lab[50:90, 50:100] = 2
    lab[60:70, 60:70] = 0  # a hole in instance 2
    p1, p2 = polygons_from_labels(lab, simplify=0.6)
    assert p1.normalize().equals(box(20, 10, 60, 30).normalize())
    assert len(p1.exterior.coords) - 1 == 4
    assert len(p2.interiors) == 1
    assert p2.area == pytest.approx(40 * 50 - 100)
    (half,) = polygons_from_labels(torch.from_numpy(lab[:40]), scale=0.5)
    assert half.bounds == pytest.approx((40, 20, 120, 60))


def test_to_geojson_transform():
    g = VectorGraph([(0, 0), (10, 0)], [(0, 1)])
    fc = to_geojson(g, transform=(2, 0, 100, 0, -2, 50))
    assert fc["features"][0]["geometry"]["coordinates"] == [[100, 50], [120, 50]]
    fc = to_geojson([Polygon([(0, 0), (1, 0), (1, 1)])], properties=[{"id": 7}])
    assert fc["features"][0]["properties"] == {"id": 7}
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
    with pytest.raises(ValueError):
        to_geojson([Polygon([(0, 0), (1, 0), (1, 1)])], properties=[])


def test_polygons_from_corners_round_trip_and_dent():
    """Perfect targets give exact polygons; a dent in the instance mask does not add corners."""
    from manyseg.vectors import corner_targets, polygons_from_corners

    polys = [box(20, 20, 120, 80),
             Polygon([(150, 150), (250, 150), (250, 200), (200, 200), (200, 250), (150, 250)])]
    t = corner_targets(polys, (300, 300))
    lab = np.zeros((300, 300), np.int32)
    lab[20:80, 20:120] = 1
    lab[150:200, 150:250] = 2
    lab[200:250, 150:200] = 2
    out = polygons_from_corners(lab, t["corner"], t["direction"])
    assert len(out) == 2
    for p in polys:
        q = max(out, key=lambda q: p.intersection(q).area / p.union(q).area)
        assert p.intersection(q).area / p.union(q).area > 0.98
        assert len(q.exterior.coords) == len(p.exterior.coords)
    lab[20:24, 60:66] = 0  # a dent in the top wall
    q = max(polygons_from_corners(lab, t["corner"], t["direction"]),
            key=lambda q: q.intersection(polys[0]).area)
    assert len(q.exterior.coords) - 1 == 4



def test_corner_split_ignores_direction_magnitude():
    """Agreement uses orientation only: a weak field across a straight wall splits it like a
    unit one; a zero field is no evidence (no split)."""
    from manyseg.vectors.corners import _split, _unit

    pts = np.stack([np.arange(20.0), np.zeros(20)], 1)
    across = np.tile([-1.0, 0.0], (20, 1))  # (cos 2t, sin 2t) of a vertical wall
    kw = dict(big_dev=8, agree=0.4)
    assert _split(pts, _unit(across), **kw)
    assert _split(pts, _unit(0.15 * across), **kw) == _split(pts, _unit(across), **kw)
    assert _split(pts, _unit(0 * across), **kw) == []

