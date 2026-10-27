"""manyseg.vectors: line graphs and polygons from dense outputs, GeoJSON export."""

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")

from shapely.geometry import Polygon, box  # noqa: E402

from manyseg.vectors import VectorGraph, graph_from_mask, polygons_from_labels, to_geojson  # noqa: E402


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


def buildings():
    return [box(100, 100, 200, 160), box(300, 100, 380, 220), box(200, 160, 260, 200),
            Polygon([(500, 500), (600, 500), (600, 560), (560, 560), (560, 620), (500, 620)]),
            box(980, 300, 1024, 380)]  # clipped by the image border


def test_graph_tensor_round_trip():
    from manyseg.vectors import encode_graph_tensor, polygons_from_tensor, vertex_graph

    polys = buildings()
    vg = vertex_graph(polys, border=(1024, 1024))
    t = encode_graph_tensor(vg, (1024, 1024))
    assert t["edge"].shape == (6, 1024, 1024) and t["vector"].shape == (6, 2, 1024, 1024)
    assert t["support"].sum() > 0 and t["vertex"].max() > 0.8
    extent = np.zeros((1024, 1024), np.float32)
    for p in polys:
        x0, y0, x1, y1 = (int(v) for v in p.bounds)
        extent[y0:y1, x0:x1] = 1  # the L-shape's bounding box is enough for face selection
    out = polygons_from_tensor(t["vertex"], t["edge"], t["vector"], extent, gap=0, link_radius=4,
                               simplify=0.5)
    assert len(out) == len(polys)
    for p in polys:
        best = max(out, key=lambda q: p.intersection(q).area / p.union(q).area)
        assert p.intersection(best).area / p.union(best).area > 0.98
        assert len(best.exterior.coords) == len(p.exterior.coords)


def test_graph_tensor_matches_e028_encoder():
    """The library encoder gives E028's targets (whose decoding the oracle measured)."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parents[4] / "experiments" / "E028_vector_oracle" / "e028.py"
    if not path.exists():
        pytest.skip("E028 code not in this checkout")
    pytest.importorskip("rasterio")  # E028's own dependency
    import sys

    spec = importlib.util.spec_from_file_location("_e028_oracle", path)
    e028 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = e028  # dataclasses look their module up
    try:
        spec.loader.exec_module(e028)
    finally:
        sys.modules.pop(spec.name, None)
    from manyseg.vectors import encode_graph_tensor, vertex_graph

    vg = vertex_graph(buildings(), border=(1024, 1024))
    ours = encode_graph_tensor(vg, (1024, 1024))
    theirs = e028.graph_tensor(e028.VertexGraph(vg.xy, vg.edges))
    np.testing.assert_allclose(ours["vertex"], theirs["vertex"], atol=1e-5)
    np.testing.assert_array_equal(ours["edge"], theirs["edge"])
    np.testing.assert_allclose(ours["vector"], theirs["vec"], atol=1e-4)


def test_graph_tensor_decoding_is_bounded_on_noise():
    from manyseg.vectors import graph_from_tensor, polygons_from_tensor

    rng = np.random.default_rng(0)
    v = rng.random((256, 256)).astype(np.float32)  # an untrained model's vertex map
    e = rng.random((6, 256, 256)).astype(np.float32)
    vec = rng.normal(0, 10, (6, 2, 256, 256)).astype(np.float32)
    g = graph_from_tensor(v, e, vec, vertex_threshold=0.1, max_vertices=300)
    assert len(g.xy) <= 300
    polygons_from_tensor(v, e, vec, rng.random((256, 256)), vertex_threshold=0.1, max_vertices=300)
