"""Sanity tests for the E028 oracle (metrics and level-0 round trips on synthetic data).

    uv run pytest -q experiments/E028_vector_oracle/test_e028.py
"""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon, box

import e028 as E


def line_graph(points, edges, **attrs):
    g = nx.Graph()
    for i, p in enumerate(points):
        g.add_node(i, xy=np.asarray(p, float))
    for a, b in edges:
        g.add_edge(a, b, bridge=False, layer=0, width=6.0, **attrs)
    return g


def grid_graph(step=200, lo=112, hi=912):
    """A road grid: horizontal and vertical roads crossing at shared nodes."""
    xs = np.arange(lo, hi + 1, step)
    pts = [(x, y) for y in xs for x in xs]
    n = len(xs)
    edges = [(j * n + i, j * n + i + 1) for j in range(n) for i in range(n - 1)]
    edges += [(j * n + i, (j + 1) * n + i) for j in range(n - 1) for i in range(n)]
    return line_graph(pts, edges)


def road_tile(g, crossings=()):
    t = E.Tile("synthetic", "roads", 0, 0.0, 0.0, 1.0, graph=g)
    t.crossings = list(crossings)
    return t


def test_clip():
    assert E._clip_t(np.array([-10.0, 5]), np.array([10.0, 5])) == pytest.approx((0.5, 1.0))
    assert E._clip_t(np.array([-10.0, -5]), np.array([-1.0, -5])) is None


def test_chains():
    g = line_graph([(0, 0), (10, 0), (20, 0), (10, 10)], [(0, 1), (1, 2), (1, 3)])
    assert sorted(len(c) for c in E.graph_chains(g)) == [2, 2, 2]
    ring = line_graph([(0, 0), (10, 0), (10, 10), (0, 10)], [(0, 1), (1, 2), (2, 3), (3, 0)])
    (c,) = E.graph_chains(ring)
    assert len(c) == 5 and c[0] == c[-1]


def test_apls_basics():
    g = grid_graph()
    assert E.apls(g, g, 1.0) == pytest.approx(1.0)
    assert E.apls(g, nx.Graph(), 1.0) == 0.0
    shifted = nx.relabel_nodes(g, lambda n: n)
    for n in shifted:
        shifted.nodes[n]["xy"] = g.nodes[n]["xy"] + np.array([2.0, 0])
    assert E.apls(g, shifted, 1.0) > 0.99
    cut = g.copy()
    cut.remove_edge(0, 1)
    cut.remove_edge(5, 6)
    assert 0.5 < E.apls(g, cut, 1.0) < 0.99
    half = g.subgraph([n for n in g if g.nodes[n]["xy"][1] < 500]).copy()
    assert E.apls(g, half, 1.0) < E.apls(g, cut, 1.0)


def test_geo_and_junctions():
    g = grid_graph()
    m = E.geo_f1(g, g, 1.0)
    assert m["geo_f1"] == pytest.approx(1.0)
    assert E.junction_f1(g, g, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("rep", ["R0", "R1", "R2"])
def test_roads_level0_round_trip(rep):
    t = road_tile(grid_graph())
    tg = E.encode(t, rep)
    a = E.corrupt(tg, t, E.LEVELS[0], 0)
    params = {"R0": dict(thr=0.5, spur=20, simplify=1, min_len=20),
              "R1": dict(thr=0.5, spur=20, simplify=1, min_len=20, thr_j=0.4, merge_r=8, kernel=5),
              "R2": dict(thr_v=0.4, thr_e=0.5, link_r=6, mutual=False, kernel=5)}[rep]
    pred = E.DECODERS[rep](a, params)
    m = E.road_metrics(t, pred)
    assert m["apls"] > 0.9, m
    assert m["junction_f1"] > 0.9, m


def test_overpass():
    # a road crossing another without a shared node; only the graph tensor can keep it apart
    g = line_graph([(100, 500), (900, 500), (510, 100), (510, 900)], [(0, 1), (2, 3)])
    t = road_tile(g, E._crossings(g))
    assert len(t.crossings) == 1
    r2 = E.decode_r2(E.encode(t, "R2").arrays, dict(thr_v=0.4, thr_e=0.5, link_r=6, mutual=False, kernel=5))
    r0 = E.decode_r0(E.encode(t, "R0").arrays, dict(thr=0.5, spur=20, simplify=1, min_len=20))
    assert E.crossing_score(t, r2) == (1, 1, 1)
    assert E.crossing_score(t, r0) == (0, 0, 1)


def building_tile():
    polys = [box(100, 100, 200, 160), box(300, 100, 380, 220),
             Polygon([(500, 500), (600, 500), (600, 560), (560, 560), (560, 620), (500, 620)]),
             box(200, 160, 260, 200),  # touches the first at a corner
             box(980, 300, 1024, 380)]  # clipped by the tile border
    return E.Tile("synthetic", "buildings", 0, 0.0, 0.0, 0.5, polygons=polys)


def test_polygon_metrics_identity():
    t = building_tile()
    m = E.polygon_metrics(t, t.polygons)
    assert m["det_f1"] == 1.0 and m["ciou"] == pytest.approx(1.0) and m["polis"] == pytest.approx(0)


@pytest.mark.parametrize("rep", ["B0", "B1", "B2"])
def test_buildings_level0_round_trip(rep):
    t = building_tile()
    tg = E.encode(t, rep)
    a = E.corrupt(tg, t, E.LEVELS[0], 0)
    base = dict(thr=0.5, thr_b=0.5, thr_s=0.3, simplify=1.0)
    params = {"B0": base, "B1": dict(base, thr_c=0.4, snap_r=4, kernel=3),
              "B2": dict(thr_v=0.4, thr_e=0.5, link_r=6, mutual=False, kernel=3, simplify=0.5,
                         face_thr=0.5)}[rep]
    m = E.polygon_metrics(t, E.DECODERS[rep](a, params))
    assert m["det_f1"] == 1.0, m
    assert m["miou"] > 0.9, m
    if rep != "B0":
        assert m["ciou"] > 0.85 and 0.9 < m["n_ratio"] < 1.2, m


def test_corruption_shared_draw():
    t = road_tile(grid_graph())
    cfg = E.LEVELS[2]
    a0 = E.corrupt(E.encode(t, "R0"), t, cfg, 3)
    a1 = E.corrupt(E.encode(t, "R1"), t, cfg, 3)
    np.testing.assert_array_equal(a0["mask"], a1["mask"])
    assert not np.array_equal(a0["mask"], E.corrupt(E.encode(t, "R0"), t, cfg, 4)["mask"])


# Regressions from the Codex review of the metrics (2026-10-26).

def test_apls_parallel_chains_identity():
    g = line_graph([(0, 0), (20, 0), (10, 0), (10, 10), (10, -10)],
                   [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)])
    for n in g:
        g.nodes[n]["xy"] = g.nodes[n]["xy"] + 300
    assert E.apls(g, g, 1.0) == pytest.approx(1.0)


def test_apls_short_cycle_identity():
    g = line_graph([(300, 300), (310, 300), (310, 310), (300, 310)], [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert E.apls(g, g, 1.0) == pytest.approx(1.0)


def test_geo_weights_by_length():
    a = line_graph([(100 + 0.1 * k, 100) for k in range(1001)], [(k, k + 1) for k in range(1000)])
    b = line_graph([(100, 300), (200, 300)], [(0, 1)])
    gt = nx.disjoint_union(a, b)
    m = E.geo_f1(gt, a, 1.0)
    assert m["geo_r"] == pytest.approx(0.5, abs=0.02)
    assert E.geo_f1(nx.Graph(), a, 1.0)["geo_f1"] == 0.0


def test_refine_border_not_biased():
    hm = E.gaussian_points(np.array([[0.2, 2.5]]), sigma=2, shape=(8, 8))
    x, y = E._refine(hm, np.array([[2, 0]]))[0]
    assert x == pytest.approx(0.5) and y == pytest.approx(2.5, abs=0.05)


def test_e035_direction_only_split_and_wraparound():
    """Codex review regressions: direction disagreement splits a straight wall; a split on the
    wall that wraps around the ring keeps the walls in order."""
    import numpy as np
    from shapely.geometry import box

    import e035 as B

    pts = np.array([[float(i), 0.0] for i in range(10)])
    dirs = np.array([[1.0, 0.0]] * 5 + [[-1.0, 0.0]] * 5)  # horizontal, then vertical
    assert B._split(pts, dirs, dev=3, agree=0.8) == [5]
    # a square outline whose corner at the ring's start is not detected: the wrap-around wall
    # must be split there, giving the square back
    sq = np.asarray(box(10, 10, 60, 60).exterior.segmentize(1.0).coords)[:-1]
    sq = np.roll(sq, 25, axis=0)  # ring starts mid-wall
    corners = np.array([[60.0, 10.0], [60.0, 60.0], [10.0, 60.0]])  # (10, 10) missed
    dirs_field = np.zeros((2, 512, 512), np.float32)
    q = B.polygon_b3(np.vstack([sq, sq[:1]]), corners, dirs_field,
                     {"snap_r": 3.0, "dev": 1.5, "agree": 0.0, "simplify": 1.0})
    assert q is not None and abs(q.area - 2500) < 50 and len(q.exterior.coords) - 1 == 4, q
