"""Vector metrics (APLS, GEO, junctions, crossings, polygons), including the regressions from
the review of the E028 definitions."""

import numpy as np
import pytest

nx = pytest.importorskip("networkx")
pytest.importorskip("shapely")
pytest.importorskip("scipy")

from manyseg.vectors.metrics import (  # noqa: E402
    apls,
    crossing_score,
    geo_f1,
    graph_metrics,
    junction_f1,
    polygon_metrics,
)
from shapely.geometry import Polygon, box  # noqa: E402


def graph(points, edges, shift=(0.0, 0.0)):
    g = nx.Graph()
    for i, p in enumerate(points):
        g.add_node(i, xy=np.asarray(p, float) + shift)
    g.add_edges_from(edges)
    return g


def grid(step=200, lo=112, hi=912):
    xs = np.arange(lo, hi + 1, step)
    n = len(xs)
    edges = [(j * n + i, j * n + i + 1) for j in range(n) for i in range(n - 1)]
    edges += [(j * n + i, (j + 1) * n + i) for j in range(n - 1) for i in range(n)]
    return graph([(x, y) for y in xs for x in xs], edges)


def test_apls_identity_shift_cut_and_empty():
    g = grid()
    assert apls(g, g) == pytest.approx(1.0)
    shifted = grid()
    for n in shifted:
        shifted.nodes[n]["xy"] = shifted.nodes[n]["xy"] + (2.0, 0.0)
    assert apls(g, shifted) > 0.99
    cut = g.copy()
    cut.remove_edges_from([(0, 1), (5, 6)])
    assert 0.5 < apls(g, cut) < 0.99
    assert apls(g, nx.Graph()) == 0.0


def test_apls_parallel_chains_identity():
    g = graph([(300, 300), (320, 300), (310, 300), (310, 310), (310, 290)],
              [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)])
    assert apls(g, g) == pytest.approx(1.0)


def test_apls_short_loop_identity():
    g = graph([(300, 300), (310, 300), (310, 310), (300, 310)], [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert apls(g, g) == pytest.approx(1.0)


def test_apls_accepts_array_graphs():
    class G:
        xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        edges = np.array([[0, 1], [1, 2]])

    assert apls(G(), G()) == pytest.approx(1.0)


def test_geo_weights_by_length_and_empty_gt():
    a = graph([(100 + 0.1 * k, 100) for k in range(1001)], [(k, k + 1) for k in range(1000)])
    b = graph([(100, 300), (200, 300)], [(0, 1)])
    m = geo_f1(nx.disjoint_union(a, b), a)
    assert m["geo_r"] == pytest.approx(0.5, abs=0.02) and m["geo_p"] == pytest.approx(1.0)
    assert geo_f1(nx.Graph(), a)["geo_f1"] == 0.0


def test_junctions_and_gsd():
    g = grid()
    assert junction_f1(g, g) == 1.0
    moved = grid()
    for n in moved:
        moved.nodes[n]["xy"] = moved.nodes[n]["xy"] + (8.0, 0.0)
    assert junction_f1(g, moved, gsd=1.0) == 0.0  # 8 px > 5 m at 1 m
    assert junction_f1(g, moved, gsd=0.5) == 1.0  # 8 px = 4 m
    assert set(graph_metrics(g, g)) == {"apls", "junction_f1", "geo_p", "geo_r", "geo_f1"}


def test_crossing_score():
    road = graph([(0, 50), (100, 50)], [(0, 1)])
    bridge_junction = graph([(0, 50), (50, 50), (100, 50), (50, 0), (50, 100)],
                            [(0, 1), (1, 2), (3, 1), (1, 4)])
    s = crossing_score([(50, 50)], road)
    assert s == {"n": 1, "no_junction": 1, "covered": 1, "covered_ok": 1}
    s = crossing_score([(50, 50)], bridge_junction)
    assert s["no_junction"] == 0 and s["covered_ok"] == 0
    assert crossing_score([(50, 50)], nx.Graph())["covered"] == 0


def test_polygon_metrics():
    gt = [box(0, 0, 10, 10), box(20, 0, 30, 10), box(50, 50, 60, 60)]
    m = polygon_metrics(gt, gt)
    assert m["det_f1"] == 1.0 and m["ciou"] == pytest.approx(1.0)
    assert m["polis"] == pytest.approx(0.0)
    # same footprint with an extra collinear vertex in the prediction: N 5 vs 4
    extra = Polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])
    m = polygon_metrics([gt[0]], [extra])
    assert m["ciou"] == pytest.approx(1 - 1 / 9) and m["n_ratio"] == pytest.approx(5 / 4)
    # GT collinear nodes are not counted
    m = polygon_metrics([extra], [gt[0]])
    assert m["ciou"] == pytest.approx(1.0)
    m = polygon_metrics(gt, gt[:1], gsd=0.5)
    assert m["det_f1"] == pytest.approx(0.5) and m["tp"] == 1
    m = polygon_metrics(gt, [box(0, 0, 10, 4)])  # IoU 0.4: no match
    assert m["tp"] == 0 and m["ciou"] == 0.0
