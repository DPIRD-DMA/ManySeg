"""A0 round-trip cases for P8 / P8r (run: uv run pytest -q test_e037.py)."""
import math

import networkx as nx
import numpy as np
from shapely.geometry import box

import e037 as P

S = (128, 128)


def g_of(lines):
    return P.lines_to_graph([np.asarray(l, float) for l in lines])


def decode(g, displace=False):
    enc = P.encode_p8(g, S, displace)
    out = P.finish(P.pixel_graph(enc.node, enc.link, enc.offset), 0.5)
    return enc, out


def junction_degrees(g):
    return sorted(d for _, d in g.degree() if d != 2)


def test_straight_and_diagonal_round_trip():
    for line in ([(10.3, 20.7), (100.2, 20.7)], [(10.5, 10.5), (90.5, 70.2)], [(5.2, 100.1), (60.8, 7.4)]):
        enc, out = decode(g_of([line]))
        assert enc.collisions == 0
        xy = sorted(map(tuple, np.round([out.nodes[n]["xy"] for n in out if out.degree(n) == 1], 3)))
        assert np.allclose(xy, sorted(map(tuple, np.round(line, 3))), atol=1e-3)


def test_y_junction_60deg():
    c = (64.4, 64.6)
    arms = [(c[0] + 40 * math.cos(math.radians(a)), c[1] + 40 * math.sin(math.radians(a))) for a in (0, 120, 240)]
    enc, out = decode(g_of([[c, a] for a in arms]))
    assert enc.collisions == 0 and junction_degrees(out) == [1, 1, 1, 3]


def test_acute_fork_collides_and_displacement_counts():
    c = (20.5, 64.5)
    arms = [(c[0] + 90 * math.cos(math.radians(a)), c[1] + 90 * math.sin(math.radians(a))) for a in (-5, 5)]
    arms.append((5.5, 64.5))
    enc, out = decode(g_of([[c, a] for a in arms]))
    assert enc.collisions > 0  # 10 deg fork: both arms share pixels after the junction
    enc_r, out_r = decode(g_of([[c, a] for a in arms]), displace=True)
    assert enc_r.collisions <= enc.collisions


def test_x_crossing_without_node():
    g = g_of([[(10.5, 10.5), (110.5, 110.5)], [(10.5, 110.5), (110.5, 10.5)]])
    enc, out = decode(g)
    # either a shared pixel (collision -> false junction) or a preserved diagonal crossing
    assert (enc.collisions > 0) == (max(dict(out.degree()).values()) > 2)


def test_parallel_lines_one_px_apart_collide():
    enc, out = decode(g_of([[(10.2, 50.2), (110.2, 50.2)], [(10.2, 51.3), (110.2, 51.3)]]))
    assert enc.collisions == 0 and nx.number_connected_components(out) == 2  # adjacent rows, no links
    enc, out = decode(g_of([[(10.2, 50.2), (110.2, 50.2)], [(10.2, 50.7), (110.2, 50.7)]]))
    assert enc.collisions > 0  # same pixel row


def test_square_ring_faces():
    poly = box(20.5, 30.5, 90.5, 80.5)
    g = g_of([np.asarray(poly.exterior.coords)])
    enc = P.encode_p8(g, S)
    mask = np.zeros(S, np.float32)
    mask[31:80, 21:90] = 1
    pg = P.pixel_graph(enc.node, enc.link, enc.offset)
    lines = P.finish(pg, 0.0)
    import shapely
    from shapely.ops import polygonize, unary_union
    from shapely.geometry import LineString
    faces = list(polygonize(unary_union([LineString([lines.nodes[u]["xy"], lines.nodes[v]["xy"]]) for u, v in lines.edges()])))
    assert len(faces) == 1 and faces[0].symmetric_difference(poly).area < 1.0


def test_geo_raster_round_trips_and_is_stable():
    import run_a2 as A
    for line in ([(10.3, 20.7), (100.2, 20.7)], [(10.5, 10.5), (90.5, 70.2)], [(5.2, 100.1), (60.8, 7.4)]):
        g = g_of([line])
        enc = P.encode_p8(g, S, raster="geo")
        out = P.finish(P.pixel_graph(enc.node, enc.link, enc.offset), 0.5)
        xy = sorted(map(tuple, np.round([out.nodes[n]["xy"] for n in out if out.degree(n) == 1], 3)))
        assert enc.collisions == 0 and np.allclose(xy, sorted(map(tuple, np.round(line, 3))), atol=1e-3)
    g = g_of([[(10.3, 20.7), (100.2, 60.1), (30.0, 90.0)]])  # a vertex exactly on a pixel corner
    base = A.undirected_links(P.encode_p8(g, S, raster="geo").link)
    enc = P.encode_p8(A.transform_graph(g, A.dihedral(4, S[0])), S, raster="geo")
    assert A.disagreement(base, A.links_back(A.undirected_links(enc.link), A._inverse_k(4), S[0])) < 0.05
