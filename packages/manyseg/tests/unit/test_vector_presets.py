"""Training presets for vector outputs: head presets, road targets (vs the E044 encoder) and
dihedral augmentation (a transformed target equals the target of the transformed geometry)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("rasterio")

import networkx as nx  # noqa: E402
import shapely  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg.vectors import (  # noqa: E402
    VectorGraph,
    corner_line_heads,
    corner_targets,
    dihedral,
    dihedral_geometry,
    dihedral_xy,
    road_heads,
    road_targets,
)
from manyseg.vectors.augment import HFLIP, TRANSPOSE, VFLIP, dihedral_shape  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"
PRIOR = math.log(0.1 / 0.9)


def test_head_presets():
    h = corner_line_heads()
    assert {k: (v["type"], v["out_channels"]) for k, v in h.items()} == {
        "extent": ("segmentation", 1), "boundary": ("segmentation", 1),
        "distance": ("regression", 1), "corner": ("segmentation", 1),
        "direction": ("regression", 2)}
    r = road_heads(decoders={"keypoints": "unet"})
    assert {k: (v["type"], v["out_channels"]) for k, v in r.items()} == {
        "mask": ("segmentation", 1), "line": ("segmentation", 1),
        "keypoints": ("segmentation", 3), "direction": ("regression", 4)}
    assert r["keypoints"]["decoder"] == "unet" and "decoder" not in r["mask"]
    assert all(v["decoder"] == "unet" for v in corner_line_heads("separate").values())
    with pytest.raises(ValueError, match="unknown heads"):
        road_heads(decoders={"kp": "unet"})
    for heads, name in ((h, "corner"), (r, "keypoints")):  # CenterNet prior, others at 0
        m = create_model(backbone="resnet18", pretrained=False, heads=heads)
        for k, head in m.heads.items():
            want = PRIOR if k == name else 0.0
            assert torch.allclose(head.conv.bias, torch.full_like(head.conv.bias, want)), k
        out = m.eval()(torch.randn(1, 3, 64, 64))
        assert {k: v.shape[1] for k, v in out.items()} == {
            k: v["out_channels"] for k, v in heads.items()}


# --- dihedral transforms ----------------------------------------------------------------------

def test_named_transforms():
    a = np.arange(12).reshape(3, 4)
    assert np.array_equal(dihedral(a, HFLIP), a[:, ::-1])
    assert np.array_equal(dihedral(a, VFLIP), a[::-1])
    assert np.array_equal(dihedral(a, TRANSPOSE), a.T)
    for d in range(8):  # torch and numpy agree, shapes as dihedral_shape says
        t = dihedral(torch.from_numpy(a), d)
        assert np.array_equal(t.numpy(), dihedral(a, d))
        assert t.shape == dihedral_shape(a.shape, d)
    with pytest.raises(ValueError):
        dihedral(a, 8)


@pytest.mark.parametrize("shape", [(5, 5), (4, 7)])
def test_points_move_with_pixels(shape):
    """A point inside pixel (r, c) lands inside the pixel that pixel moves to."""
    H, W = shape
    rng = np.random.default_rng(0)
    for d in range(8):
        for _ in range(10):
            r, c = int(rng.integers(H)), int(rng.integers(W))
            a = np.zeros(shape)
            a[r, c] = 1
            r2, c2 = np.argwhere(dihedral(a, d))[0]
            x, y = dihedral_xy([c + rng.uniform(0.1, 0.9), r + rng.uniform(0.1, 0.9)], d, shape)
            assert (int(math.floor(y)), int(math.floor(x))) == (r2, c2)


def test_direction_values():
    """Lines at angle t: the transformed field holds the transformed line's (cos 2t, sin 2t)."""
    for t in np.linspace(0, math.pi, 7, endpoint=False):
        f = np.array([math.cos(2 * t), math.sin(2 * t)], np.float32).reshape(2, 1, 1)
        u = np.array([[0.0, 0.0], [math.cos(t), math.sin(t)]])
        for d in range(8):
            v = dihedral_xy(u, d, (1, 1))
            t2 = math.atan2(*(v[1] - v[0])[::-1])
            got = dihedral(f, d, direction=True)[:, 0, 0]
            assert np.allclose(got, [math.cos(2 * t2), math.sin(2 * t2)], atol=1e-6), (t, d)
            assert np.allclose(dihedral(torch.from_numpy(f), d, direction=True)[:, 0, 0], got)


def _polygons(rng, size):
    out = []
    for _ in range(5):  # random rotated rectangles and an L shape (generic float coordinates)
        cx, cy = rng.uniform(12, size - 12, 2)
        w, h = rng.uniform(6, 14, 2)
        p = shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        out.append(shapely.affinity.rotate(p, rng.uniform(0, 180), origin="centroid"))
    out.append(shapely.Polygon([(3.3, 3.7), (20.2, 3.1), (20.6, 9.4), (9.9, 9.8), (10.4, 21.1),
                                (3.6, 21.5)]))
    return out


def test_corner_targets_commute_with_dihedral():
    size, rng = 64, np.random.default_rng(1)
    polys = _polygons(rng, size)
    t0 = corner_targets(polys, (size, size))
    assert t0["direction_mask"].sum() > 200 and (t0["corner"] > 0.5).sum() >= 10
    for d in range(8):
        t1 = corner_targets([dihedral_geometry(p, d, (size, size)) for p in polys], (size, size))
        assert np.allclose(dihedral(t0["corner"], d), t1["corner"], atol=1e-6), d
        assert np.array_equal(dihedral(t0["direction_mask"], d), t1["direction_mask"]), d
        assert np.allclose(dihedral(t0["direction"], d, direction=True), t1["direction"],
                           atol=1e-6), d


def _road_graph(size, rng, crossing=True):
    """A small network: a junction, a dead end, a bend, and (optionally) an overpass."""
    g = nx.Graph()
    j = np.array([size * 0.45, size * 0.5]) + rng.uniform(-1, 1, 2)
    pts = {"j": j, "a": [0.0, j[1] + 3.3], "b": [size * 0.8 + 0.37, size * 0.21 + 0.11],
           "c": [j[0] - 1.7, size - 0.0], "e": [size * 0.75 + 0.29, size * 0.83 + 0.41],
           "m": [size * 0.6 + 0.13, size * 0.7 + 0.27]}
    for k, p in pts.items():
        g.add_node(k, xy=np.asarray(p, float))
    for u, v in (("j", "a"), ("j", "b"), ("j", "c"), ("j", "m"), ("m", "e")):
        g.add_edge(u, v, width=6.0)
    cr = []
    if crossing:  # an overpass across the j-b road: no node where they cross
        g.add_node("p", xy=np.array([size * 0.55 + 0.21, 0.0]))
        g.add_node("q", xy=np.array([size * 0.71 + 0.33, size * 0.0 + float(size)]))
        g.add_edge("p", "q", width=8.0)
        cr = [[size * 0.6, size * 0.33]]
    return g, cr


def test_road_targets_basic():
    size = 96
    g, cr = _road_graph(size, np.random.default_rng(2))
    t = road_targets(g, (size, size), crossings=cr)
    assert {k: v.shape for k, v in t.items()} == {
        "mask": (size, size), "line": (size, size), "keypoints": (3, size, size),
        "direction": (4, size, size)}
    assert all(v.dtype == np.float32 for v in t.values())
    peaks = [(ch > 0.6).sum() for ch in t["keypoints"]]
    assert all(p > 0 for p in peaks)  # one junction, one in-image end (e), one crossing
    assert (t["line"] <= t["mask"]).all()  # centreline band inside the road surface
    on = t["mask"] > 0
    assert np.allclose(np.hypot(t["direction"][0], t["direction"][1])[on], 1, atol=1e-5)
    second = np.abs(t["direction"][2:]).sum(0) > 0
    assert second.sum() > 20 and (second <= on).all()
    # empty graph and a VectorGraph with per-edge widths
    e = road_targets(nx.Graph(), (8, 8), crossings=[[4, 4]])
    assert e["mask"].sum() == 0 and e["keypoints"][2].max() > 0.5
    vg = VectorGraph([[2.5, 10.2], [30.4, 10.7]], [[0, 1]])
    thin, wide = (road_targets(vg, (24, 40), widths=[w])["mask"].sum() for w in (2.0, 8.0))
    assert 0 < thin < wide
    assert road_targets(vg, (24, 40), width=8.0)["mask"].sum() == wide


def _transform_graph(g, d, shape):
    h = nx.Graph()
    for n, a in g.nodes(data=True):
        h.add_node(n, xy=dihedral_xy(a["xy"], d, shape))
    h.add_edges_from(g.edges(data=True))
    return h


def test_road_targets_commute_with_dihedral():
    size = 96
    g, cr = _road_graph(size, np.random.default_rng(3))
    t0 = road_targets(g, (size, size), crossings=cr)
    for d in range(8):
        t1 = road_targets(_transform_graph(g, d, (size, size)), (size, size),
                          crossings=dihedral_xy(cr, d, (size, size)))
        for k in ("mask", "line", "keypoints"):
            assert np.allclose(dihedral(t0[k], d), t1[k], atol=1e-6), (k, d)
        assert np.allclose(dihedral(t0["direction"], d, direction=True), t1["direction"],
                           atol=1e-6), d


# --- the E044 encoder --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e044():
    if not (EXPERIMENTS / "E044_cityscale_unified" / "data.py").exists():
        pytest.skip("not a repository checkout (no experiments/)")
    sys.path[:0] = [str(EXPERIMENTS / "E044_cityscale_unified")]
    try:
        import data
    except ImportError as e:  # pragma: no cover - experiment dependencies missing
        pytest.skip(f"E044 encoder not importable: {e}")
    finally:
        sys.path.remove(str(EXPERIMENTS / "E044_cityscale_unified"))
    return data


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_road_targets_match_e044_encode_fast(e044, seed):
    """Same graph, same 1024 px tile: identical targets to E044's training encoder."""
    T = e044.T
    rng = np.random.default_rng(seed)
    g = nx.Graph()
    xy = rng.uniform(-50, T + 50, (12, 2))  # some nodes outside: clipped roads, border ends
    for i, p in enumerate(xy):
        g.add_node(i, xy=p)
    for i in range(1, 12):  # a random tree plus a cycle: junctions, ends, chains
        g.add_edge(i, int(rng.integers(i)), width=float(rng.choice([6.0, 9.0])))
    g.add_edge(3, 9, width=6.0)
    cr = rng.uniform(0, T, (2, 2))
    ref = e044.encode_fast(SimpleNamespace(graph=g, crossings=list(cr)))
    got = road_targets(g, (T, T), crossings=cr)
    assert np.array_equal(got["mask"], ref["mask"])
    assert np.array_equal(got["line"], ref["line"])
    for i, c in enumerate(("junction", "end", "crossing")):
        assert np.array_equal(got["keypoints"][i], ref[c]), c
    assert np.array_equal(got["direction"], ref["direction"])
