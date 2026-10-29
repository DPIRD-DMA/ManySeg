"""Graph tensor (experiments/E030_whu_polygons/graph_tensor.py): encoder and decoder tests.

Moved from the library's tests/unit/test_vectors.py and test_peaks.py when the graph tensor left
``manyseg.vectors`` before 0.1.0. Run from the repo root:
``uv run pytest experiments/E030_whu_polygons/test_graph_tensor.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402


def buildings():
    return [box(100, 100, 200, 160), box(300, 100, 380, 220), box(200, 160, 260, 200),
            Polygon([(500, 500), (600, 500), (600, 560), (560, 560), (560, 620), (500, 620)]),
            box(980, 300, 1024, 380)]  # clipped by the image border


def test_graph_tensor_round_trip():
    from graph_tensor import encode_graph_tensor, polygons_from_tensor, vertex_graph

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
    """This encoder gives E028's targets (whose decoding the oracle measured)."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "E028_vector_oracle" / "e028.py"
    pytest.importorskip("rasterio")  # E028's own dependency
    import sys

    spec = importlib.util.spec_from_file_location("_e028_oracle", path)
    e028 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = e028  # dataclasses look their module up
    try:
        spec.loader.exec_module(e028)
    finally:
        sys.modules.pop(spec.name, None)
    from graph_tensor import encode_graph_tensor, vertex_graph

    vg = vertex_graph(buildings(), border=(1024, 1024))
    ours = encode_graph_tensor(vg, (1024, 1024))
    theirs = e028.graph_tensor(e028.VertexGraph(vg.xy, vg.edges))
    np.testing.assert_allclose(ours["vertex"], theirs["vertex"], atol=1e-5)
    np.testing.assert_array_equal(ours["edge"], theirs["edge"])
    np.testing.assert_allclose(ours["vector"], theirs["vec"], atol=1e-4)


def test_graph_tensor_decoding_is_bounded_on_noise():
    from graph_tensor import graph_from_tensor, polygons_from_tensor

    rng = np.random.default_rng(0)
    v = rng.random((256, 256)).astype(np.float32)  # an untrained model's vertex map
    e = rng.random((6, 256, 256)).astype(np.float32)
    vec = rng.normal(0, 10, (6, 2, 256, 256)).astype(np.float32)
    vec = rng.normal(0, 100, (6, 2, 256, 256)).astype(np.float32)  # long random vectors
    g = graph_from_tensor(v, e, vec, vertex_threshold=0.1, max_vertices=300)
    assert len(g.xy) <= 300
    d = g.xy[g.edges[:, 0]] - g.xy[g.edges[:, 1]]
    assert len(d) == 0 or np.hypot(d[:, 0], d[:, 1]).max() <= 60 + 1e-6
    polygons_from_tensor(v, e, vec, rng.random((256, 256)), vertex_threshold=0.1, max_vertices=300)


def _encode_loop_reference(vg, shape, sectors=6, radius=3.0, sigma=1.5):
    """The original per-vertex loop encoder (kept to check the vectorised one)."""
    import math

    H, W = shape
    V = len(vg.xy)
    nb = np.zeros((V + 1, sectors, 2), np.float32)
    has = np.zeros((V + 1, sectors), bool)
    width = 2 * math.pi / sectors
    adj = [[] for _ in range(V)]
    for a, b in vg.edges:
        adj[a].append(b)
        adj[b].append(a)
    for v in range(V):
        for u in adj[v]:
            d = vg.xy[u] - vg.xy[v]
            ang = math.atan2(d[1], d[0]) % (2 * math.pi)
            own = int(ang // width) % sectors
            side = 1 if ang / width - own >= 0.5 else -1
            for k in (own, (own + side) % sectors, (own - side) % sectors):
                if not has[v, k]:
                    has[v, k], nb[v, k] = True, vg.xy[u]
                    break
    best = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), V, np.int64)
    r = int(math.ceil(radius))
    for v, (x, y) in enumerate(vg.xy):
        c0, r0 = int(math.floor(x)) - r, int(math.floor(y)) - r
        cs, rs, ce, re = max(c0, 0), max(r0, 0), min(c0 + 2 * r + 1, W), min(r0 + 2 * r + 1, H)
        if cs >= ce or rs >= re:
            continue
        yy, xx = np.mgrid[rs:re, cs:ce]
        d2 = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2
        win = best[rs:re, cs:ce]
        upd = (d2 <= radius ** 2) & (d2 < win)
        win[upd] = d2[upd]
        owner[rs:re, cs:ce][upd] = v
    support = owner < V
    vertex = np.where(support, np.exp(-best / (2 * sigma ** 2)), 0).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pix = np.stack([xx + 0.5, yy + 0.5], -1).astype(np.float32)
    edge = has[owner].transpose(2, 0, 1).astype(np.float32)
    vec = (nb[owner] - pix[:, :, None, :]) * has[owner][..., None]
    return {"vertex": vertex, "edge": edge, "vector": vec.transpose(2, 3, 0, 1).astype(np.float32), "support": support}


def test_vectorised_encoder_matches_loop():
    from graph_tensor import encode_graph_tensor, vertex_graph
    from shapely import affinity

    rng = np.random.default_rng(0)
    for trial in range(5):
        polys = []
        for _ in range(25):  # random rotated rectangles, some touching or overlapping, some cut by the border
            cx, cy = rng.uniform(-20, 532, 2)
            p = box(cx - rng.uniform(5, 40), cy - rng.uniform(5, 30), cx + rng.uniform(5, 40), cy + rng.uniform(5, 30))
            p = affinity.rotate(p, rng.uniform(0, 90)).intersection(box(0, 0, 512, 512))
            if p.geom_type == "Polygon" and p.area > 20:
                polys.append(p)
        vg = vertex_graph(polys, 20.0, border=(512, 512))
        a, b = encode_graph_tensor(vg, (512, 512)), _encode_loop_reference(vg, (512, 512))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"trial {trial}, {k}")


def test_offset_field_points_and_snapping():
    from graph_tensor import offset_field, snap_points

    pts = np.array([[20.3, 30.7], [40.0, 10.5]])
    f, m = offset_field(pts, (64, 64), radius=6)
    assert m[30, 20] and not m[0, 0]
    # a peak found 4 px off the true point snaps back onto it
    snapped = snap_points(np.array([[23.5, 32.5]]), f)
    np.testing.assert_allclose(snapped[0], pts[0], atol=1e-5)
    # outside the radius there is no target (and no movement)
    assert np.allclose(snap_points(np.array([[60.5, 60.5]]), f), [[60.5, 60.5]])


def test_offset_field_from_mask_lines():
    from graph_tensor import offset_field_from_mask

    g = np.zeros((32, 32), bool)
    g[10, 5:25] = True
    f, m = offset_field_from_mask(g, radius=4)
    assert m[13, 12] and not m[20, 12]
    assert f[1, 13, 12] == -3 and f[0, 13, 12] == 0  # straight up to the line


# --- frozen copies of the peak helpers before manyseg.utils._peaks (do not edit) ---

def _old_peak_pixels(hm, threshold, kernel):  # graph tensor _peak_pixels
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


def _old_refine(hm, pix):  # graph tensor _refine
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


def test_graph_from_tensor_equals_old(monkeypatch):
    import graph_tensor as G
    from graph_tensor import encode_graph_tensor, graph_from_tensor, vertex_graph
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
