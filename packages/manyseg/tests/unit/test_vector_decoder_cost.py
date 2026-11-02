"""Decoder cost on garbage input (an untrained model's noise) stays bounded, and the faster graph
steps give what the plain definitions give; holes (courtyards) survive corner decoding."""

import time

import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("skimage")

import shapely  # noqa: E402
from manyseg.vectors import (  # noqa: E402
    block_decoder,
    decode_scene,
    graph_from_mask,
    graph_from_road_heads,
    polygons_from_corners,
)
from manyseg.vectors import corners as corners_module  # noqa: E402
from manyseg.vectors.graph import (  # noqa: E402
    _chain_len,
    _contract_short,
    _degrees,
    _join_degree2,
    _remove_small,
    _skeleton_chains,
)
from skimage.morphology import skeletonize  # noqa: E402


def contract_short_by_definition(nodes, chains, min_len=3.0):
    changed = True
    while changed:
        changed = False
        for i, (u, v, c) in enumerate(chains):
            if u != v and _chain_len(c) < min_len:
                nodes[u] = (np.asarray(nodes[u]) + np.asarray(nodes[v])) / 2
                new = []
                for j, (a, b, cc) in enumerate(chains):
                    if j == i:
                        continue
                    a2, b2 = (u if a == v else a), (u if b == v else b)
                    cc = cc.copy()
                    if a2 == u:
                        cc[0] = nodes[u]
                    if b2 == u:
                        cc[-1] = nodes[u]
                    new.append((a2, b2, cc))
                del nodes[v]
                chains, changed = new, True
                break
    return nodes, chains


def join_degree2_by_definition(nodes, chains):
    while True:
        deg = _degrees(chains)
        loops = {u for u, v, _ in chains if u == v}
        n2 = next((n for n, d in deg.items() if d == 2 and n not in loops), None)
        if n2 is None:
            return nodes, chains
        ids = [i for i, (a, b, _) in enumerate(chains) if a == n2 or b == n2]
        (a1, b1, c1), (a2, b2, c2) = chains[ids[0]], chains[ids[1]]
        if b1 != n2:
            a1, b1, c1 = b1, a1, c1[::-1]
        if a2 != n2:
            a2, b2, c2 = b2, a2, c2[::-1]
        chains = [c for i, c in enumerate(chains) if i not in ids]
        chains.append((a1, b2, np.concatenate([c1, c2[1:]])))
        nodes.pop(n2, None)


def same_graph(a, b):
    (na, ca), (nb, cb) = a, b
    assert list(na) == list(nb)
    assert all(np.array_equal(na[k], nb[k]) for k in na)
    assert [c[:2] for c in ca] == [c[:2] for c in cb]
    assert all(np.array_equal(x[2], y[2]) for x, y in zip(ca, cb))


def copy_graph(g):
    return dict(g[0]), [(u, v, c.copy()) for u, v, c in g[1]]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("density", [0.35, 0.5, 0.65])
def test_graph_steps_match_their_definitions(seed, density):
    fg = np.random.default_rng(seed).random((64, 64)) < density
    start = _skeleton_chains(skeletonize(fg))
    fast = _contract_short(*copy_graph(start))
    plain = contract_short_by_definition(*copy_graph(start))
    same_graph(fast, plain)
    same_graph(_join_degree2(*copy_graph(fast)), join_degree2_by_definition(*copy_graph(plain)))


def test_remove_small_matches_scikit_image():
    from skimage.morphology import remove_small_objects

    fg = np.random.default_rng(0).random((64, 64)) < 0.5
    try:
        expected = remove_small_objects(fg, max_size=4)
    except TypeError:  # scikit-image < 0.26
        expected = remove_small_objects(fg, min_size=5)
    assert np.array_equal(_remove_small(fg, 5), expected)
    assert np.array_equal(_remove_small(fg, 0), fg)
    assert np.array_equal(_remove_small(np.zeros((4, 4), bool), 5), np.zeros((4, 4), bool))


def test_graph_from_mask_drops_small_blobs():
    m = np.zeros((64, 64), np.float32)
    m[10, 10:14] = 1  # 4 px
    m[30:34, 5:60] = 1
    g = graph_from_mask(m, min_length=5)
    assert len(g.xy) and g.xy[:, 1].min() > 20


def noise(rng, shape):
    return rng.random(shape, np.float32)


def test_road_decode_of_noise_is_fast():
    rng, n = np.random.default_rng(0), 384  # > 1 min when the steps were quadratic in the skeleton
    t = time.perf_counter()
    graph_from_road_heads(noise(rng, (n, n)), noise(rng, (3, n, n)), noise(rng, (4, n, n)) * 2 - 1)
    assert time.perf_counter() - t < 45


def test_junctions_per_zone_are_capped(monkeypatch):
    import scipy.sparse.csgraph as csgraph
    from manyseg.vectors.roads import _MAX_ZONE_JUNCTIONS

    sizes, real = [], csgraph.minimum_spanning_tree
    monkeypatch.setattr(csgraph, "minimum_spanning_tree",
                        lambda d: (sizes.append(d.shape[0]), real(d))[1])
    n = 512
    m = np.zeros((n, n), np.float32)
    m[:400, :400] = 1  # surface for the keypoints, and a road leaving their zone
    m[200:206, 400:] = 1
    kp = np.zeros((3, n, n), np.float32)
    kp[0, 6:400:12, 6:400:12] = 0.9  # the zones chain into one holding 1,089 junctions
    graph_from_road_heads(m, kp, np.zeros((4, n, n), np.float32))
    assert sizes and max(sizes) == _MAX_ZONE_JUNCTIONS


def test_each_instance_reads_only_nearby_corner_peaks(monkeypatch):
    rng, n = np.random.default_rng(0), 256
    lab = np.zeros((n, n), np.int32)
    k = 0
    for y in range(2, n - 8, 10):
        for x in range(2, n - 8, 10):
            k += 1
            lab[y:y + 7, x:x + 7] = k
    corner, direction = noise(rng, (n, n)), noise(rng, (2, n, n)) * 2 - 1
    seen, real = [], corners_module._polygon
    monkeypatch.setattr(corners_module, "_polygon",
                        lambda ring, peaks, *a, **kw: (seen.append(len(peaks)),
                                                       real(ring, peaks, *a, **kw))[1])
    near = polygons_from_corners(lab, corner, direction)
    assert len(seen) == k and max(seen) < 40

    # the same polygons as giving every instance every peak
    monkeypatch.setattr(np, "searchsorted", lambda a, v: (0, len(a)))
    monkeypatch.setattr(corners_module, "_polygon", real)
    every = polygons_from_corners(lab, corner, direction)
    assert all(a.equals_exact(b, 0) for a, b in zip(near, every)) and len(near) == len(every)


def courtyard(hole=20):
    lab = np.zeros((96, 96), np.int32)
    lab[20:80, 20:80] = 1
    c = 50 - hole // 2
    lab[c:c + hole, c:c + hole] = 0
    corner, direction = np.zeros((96, 96), np.float32), np.zeros((2, 96, 96), np.float32)
    for y, x in ((20, 20), (20, 80), (80, 20), (80, 80)):
        corner[y - 1, x - 1] = 1
    return lab, corner, direction


@pytest.mark.parametrize("corners", [True, False])  # straightened, and the outline fallback
def test_corner_decoding_keeps_a_courtyard(corners):
    lab, corner, direction = courtyard()
    if not corners:
        corner[:] = 0
    (p,) = polygons_from_corners(lab, corner, direction)
    assert len(p.interiors) == 1 and p.area == pytest.approx(60**2 - 20**2, abs=30)
    assert shapely.Polygon(p.interiors[0]).equals(shapely.box(40, 40, 60, 60))
    (filled,) = polygons_from_corners(lab, corner, direction, holes=False)
    assert not filled.interiors and filled.area == pytest.approx(60**2, abs=30)


def test_a_small_hole_is_filled():
    lab, corner, direction = courtyard(hole=3)
    (p,) = polygons_from_corners(lab, corner, direction)
    assert not p.interiors


def test_block_decoders_holes():
    from manyseg.vectors.blocks import InstanceBlockDecoder, RoofBlockDecoder

    assert block_decoder("footprints").holes and block_decoder("fields").holes
    assert InstanceBlockDecoder().holes and not RoofBlockDecoder().holes
    lab, corner, direction = courtyard()
    fg = (lab > 0).astype(np.float32)
    from scipy import ndimage

    fields = {"extent": fg, "boundary": np.zeros_like(fg), "corner": corner,
              "distance": (ndimage.distance_transform_edt(fg) / 15).astype(np.float32).clip(0, 1),
              "direction": direction}
    d = block_decoder("footprints")
    (r,) = decode_scene(fields, d, block=48, halo=d.min_halo)
    assert len(r["geometry"].interiors) == 1
