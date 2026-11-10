"""Regression tests for the instance, point and vector fixes of the pre-release sweep."""

import importlib.util
import inspect
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from manyseg.instances import instance_heads, instance_losses, instance_targets
from manyseg.instances.postprocess import instances_from_outputs
from manyseg.losses import multiclass_dice
from manyseg.points import (
    calibrate_threshold,
    find_peaks,
    group_by_offsets,
    heatmap_from_instances,
)
from manyseg.points.postprocess import _match_points

# --- instance heads: multi-class extent ----------------------------------------------------------


def _semantic_batch():
    ids = torch.zeros(2, 32, 32, dtype=torch.long)
    ids[0, 4:14, 4:14], ids[0, 18:30, 6:20], ids[1, 10:25, 10:25] = 1, 2, 3
    sem = torch.zeros_like(ids)
    sem[ids == 1], sem[ids == 2], sem[ids == 3] = 1, 2, 2  # two foreground classes
    return ids, sem


def test_instance_losses_multiclass_extent():
    ids, sem = _semantic_batch()
    t = instance_targets(ids, semantic=sem)
    assert t["extent"].dtype == torch.long and int(t["extent"].max()) == 2
    torch.manual_seed(0)
    heads = instance_heads(num_classes=2)
    assert heads["extent"]["out_channels"] == 3
    out = {"extent": torch.randn(2, 3, 32, 32, requires_grad=True),
           "boundary": torch.randn(2, 1, 32, 32), "distance": torch.randn(2, 1, 32, 32)}
    parts = instance_losses(out, t)
    want = F.cross_entropy(out["extent"], sem) + multiclass_dice(out["extent"], sem)
    assert torch.allclose(parts["extent"], want)
    parts["extent"].backward()
    assert out["extent"].grad is not None
    # the distance loss covers every foreground class (extent > 0), not just class 1
    d_err = (out["distance"] - t["distance"]).abs()
    assert torch.allclose(parts["distance"], d_err[sem[:, None] > 0].mean())
    # (B, 1, H, W) class ids work as well
    t1 = {**t, "extent": sem[:, None]}
    assert torch.allclose(instance_losses(out, t1)["extent"], want)


def test_instance_losses_rejects_class_ids_on_a_binary_head():
    ids, sem = _semantic_batch()
    t = instance_targets(ids, semantic=sem)
    out = {"extent": torch.randn(2, 1, 32, 32), "boundary": torch.randn(2, 1, 32, 32),
           "distance": torch.randn(2, 1, 32, 32)}
    with pytest.raises(ValueError, match="num_classes"):
        instance_losses(out, t)
    with pytest.raises(ValueError, match="num_classes"):
        instance_losses(out, {**t, "extent": sem.float()})
    # binary targets still train a 1-channel head
    tb = instance_targets(ids)
    parts = instance_losses(out, tb)
    assert all(v.isfinite() for v in parts.values())
    # a multi-channel head needs class ids, not a one-hot / multi-channel target
    out3 = {**out, "extent": torch.randn(2, 3, 32, 32)}
    with pytest.raises(ValueError, match="class-id"):
        instance_losses(out3, {**t, "extent": torch.zeros(2, 3, 32, 32)})


def test_instances_from_multiclass_extent_use_argmax_background_zero():
    ids, sem = _semantic_batch()
    t = instance_targets(ids, semantic=sem, distance="normalized")
    scores = F.one_hot(sem, 3).permute(0, 3, 1, 2).float() * 10  # class 2 dominant where set
    boundary = t["boundary"] * 20 - 10
    got = instances_from_outputs(scores, boundary, t["distance"], logits=True, backend="torch")
    # foreground = argmax > 0: every class-1 and class-2 pixel, nothing else
    assert torch.equal(got > 0, sem > 0)
    assert [len(torch.unique(g)) - 1 for g in got] == [2, 1]
    # probabilities give the same result
    probs = scores.softmax(1)
    got_p = instances_from_outputs(probs, t["boundary"], t["distance"], logits=False,
                                   backend="torch")
    assert torch.equal(got_p, got)


def test_instance_heads_has_no_distance_parameter():
    assert "distance" not in inspect.signature(instance_heads).parameters
    with pytest.raises(TypeError):
        instance_heads(distance="absolute")


# --- points ---------------------------------------------------------------------------------------


def test_group_by_offsets_accepts_channel_masks_and_rejects_float():
    b, h, w = 2, 8, 10
    offsets = torch.zeros(b, 2, h, w)
    centers = [torch.tensor([[2.0, 2.0], [6.0, 8.0]]), torch.tensor([[4.0, 4.0]])]
    fg = torch.zeros(b, h, w, dtype=torch.bool)
    fg[0, :, :5], fg[1, 2:6, 2:6] = True, True
    ref = group_by_offsets(centers, offsets, fg)
    assert ref.shape == (b, h, w)
    assert torch.equal(ref > 0, fg)
    assert set(ref[1].unique().tolist()) == {0, 1}
    assert torch.equal(group_by_offsets(centers, offsets, fg[:, None]), ref)
    assert torch.equal(group_by_offsets(centers, offsets, fg[:, None].long()), ref)
    assert torch.equal(group_by_offsets(centers, offsets, fg.to(torch.uint8)), ref)
    with pytest.raises(ValueError, match="threshold"):
        group_by_offsets(centers, offsets, fg.float() * 0.9)
    with pytest.raises(ValueError):
        group_by_offsets(centers, offsets, fg[:1])  # one mask for the batch
    with pytest.raises(ValueError):
        group_by_offsets(centers, offsets, torch.stack([fg, fg], 1))  # (B, 2, H, W)


def test_find_peaks_input_shapes():
    torch.manual_seed(0)
    hm = torch.rand(2, 1, 16, 16)
    ref = find_peaks(hm, 0.5)
    got = find_peaks(hm[:, 0], 0.5)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(got, ref))
    # a single (H, W) map is one image: the same as its (1, H, W) batch
    one = find_peaks(hm[0, 0], 0.5)
    assert len(one) == 1
    assert torch.equal(one[0][0], ref[0][0]) and torch.equal(one[0][1], ref[0][1])
    mask = hm[0, 0] > 0.2
    m2 = find_peaks(hm[0, 0], 0.5, mask=mask)
    m3 = find_peaks(hm[:1], 0.5, mask=mask[None])
    assert torch.equal(m2[0][0], m3[0][0])
    # consistent with calibrate_threshold, which treats a 2-D map as one image
    pts = ref[0][0]
    best, curve = calibrate_threshold(hm[0, 0], [pts], thresholds=[0.5], metric="f1")
    assert curve[0.5] == 1.0
    with pytest.raises(ValueError, match="channels"):
        find_peaks(torch.rand(2, 3, 16, 16), 0.5)
    with pytest.raises(ValueError):
        find_peaks(torch.rand(16), 0.5)


def test_match_points_long_chain_does_not_recurse():
    n = 1500
    pred = torch.stack([torch.zeros(n), torch.arange(n, dtype=torch.float32)], 1)
    gt = torch.stack([torch.zeros(n), torch.arange(n, dtype=torch.float32) + 0.5], 1)
    limit = sys.getrecursionlimit()
    assert _match_points(pred, gt, 0.6) == n
    assert sys.getrecursionlimit() == limit
    # through calibrate_threshold: one image holding the chain as peaks
    hm = torch.zeros(1, 1, 3, 2 * n + 2)
    hm[0, 0, 1, 1:2 * n + 1:2] = 0.9  # peaks at x = 1, 3, 5, ...
    peaks = find_peaks(hm, 0.5)[0][0]
    gt_pts = peaks.float() + torch.tensor([0.0, 0.5])
    _, curve = calibrate_threshold(hm, [gt_pts], thresholds=[0.5], metric="f1", radius=0.6)
    assert curve[0.5] == 1.0


def _brute_max_matching(near):
    """Maximum matching size by exhaustive search (tiny inputs)."""
    from itertools import permutations
    n, m = near.shape
    best = 0
    for perm in permutations(range(max(n, m)), n):
        best = max(best, sum(1 for i, j in enumerate(perm) if j < m and near[i, j]))
    return best


def test_match_points_is_a_maximum_matching():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n, m = rng.integers(1, 6, 2)
        pred = torch.tensor(rng.uniform(0, 4, (n, 2)))
        gt = torch.tensor(rng.uniform(0, 4, (m, 2)))
        near = (torch.cdist(pred, gt) <= 1.5).numpy()
        assert _match_points(pred, gt, 1.5) == _brute_max_matching(near)


def test_heatmap_from_instances_needs_a_sigma():
    m = torch.zeros(1, 8, 8, dtype=torch.long)
    m[0, 2:5, 2:5] = 1
    with pytest.raises(ValueError, match="sigma"):
        heatmap_from_instances(m, sigma=None)
    assert heatmap_from_instances(m, sigma=None, sigma_scale=0.5).max() > 0.9


# --- vectors: Gaussian windows commute with the dihedral transforms -------------------------------

needs_vectors = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("shapely", "networkx", "rasterio")),
    reason="needs the vectors extras (shapely, networkx, rasterio)")

@needs_vectors
def test_gaussian_window_symmetric_and_peaks_unchanged():
    from manyseg.vectors.road_targets import _gaussians as g
    # integer point (a pixel corner): the window is symmetric about it
    a = g(np.array([[10.0, 12.0]]), 2.0, (30, 30))
    assert np.allclose(a[:, 3:17], a[:, 3:17][:, ::-1])  # columns 3..16 around x = 10
    assert np.allclose(a[5:19], a[5:19][::-1])  # rows 5..18 around y = 12
    assert a[12, 3] > 0 and a[12, 16] > 0 and a[12, 2] == 0 and a[12, 17] == 0
    # a point at a pixel centre: the peak is exactly 1, the window 13 px wide
    b = g(np.array([[10.5, 12.5]]), 2.0, (30, 30))
    assert b[12, 10] == 1.0 and (b[12] > 0).sum() == 13
    # generic float points: the same pixels and values as the 2 sigma window before the fix
    x, y = 10.3, 12.8
    c = g(np.array([[x, y]]), 2.0, (30, 30))
    cs, rs = np.arange(10 - 6, 10 + 7), np.arange(12 - 6, 12 + 7)
    old = np.zeros((30, 30), np.float32)
    old[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1] = np.exp(
        -((cs + 0.5 - x)[None] ** 2 + (rs + 0.5 - y)[:, None] ** 2) / 8).astype(np.float32)
    assert np.array_equal(c, old)


@needs_vectors
def test_corner_targets_exactly_equivariant_at_integer_corners():
    import shapely
    from manyseg.vectors import corner_targets
    from manyseg.vectors.augment import dihedral, dihedral_geometry

    size = 48
    polys = [shapely.box(8, 10, 22, 30), shapely.Polygon([(28, 6), (42, 6), (42, 20), (34, 20),
                                                        (34, 14), (28, 14)])]
    t0 = corner_targets(polys, (size, size))
    for d in range(8):
        t1 = corner_targets([dihedral_geometry(p, d, (size, size)) for p in polys], (size, size))
        assert np.allclose(dihedral(t0["corner"], d), t1["corner"], atol=1e-6, rtol=0), d


@needs_vectors
def test_road_keypoints_exactly_equivariant_at_integer_nodes():
    import networkx as nx
    from manyseg.vectors import road_targets
    from manyseg.vectors.augment import dihedral, dihedral_xy

    size = 64
    g = nx.Graph()
    for n, xy in {"j": (30, 30), "a": (10, 30), "b": (30, 8), "c": (50, 44), "e": (30, 52)}.items():
        g.add_node(n, xy=np.array(xy, float))
    g.add_edges_from([("j", "a"), ("j", "b"), ("j", "c"), ("j", "e")], width=6.0)
    cr = np.array([[20.0, 40.0]])
    t0 = road_targets(g, (size, size), crossings=cr)
    for d in range(8):
        h = nx.Graph()
        for n, a in g.nodes(data=True):
            h.add_node(n, xy=dihedral_xy(a["xy"], d, (size, size)))
        h.add_edges_from(g.edges(data=True))
        t1 = road_targets(h, (size, size), crossings=dihedral_xy(cr, d, (size, size)))
        assert np.allclose(dihedral(t0["keypoints"], d), t1["keypoints"], atol=1e-6, rtol=0), d


@needs_vectors
def test_road_direction_mode2_is_set_at_junctions_too():
    """Documented rule: mode 2 wherever two degree-2 chains overlap at an angle, so a T-junction
    with no crossing gets it near the node."""
    import shapely
    from manyseg.vectors import road_graph, road_targets

    graph, crossings = road_graph([shapely.LineString([(0, 32), (64, 32)]),
                                   shapely.LineString([(32, 32), (32, 63)])])
    assert len(crossings) == 0
    t = road_targets(graph, (64, 64), crossings=crossings)
    second = np.abs(t["direction"][2:]).sum(0) > 0
    assert second.sum() > 0
    rows, cols = np.nonzero(second)
    assert np.hypot(rows + 0.5 - 32, cols + 0.5 - 32).max() < 10  # around the junction only
