"""Point (centre heatmap + offsets) and density outputs: targets, post-processing, presets."""

import pytest
import torch
from manyseg import create_model
from manyseg.points import (
    centers_from_instances,
    count_from_density,
    density_head,
    density_map,
    find_peaks,
    gaussian_heatmap,
    group_by_offsets,
    heatmap_from_instances,
    offset_targets,
    panoptic_labels,
    point_heads,
)

PTS = torch.tensor([[5.0, 7.0], [20.0, 30.0], [33.0, 9.0], [12.0, 40.0]])


def blocks():
    m = torch.zeros(40, 48, dtype=torch.long)
    m[2:9, 3:12] = 1
    m[2:9, 12:20] = 2      # touches 1
    m[20:35, 25:40] = 3
    m[30:38, 2:8] = 4
    return m


def same_partition(a, b):
    if not torch.equal(a > 0, b > 0):
        return False
    pairs = torch.unique(torch.stack([a[a > 0], b[b > 0]]), dim=1)
    return pairs.shape[1] == len(torch.unique(a[a > 0])) == len(torch.unique(b[b > 0]))


# --- targets -----------------------------------------------------------------------------

def test_heatmap_peaks_at_points():
    hm = gaussian_heatmap([PTS], (40, 48), sigma=2.0)
    assert hm.shape == (1, 1, 40, 48)
    for y, x in PTS.long().tolist():
        assert hm[0, 0, y, x] == 1.0
    assert hm.max() == 1.0 and hm.min() >= 0


def test_heatmap_sigma_per_point():
    hm = gaussian_heatmap([PTS[:2]], (40, 48), sigma=torch.tensor([1.0, 4.0]))
    assert hm[0, 0, 5, 9] < hm[0, 0, 20, 32]  # wider blob around the second point


def test_heatmap_empty_image():
    assert gaussian_heatmap([torch.zeros(0, 2)], (8, 8), sigma=2).sum() == 0


def test_density_mass_preserved_including_border_points():
    pts = torch.cat([PTS, torch.tensor([[0.0, 0.0], [39.0, 47.0]])])
    d = density_map([pts], (40, 48), sigma=3.0)
    torch.testing.assert_close(d.sum(), torch.tensor(float(len(pts))), atol=1e-4, rtol=0)
    torch.testing.assert_close(count_from_density(d), torch.tensor([float(len(pts))]),
                               atol=1e-4, rtol=0)


def test_centers_from_instances():
    centers, ids = centers_from_instances(blocks())[0]
    assert ids.tolist() == [1, 2, 3, 4]
    torch.testing.assert_close(centers[0], torch.tensor([5.0, 7.0]))   # rows 2..8, cols 3..11


def test_offsets_point_to_centroid():
    m = blocks()
    off = offset_targets(m)
    assert off.shape == (1, 2, 40, 48)
    y, x = 4, 15  # inside instance 2 (centroid (5, 15.5))
    torch.testing.assert_close(off[0, :, y, x], torch.tensor([1.0, 0.5]))
    assert (off[0, :, m == 0] == 0).all()


def test_heatmap_from_instances_size_scaled_sigma():
    hm = heatmap_from_instances(blocks(), sigma_scale=0.1)
    assert hm.shape == (1, 1, 40, 48) and hm.max() <= 1


# --- post-processing round trips ---------------------------------------------------------

def test_find_peaks_round_trip_integer_points():
    hm = gaussian_heatmap([PTS, PTS[:2]], (40, 48), sigma=2.0)
    res = find_peaks(hm, threshold=0.5)
    assert len(res) == 2
    got = {tuple(p) for p in res[0][0].tolist()}
    assert got == {tuple(p) for p in PTS.long().tolist()}
    assert torch.allclose(res[0][1], torch.ones(4))
    assert len(res[1][0]) == 2


def test_find_peaks_max_points_keeps_highest():
    hm = gaussian_heatmap([PTS], (40, 48), sigma=2.0) * torch.linspace(0.6, 1, 48)
    pts, scores = find_peaks(hm, threshold=0.1, max_points=2)[0]
    assert len(pts) == 2 and scores[0] >= scores[1]


def test_group_by_offsets_round_trip():
    m = blocks()
    centers, _ = centers_from_instances(m)[0]
    labels = group_by_offsets([centers], offset_targets(m), m[None] > 0)
    assert same_partition(labels[0], m)


def test_full_chain_heatmap_peaks_offsets_to_instances():
    """instance mask -> (heatmap, offsets) -> peaks -> grouping recovers the instances."""
    m = blocks()
    hm = heatmap_from_instances(m, sigma=2.0)
    peaks = [p.float() for p, _ in find_peaks(hm, threshold=0.5)]
    labels = group_by_offsets(peaks, offset_targets(m), m[None] > 0)
    assert same_partition(labels[0], m)


def test_panoptic_labels():
    sem = torch.tensor([[[0, 1, 1, 2]]])
    inst = torch.tensor([[[0, 1, 2, 0]]])
    pan = panoptic_labels(sem, inst, thing_classes=(1,), label_divisor=1000)
    assert pan.tolist() == [[[0, 1001, 1002, 2000]]]


# --- presets ------------------------------------------------------------------------------

@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("decoders", [None, "separate"])
def test_point_preset_shapes(offsets, decoders):
    heads = point_heads(offsets=offsets, decoders=decoders)
    model = create_model(backbone="resnet18", pretrained=False, heads=heads)
    out = model(torch.randn(2, 3, 65, 97))
    assert out["heatmap"].shape == (2, 1, 65, 97)
    if offsets:
        assert out["offset"].shape == (2, 2, 65, 97)
    sum(v.mean() for v in out.values()).backward()


def test_density_preset_and_count():
    model = create_model(backbone="resnet18", pretrained=False, heads=density_head()).eval()
    d = model(torch.randn(1, 3, 64, 64))["density"]
    assert d.shape == (1, 1, 64, 64) and count_from_density(d).shape == (1,)


def test_heatmap_preset_kind():
    assert point_heads(heatmap="segmentation")["heatmap"]["type"] == "segmentation"
    with pytest.raises(ValueError):
        point_heads(heatmap="nope")


# --- regressions from Codex review 2026-10-14 ----------------------------------------------

def test_centers_many_instances_without_k_h_w_tensors():
    m = torch.zeros(1, 128, 128, dtype=torch.long)
    m[0] = torch.arange(128 * 128).view(128, 128) // 4 + 1   # 4096 tiny instances
    centers, ids = centers_from_instances(m)[0]
    assert len(ids) == 4096
    torch.testing.assert_close(centers[0], torch.tensor([0.0, 1.5]))


def test_find_peaks_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        find_peaks(torch.zeros(1, 1, 8, 8), kernel=4)


def test_focal_heatmap_preset_starts_at_centernet_prior():
    """heatmap='segmentation' (focal loss) initialises the head bias to logit(0.1), as in
    CenterNet; a zero bias (p=0.5 everywhere) makes the focal loss explode at the start."""
    import math

    model = create_model(backbone="resnet18", pretrained=False,
                         heads=point_heads(heatmap="segmentation")).eval()
    torch.testing.assert_close(model.heads["heatmap"].conv.bias,
                               torch.full((1,), math.log(0.1 / 0.9)))
    assert point_heads()["heatmap"].get("bias_init") is None  # MSE regression keeps zero bias
