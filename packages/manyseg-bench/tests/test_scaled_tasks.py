"""Instance and point tasks with scaled heads: losses and counts on the head's own grid.

Regressions (Codex review of feature/reduced-resolution): density was upsampled bilinearly and
then summed (counts inflated by ~f^2), losses compared upsampled predictions with full-resolution
targets, and peaks were picked on an interpolated heatmap.
"""

import pytest
import torch
from manyseg.utils import downsample_sum, scaled_size, to_input_coords
from manyseg_bench.tasks import InstanceTask, PointTask

H, W = 33, 47  # ragged at every scale


def instances():
    y = torch.zeros(1, H, W, dtype=torch.long)
    y[0, 2:6, 3:7] = 1
    y[0, 10:13, 20:28] = 2
    y[0, 28:33, 40:47] = 3      # touches the ragged bottom-right corner
    y[0, 20:24, 1:5] = 4
    y[0, 15:18, 33:36] = 5
    return y


def native_density(task, y, f):
    t = task.targets(y)
    return {"density": downsample_sum(t["density"] * task.density_scale, 1 / f)}


@pytest.mark.parametrize("f", [2, 4])
def test_density_count_on_native_grid(f):
    task = PointTask(density=True)
    y = instances()
    out = native_density(task, y, f)
    assert out["density"].shape[-2:] == scaled_size(H, W, 1 / f)
    assert out["density"].sum() / task.density_scale == pytest.approx(5.0, abs=1e-4)
    ev = task.evaluator()
    ev.update(out, y)
    r = ev.compute()
    assert r["mae"] == pytest.approx(0.0, abs=1e-4)
    assert task.loss(out, task.targets(y)) == pytest.approx(0.0, abs=1e-8)


def test_density_in_a_fully_ignored_region_does_not_count():
    task = PointTask(density=True)
    y = torch.zeros(1, H, W, dtype=torch.long)
    y[0, 8:24, 8:24] = -1
    y[0, 2:4, 30:32] = 1
    out = {"density": torch.zeros(1, 1, *scaled_size(H, W, 0.25))}
    out["density"][0, 0, 3, 3] = 7.0 * task.density_scale   # input pixels 12..15: ignored
    ev = task.evaluator()
    ev.update(out, y)
    assert ev.cm.compute()["mae"] == pytest.approx(1.0)       # predicted 0, true 1


@pytest.mark.parametrize("kind", ["regression", "segmentation"])
@pytest.mark.parametrize("offsets", [False, True])
def test_point_targets_on_native_grid_give_zero_regression_loss(kind, offsets):
    task = PointTask(offsets=offsets, heatmap=kind)
    y = instances()
    t = task.targets(y)
    nt = task.head_targets({"heatmap": torch.zeros(1, 1, 17, 24)}, t)
    assert nt["heatmap"].shape[-2:] == (17, 24)
    # max-pool keeps every centre's peak of exactly 1 on the coarse grid
    assert (nt["heatmap"] >= 0.999).sum() == 5
    if offsets:
        assert nt["offset"].shape == (1, 2, 17, 24)
    if kind == "regression":
        out = {"heatmap": nt["heatmap"]}
        if offsets:
            out["offset"] = nt["offset"]
        assert task.loss(out, t) == pytest.approx(0.0, abs=1e-8)


def test_native_offsets_are_head_pixel_offsets_to_the_centroid():
    """A 4x4 instance at f=2: every coarse pixel points at the centroid in head pixels."""
    task = PointTask(offsets=True)
    y = torch.zeros(1, 16, 16, dtype=torch.long)
    y[0, 4:8, 8:12] = 1                       # centroid (5.5, 9.5) -> head (2.5, 4.5)
    nt = task.head_targets({"heatmap": torch.zeros(1, 1, 8, 8)}, task.targets(y))
    torch.testing.assert_close(nt["offset"][0, :, 2:4, 4:6],
                               torch.tensor([[[0.5, 0.5], [-0.5, -0.5]],
                                             [[0.5, -0.5], [0.5, -0.5]]]))


@pytest.mark.parametrize("f", [2, 4])
def test_isolated_coarse_peak_is_one_detection_at_its_input_location(f):
    task = PointTask()
    y = torch.zeros(1, H, W, dtype=torch.long)
    i, j = 3, 4
    y[0, i * f:(i + 1) * f, j * f:(j + 1) * f] = 1          # the area of coarse pixel (i, j)
    hm = torch.zeros(1, 1, *scaled_size(H, W, 1 / f))
    hm[0, 0, i, j] = 1.0
    (pts, _), = task.find_points({"heatmap": hm}, task.targets(y))
    torch.testing.assert_close(pts, to_input_coords(torch.tensor([[i, j]]), 1 / f))
    ev = task.evaluator()
    ev.update({"heatmap": hm}, y)
    r = ev.compute()
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


def test_instance_targets_on_native_grid():
    """extent: majority; boundary: any (thin boundaries survive); distance: mean over the
    block's instance pixels."""
    task = InstanceTask(boundary_width=1.0)
    y = instances()
    t = task.targets(y)
    nt = task.head_targets({"extent": torch.zeros(1, 1, 9, 12)}, t)
    assert {k: tuple(v.shape) for k, v in nt.items() if k != "instances"} == {
        "extent": (1, 1, 9, 12), "boundary": (1, 1, 9, 12), "distance": (1, 1, 9, 12),
        "distance_mask": (1, 1, 9, 12)}
    # every instance keeps some boundary at 1/4 even though its band is 1 px wide
    from manyseg.utils import downsample_max

    for k in range(1, 6):
        touched = downsample_max(y[0] == k, 0.25)
        assert (nt["boundary"][0, 0][touched] > 0).any(), k
    assert nt["distance"].max() <= 1.0 and (nt["distance"][~nt["distance_mask"]] == 0).all()


def test_instance_loss_is_computed_on_the_native_grid():
    task = InstanceTask()
    y = instances()
    t = task.targets(y)
    out = {k: torch.randn(1, 1, 17, 24) for k in ("extent", "boundary", "distance")}
    nt = task.head_targets(out, t)
    from manyseg_bench import losses

    ref = (losses.dice_bce(out["extent"], nt["extent"])
           + losses.dice_bce(out["boundary"], nt["boundary"])
           + losses.masked_regression(out["distance"], nt["distance"], nt["distance_mask"]))
    torch.testing.assert_close(task.loss(out, t), ref)
