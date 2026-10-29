"""Multi-head tasks through the provider-neutral trainer (regression test for dict outputs)."""

import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader
from manyseg_bench.providers import ModelSpec, get_provider
from manyseg_bench.tasks import InstanceTask, PointTask, SemanticTask, build_task
from manyseg_bench.training import get_recipe, train


def instance_data(n=8, h=48, w=48, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, h, w), np.int32)
    imgs = rng.integers(0, 40, (n, h, w, 3)).astype(np.uint8)
    for i in range(n):
        k = 1
        for y in range(2, h - 12, 15):
            for x in range(2, w - 12, 15):
                hh, ww = rng.integers(6, 12, 2)
                masks[i, y:y + hh, x:x + ww] = k
                imgs[i, y:y + hh, x:x + ww] += np.uint8(150)
                k += 1
    return imgs, masks


def loaders():
    imgs, masks = instance_data()
    mk = lambda train: GPULoader(imgs, masks, 4, (0.3,) * 3, (0.3,) * 3,  # noqa: E731
                                 torch.device("cpu"), train, scale=(0.8, 1.0))
    return mk(True), mk(False)


@pytest.mark.parametrize("task", [InstanceTask(), PointTask(offsets=True), PointTask(density=True)])
def test_task_trains_and_evaluates(task):
    torch.manual_seed(0)
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=1,
                     options={"heads": task.heads()})
    b = get_provider("manyseg").build(spec)
    rec = get_recipe("standard", epochs=4, batch_size=4, lr=1e-3, precision="fp32",
                     warmup_epochs=0)
    tr, va = loaders()
    res = train(b.model, b.encoder_params, b.decoder_params, tr, va, num_classes=1, recipe=rec,
                device=torch.device("cpu"), task=task, log=lambda *_: None)
    losses = [h["train_loss"] for h in res["history"]]
    assert losses[-1] < losses[0]
    assert task.primary_metric in res["final_val"]


def test_semantic_task_is_default_and_named_miou():
    t = SemanticTask(num_classes=3)
    assert t.primary_metric == "miou"


def test_build_task():
    assert isinstance(build_task("instances", num_classes=1), InstanceTask)
    with pytest.raises(ValueError):
        build_task("nope", num_classes=1)


def test_extent_only_instance_task_uses_connected_components():
    task = InstanceTask(mode="extent_cc")
    assert set(task.heads()) == {"extent"}
    torch.manual_seed(0)
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=1,
                     options={"heads": task.heads()})
    b = get_provider("manyseg").build(spec)
    rec = get_recipe("standard", epochs=2, batch_size=4, lr=1e-3, precision="fp32",
                     warmup_epochs=0)
    tr, va = loaders()
    res = train(b.model, b.encoder_params, b.decoder_params, tr, va, num_classes=1, recipe=rec,
                device=torch.device("cpu"), task=task, log=lambda *_: None)
    assert "pq" in res["final_val"]


def test_point_task_ignores_negative_label_regions():
    """Pixels with y < 0 (e.g. OAM-TCD canopy groups) carry no loss and drop predicted peaks."""
    task = PointTask()
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[0, 4:10, 4:10] = 1
    y[0, 16:30, 16:30] = -1
    t = task.targets(y)
    assert len(t["points"][0]) == 1
    out = {"heatmap": t["heatmap"].clone()}
    out["heatmap"][0, 0, 20:26, 20:26] = 5.0  # garbage inside the ignore region
    assert task.loss(out, t) == pytest.approx(0.0, abs=1e-6)
    ev = task.evaluator()
    ev.update(out, y)
    r = ev.compute()
    assert r["fp"] == 0 and r["tp"] == 1


def test_point_threshold_calibration_grid():
    """calibrate=True scores a grid of peak thresholds; low peaks are missed at 0.5 only."""
    task = PointTask(calibrate=True)
    y = torch.zeros(1, 48, 48, dtype=torch.long)
    y[0, 4:10, 4:10] = 1
    y[0, 30:36, 30:36] = 2
    t = task.targets(y)
    out = {"heatmap": t["heatmap"].clone() * 0.3}  # under-confident peaks (MSE-like)
    ev = task.evaluator()
    ev.update(out, y)
    r = ev.compute()
    assert r["tp"] == 0  # default threshold 0.5 misses both
    assert r["f1_by_threshold"][0.2] == 1.0 and r["f1_by_threshold"][0.5] == 0.0
    assert r["best_threshold"] <= 0.3


def test_runner_applies_val_calibrated_threshold_to_test(tmp_path):
    """The threshold is chosen on val only and then used for the test evaluation."""
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import run_training

    imgs, masks = instance_data(n=6, h=48, w=48)
    cache = tmp_path / "cache" / "oam_tcd"
    for split in ("train", "val", "test"):
        save_split(cache, split, imgs, masks.astype("int16"), [str(i) for i in range(6)])
    rec = run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), "oam_tcd",
                       "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                       evaluate_test=True,
                       recipe_overrides={"precision": "fp32", "batch_size": 2},
                       task_overrides={"calibrate": True})
    thr = rec["calibrated_threshold"]
    assert thr == rec["final_val"]["best_threshold"]
    assert rec["test"]["threshold"] == thr


def test_point_task_calibrates_by_default():
    """E011b: val-calibrated peak thresholds are the default; calibrate=False opts out."""
    assert PointTask().calibrate
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[0, 4:10, 4:10] = 1
    task = PointTask(calibrate=False)
    ev = task.evaluator()
    ev.update({"heatmap": task.targets(y)["heatmap"]}, y)
    assert "best_threshold" not in ev.compute()
    task = PointTask(density=True)
    ev = task.evaluator()
    ev.update({"density": task.targets(y)["density"]}, y)
    assert "best_threshold" not in ev.compute()


def test_point_task_density_activation():
    """E011c: density_activation reaches the head preset; counts report a signed bias."""
    assert "activation" not in PointTask(density=True, density_activation=None).heads()["density"]
    h = PointTask(density=True).heads()["density"]
    assert h["activation"] == "softplus" and h["bias_init"] == -4.0
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[0, 4:10, 4:10] = 1
    task = PointTask(density=True)
    ev = task.evaluator()
    ev.update({"density": task.targets(y)["density"] * task.density_scale * 3}, y)
    r = ev.compute()
    assert r["bias"] == pytest.approx(2.0, abs=1e-4) and r["mae"] == pytest.approx(2.0, abs=1e-4)
