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
