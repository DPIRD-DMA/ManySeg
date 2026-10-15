"""Image-level tasks: classification (single / multi-label) and regression, the loader with
image-level labels, the timm provider and an end-to-end run."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader
from manyseg_bench.providers import ModelSpec, get_provider
from manyseg_bench.tasks import build_task


def loader(labels, train, n=6, hw=(16, 16), **kw):
    imgs = np.random.default_rng(0).normal(size=(n, *hw, 3)).astype(np.float16)
    return GPULoader(imgs, labels, 4, [0.0] * 3, [1.0] * 3, torch.device("cpu"), train=train,
                     drop_last=False, **kw)


# --- loader ---------------------------------------------------------------------------------

@pytest.mark.parametrize("train", [False, True])
def test_loader_class_labels_are_untouched_by_augmentation(train):
    y = np.arange(6, dtype=np.int64)
    got = [(x, t) for x, t in loader(y, train, out_size=(12, 12) if train else None)]
    xs, ys = torch.cat([x for x, _ in got]), torch.cat([t for _, t in got])
    assert ys.dtype == torch.long and sorted(ys.tolist()) == list(range(6))
    assert xs.shape[-2:] == ((12, 12) if train else (16, 16))


def test_loader_keeps_float_and_multi_hot_labels():
    reg = np.linspace(0, 1, 12, dtype=np.float32).reshape(6, 2)
    ((_, y), *_) = loader(reg, train=False)
    assert y.dtype == torch.float32 and torch.allclose(y, torch.from_numpy(reg[:4]))
    hot = (np.arange(18).reshape(6, 3) % 2).astype(np.uint8)
    ((_, y), *_) = loader(hot, train=True)
    assert y.shape == (4, 3) and y.dtype == torch.long


def test_loader_out_size_comes_from_images():
    assert loader(np.zeros(6, np.int64), train=False).out_size == (16, 16)


# --- tasks ----------------------------------------------------------------------------------

def test_classification_loss_and_metrics():
    t = build_task("classification", 3)
    assert t.image_level and t.primary_metric == "accuracy" and t.higher_is_better
    logits = torch.tensor([[5.0, 0, 0], [0, 5, 0], [0, 5, 0], [0, 0, 5]])
    y = torch.tensor([0, 1, 2, 2])
    assert torch.isclose(t.loss(logits, y), torch.nn.functional.cross_entropy(logits, y))
    ev = t.evaluator()
    ev.update({"logits": logits[:2]}, y[:2])  # dict outputs (ManySeg named heads) work too
    ev.update(logits[2:], y[2:])
    m = ev.compute()
    assert m["accuracy"] == pytest.approx(0.75)
    # per-class F1: c0 1, c1 2/3 (p 1/2, r 1), c2 2/3 (p 1, r 1/2)
    assert m["f1"] == pytest.approx([1.0, 2 / 3, 2 / 3])
    assert m["macro_f1"] == pytest.approx((1 + 4 / 3) / 3)
    assert m["balanced_accuracy"] == pytest.approx((1 + 1 + 0.5) / 3)


def _ap(scores, labels):
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / hits.sum())


def test_multilabel_metrics_match_a_reference():
    t = build_task("multilabel", 3)
    assert t.primary_metric == "map_macro"
    rng = np.random.default_rng(0)
    y = (rng.random((40, 3)) < 0.4).astype(np.int64)
    logits = rng.normal(size=(40, 3)) + 2 * (y - 0.5)
    ev = t.evaluator()
    for i in range(0, 40, 16):
        ev.update(torch.tensor(logits[i:i + 16], dtype=torch.float32), torch.tensor(y[i:i + 16]))
    m = ev.compute()
    ap = [_ap(logits[:, k], y[:, k]) for k in range(3)]
    assert m["ap"] == pytest.approx(ap, abs=1e-6)
    assert m["map_macro"] == pytest.approx(np.mean(ap), abs=1e-6)
    assert m["map_micro"] == pytest.approx(_ap(logits.ravel(), y.ravel()), abs=1e-6)
    pred = logits > 0
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
    assert m["f1_micro"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits[:16], dtype=torch.float32), torch.tensor(y[:16], dtype=torch.float32))
    assert torch.isclose(t.loss(torch.tensor(logits[:16], dtype=torch.float32),
                                torch.tensor(y[:16])), loss)


def test_multilabel_class_without_positives_is_excluded_from_macro_ap():
    ev = build_task("multilabel", 2).evaluator()
    ev.update(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), torch.tensor([[1, 0], [0, 0]]))
    m = ev.compute()
    assert m["map_macro"] == 1.0 and math.isnan(m["ap"][1]) and m["classes_without_positives"] == 1


def test_regression_metrics():
    t = build_task("regression", 2)
    assert t.primary_metric == "rmse" and not t.higher_is_better
    y = torch.tensor([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
    p = y + torch.tensor([[0.5, 0.0], [-0.5, 0.0], [0.5, 1.0]])
    ev = t.evaluator()
    ev.update(p, y)
    m = ev.compute()
    err = (p - y).numpy()
    assert m["mae"] == pytest.approx(np.abs(err).mean())
    assert m["rmse"] == pytest.approx(np.sqrt((err ** 2).mean()))
    r2 = [1 - (err[:, k] ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum().item()
          for k in range(2)]
    assert m["r2"] == pytest.approx(r2) and m["r2_mean"] == pytest.approx(np.mean(r2))
    assert torch.isclose(t.loss(p, y), torch.nn.functional.mse_loss(p, y))
    assert build_task("regression", 1).loss(p[:, :1], y[:, :1]).ndim == 0


def test_unknown_task():
    with pytest.raises(ValueError, match="classification"):
        build_task("ranking", 2)


# --- providers ------------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["manyseg", "timm"])
def test_image_level_providers(provider):
    spec = ModelSpec("none", "resnet18", weights=None, in_channels=13, num_classes=7,
                     output="image")
    built = get_provider(provider).build(spec)
    y = built.model.eval()(torch.randn(2, 13, 64, 64))
    assert isinstance(y, torch.Tensor) and y.shape == (2, 7)
    head = {id(p) for p in built.decoder_params}
    n_head = sum(p.numel() for p in built.model.parameters() if id(p) in head)
    assert n_head == 512 * 7 + 7  # only the linear classifier is "decoder" (fastai freezing)


def test_manyseg_and_timm_classifiers_have_equal_parameter_counts():
    spec = ModelSpec("none", "resnet18", weights=None, in_channels=13, num_classes=7,
                     output="image")
    counts = [sum(p.numel() for p in get_provider(p).build(spec).model.parameters())
              for p in ("manyseg", "timm")]
    assert counts[0] == counts[1]


def test_timm_provider_is_image_level_only():
    with pytest.raises(ValueError, match="image"):
        get_provider("timm").build(ModelSpec("unet", "resnet18", weights=None))


def test_manyseg_image_level_with_band_groups():
    groups = {"a": {"bands": [0, 1, 2, 3]}, "b": {"bands": [4, 5], "factor": 2}}
    spec = ModelSpec("none", "resnet18", weights=None, in_channels=6, num_classes=4,
                     output="image", options={"band_groups": groups})
    model = get_provider("manyseg").build(spec).model.eval()
    assert model(torch.randn(2, 6, 64, 64)).shape == (2, 4)


# --- end to end -----------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["manyseg", "timm"])
def test_run_training_classification_end_to_end(tmp_path, provider, monkeypatch):
    from manyseg_bench import datasets
    from manyseg_bench.datasets.base import DatasetInfo, save_split
    from manyseg_bench.experiments import run_training

    class Fake:
        INFO = DatasetInfo(name="fake_cls", num_classes=3, in_channels=4,
                           class_names=("a", "b", "c"), mean=(0.0,) * 4, std=(1.0,) * 4)
        TASK = "classification"
        LOADER_KWARGS = {"vflip": True}

    monkeypatch.setitem(datasets.DATASETS, "fake_cls", Fake)
    cache = tmp_path / "cache" / "fake_cls"
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        y = rng.integers(0, 3, n)
        x = rng.normal(size=(n, 32, 32, 4)) + y[:, None, None, None]
        save_split(cache, split, x.astype(np.float16), y.astype(np.int64),
                   [f"{split}{i}" for i in range(n)])
    spec = ModelSpec("none", "resnet18", weights=None)
    rec = run_training(provider, spec, "fake_cls", "smoke", device="cpu", data=tmp_path,
                       out_dir=tmp_path / "runs", evaluate_test=True,
                       recipe_overrides={"batch_size": 4, "precision": "fp32"})
    assert rec["task"] == "classification" and rec["spec"]["output"] == "image"
    assert 0.0 <= rec["test"]["accuracy"] <= 1.0 and rec["img_size"] == [32, 32]
    assert json.loads(Path(rec["path"]).read_text())["primary_metric"] == "accuracy"
