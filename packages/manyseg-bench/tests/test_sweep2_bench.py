"""Regression tests for the second library sweep (synthetic tensors, no downloads)."""

from __future__ import annotations

import builtins
import json
import math

import numpy as np
import pytest
import torch
from manyseg_bench import tasks
from manyseg_bench.evaluation import objects
from manyseg_bench.tasks import build_task

# --- B1: seed_threshold reaches the instance decoding -----------------------------------------


def _fields():
    y = torch.zeros(1, 40, 60, dtype=torch.long)
    y[0, 5:35, 5:30] = 1
    y[0, 5:35, 30:55] = 2  # two touching fields
    return y


def test_instance_seed_threshold_defaults_and_pass_through(monkeypatch):
    pytest.importorskip("manyseg")
    assert tasks.InstanceTask().seed_threshold == 0.25
    assert tasks.InstanceTask(distance="absolute").seed_threshold == 3.0
    assert tasks.InstanceTask(distance="absolute", seed_threshold=5).seed_threshold == 5.0
    assert tasks.CornerLineTask(distance="absolute").seed_threshold == 3.0
    assert tasks.CornerLineTask(seed_threshold=0.4).inner.seed_threshold == 0.4
    with pytest.raises(ValueError, match="distance"):
        tasks.InstanceTask(distance="relative")

    import manyseg.instances as mi

    seen = []
    orig = mi.instances_from_outputs

    def spy(*a, **kw):
        seen.append(kw.get("seed_threshold"))
        return orig(*a, **kw)

    monkeypatch.setattr(mi, "instances_from_outputs", spy)
    y = _fields()
    task = tasks.InstanceTask(distance="absolute")
    t = task.targets(y)
    out = {"extent": (t["extent"].float() * 2 - 1) * 20, "boundary": (t["boundary"] * 2 - 1) * 20,
           "distance": t["distance"].float()}
    ev = task.evaluator()
    ev.update(out, y)
    assert seen == [3.0]
    assert ev.compute()["tp"] == 2  # the perfect output decodes to both fields


# --- B2: R2 of a constant target is NaN, r2_mean a NaN-mean ------------------------------------

def test_r2_constant_target_is_nan():
    task = build_task("regression", 2)
    ev = task.evaluator()
    y = torch.stack([torch.full((6,), 3.0), torch.arange(6.0)], 1)
    ev.update(y + torch.tensor([0.5, 0.0]), y)  # output 0 constant, output 1 perfect
    r = ev.compute()
    assert math.isnan(r["r2"][0]) and r["r2"][1] == pytest.approx(1.0)
    assert r["r2_mean"] == pytest.approx(1.0)
    ev = task.evaluator()
    ev.update(torch.zeros(4, 2), torch.ones(4, 2))
    assert math.isnan(ev.compute()["r2_mean"])


# --- B3: instance matching below IoU 0.5 is exact and scipy-free ------------------------------

def test_low_iou_matching_does_not_depend_on_scipy(monkeypatch):
    iou = torch.tensor([[0.45, 0.35], [0.35, 0.0]], dtype=torch.double)
    with_scipy = objects._one_to_one(iou, 0.3)
    real = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    without = objects._one_to_one(iou, 0.3)
    assert torch.equal(with_scipy, without)
    assert int(without.sum()) == 2  # 0.35 + 0.35 > 0.45 (greedy took 0.45 alone)


def _best_sum(w, i, used) -> float:
    """Brute force: the largest weight sum of rows i.. assigned to distinct unused columns."""
    if i == w.shape[0]:
        return 0.0
    best = _best_sum(w, i + 1, used)
    for j in range(w.shape[1]):
        if j not in used and w[i, j] > 0:
            best = max(best, w[i, j] + _best_sum(w, i + 1, used | {j}))
    return best


def test_low_iou_matching_is_max_iou_sum():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g, p = rng.integers(1, 7, 2)
        iou = torch.tensor(rng.random((g, p)) * 0.7)
        m = objects._one_to_one(iou, 0.2)
        assert (m.sum(0) <= 1).all() and (m.sum(1) <= 1).all() and bool((iou[m] > 0.2).all())
        w = torch.where(iou > 0.2, iou, torch.zeros_like(iou)).numpy()
        assert float(iou[m].sum()) == pytest.approx(_best_sum(w, 0, frozenset()))


# --- B4: change scores follow the NaN rule ----------------------------------------------------

def test_change_scores_nan_rule():
    logits = torch.zeros(2, 2, 8, 8)
    logits[:, 0] = 5  # never predicts change
    ev = build_task("change", 2).evaluator()
    ev.update(logits, torch.zeros(2, 8, 8, dtype=torch.long))  # no change in the split
    r = ev.compute()
    for k in ("precision_change", "recall_change", "f1_change", "iou_change"):
        assert math.isnan(r[k]), k
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    y[:, :4, :4] = 1
    ev = build_task("change", 2).evaluator()
    ev.update(logits[:1], y)
    r = ev.compute()
    assert math.isnan(r["precision_change"])
    assert r["recall_change"] == 0.0 and r["f1_change"] == 0.0 and r["iou_change"] == 0.0


# --- B5: label smoothing applied, or refused ---------------------------------------------------

@pytest.mark.parametrize("name,kw", [("multilabel", {}), ("seg_multilabel", {})])
def test_label_smoothing_refused_where_not_applied(name, kw):
    with pytest.raises(ValueError, match="label_smoothing"):
        build_task(name, 4, label_smoothing=0.1, **kw)
    build_task(name, 4, label_smoothing=0.0, **kw)


def test_label_smoothing_in_validation_loss():
    out, y = torch.randn(6, 4), torch.randint(0, 4, (6,))
    task = build_task("classification", 4, label_smoothing=0.2)
    ev = task.evaluator()
    ev.update(out, y)
    assert ev.compute()["loss"] == pytest.approx(float(task.loss(out, y)), rel=1e-5)

    seg = build_task("semantic", 3, label_smoothing=0.2)
    out, y = torch.randn(2, 3, 8, 8), torch.randint(0, 3, (2, 8, 8))
    ev = seg.evaluator()
    ev.update(out, y)
    assert ev.compute()["loss"] == pytest.approx(float(seg.loss(out, y)), rel=1e-5)


def test_semantic_change_applies_label_smoothing():
    task0, task1 = (build_task("semantic_change", 3, label_smoothing=s) for s in (0.0, 0.3))
    out = {k: torch.randn(1, c, 4, 4) for k, c in (("change", 2), ("lc1", 3), ("lc2", 3))}
    y = torch.zeros(1, 3, 4, 4, dtype=torch.long)
    y[:, 0, :2], y[:, 1, :2], y[:, 2, :2] = 1, 2, 1
    assert float(task0.loss(out, task0.targets(y))) != float(task1.loss(out, task1.targets(y)))


def _oxford_cache(tmp_path, n=8, meta=None):
    from manyseg_bench.datasets.base import save_split

    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (n, 32, 32, 3), dtype=np.uint8)
    masks = rng.integers(0, 3, (n, 32, 32)).astype(np.uint8)
    cache = tmp_path / "cache" / "oxford_pet"
    for split in ("train", "val"):
        save_split(cache, split, imgs, masks, [str(i) for i in range(n)])
    if meta is not None:
        (cache / "meta.json").write_text(json.dumps(meta))


def test_runner_refuses_label_smoothing_for_tasks_without_it(tmp_path, monkeypatch):
    pytest.importorskip("manyseg")
    from manyseg_bench.datasets import oxford_pet
    from manyseg_bench.experiments import run_training
    from manyseg_bench.providers import ModelSpec

    _oxford_cache(tmp_path)
    monkeypatch.setattr(oxford_pet, "TASK", "multilabel_seg", raising=False)
    with pytest.raises(ValueError, match="label_smoothing"):
        run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), "oxford_pet",
                     "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                     recipe_overrides={"precision": "fp32", "label_smoothing": 0.1})


# --- B6: AP does not depend on the order of tied scores ---------------------------------------

def test_average_precision_ties_order_free():
    s = torch.tensor([0.9, 0.5, 0.5, 0.5, 0.1])
    a = torch.tensor([1, 1, 0, 0, 1])
    b = torch.tensor([1, 0, 0, 1, 1])  # the same labels, tied block reordered
    ap_a, ap_b = tasks.average_precision(s, a), tasks.average_precision(s, b)
    assert ap_a == pytest.approx(ap_b)
    # sklearn: thresholds 0.9 (P 1, R 1/3), 0.5 (P 2/4, R 2/3), 0.1 (P 3/5, R 1)
    assert ap_a == pytest.approx(1 / 3 * 1 + 1 / 3 * 0.5 + 1 / 3 * 0.6)
    s2 = torch.tensor([0.9, 0.8, 0.3, 0.2])  # no ties: the mean precision at each positive
    assert tasks.average_precision(s2, torch.tensor([1, 0, 1, 0])) == pytest.approx(
        (1 + 2 / 3) / 2)
    assert math.isnan(tasks.average_precision(s2, torch.zeros(4)))


# --- B7: the recorded normalisation source -----------------------------------------------------

@pytest.mark.parametrize("meta,source", [
    ({"summary": {}}, "dataset constants"),
    ({"mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2]}, "meta.json (official train split)")])
def test_runner_records_norm_source(tmp_path, meta, source):
    pytest.importorskip("manyseg")
    from manyseg_bench.experiments import run_training
    from manyseg_bench.providers import ModelSpec

    _oxford_cache(tmp_path, meta=meta)
    rec = run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), "oxford_pet",
                       "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                       recipe_overrides={"precision": "fp32"})
    assert rec["norm_stats"]["source"].startswith(source)
    assert rec["task_settings"] == {"label_smoothing": 0.0}


# --- B8: stem adaptation recorded per provider ------------------------------------------------

def test_stem_adaptation_notes():
    from manyseg_bench.providers.base import stem_adaptation_note

    assert stem_adaptation_note("smp", 3, True) == []
    assert stem_adaptation_note("smp", 6, False) == []
    assert "scaled by 3/C" in stem_adaptation_note("smp", 6, True)[0]
    assert "zero" in stem_adaptation_note("fastai", 6, True, rule="fastai")[0]
    assert "x 1.5" in stem_adaptation_note("fastai", 2, True, rule="fastai")[0]
    assert "summed" in stem_adaptation_note("timm", 1, True)[0]


def test_manyseg_provider_records_stem_adaptation(monkeypatch):
    pytest.importorskip("manyseg")
    from manyseg_bench.providers import ModelSpec, get_provider

    built = get_provider("manyseg").build(
        ModelSpec("unet", "resnet18", weights=None, in_channels=5))
    assert built.notes == []  # random init: nothing adapted
    import manyseg

    class Stub(torch.nn.Module):  # no download: a pretrained build's model is stubbed
        def __init__(self, **kw):
            super().__init__()
            self.backbone = torch.nn.Conv2d(5, 1, 1)

    monkeypatch.setattr(manyseg, "create_model", Stub)
    built = get_provider("manyseg").build(ModelSpec("unet", "resnet18", in_channels=5))
    assert len(built.notes) == 1 and "scaled by 3/C" in built.notes[0]


# --- B9: prepare --size on fixed-size datasets ------------------------------------------------

def test_fixed_size_datasets_reject_other_sizes(tmp_path):
    from manyseg_bench.datasets import ai4boundaries_s2, bigearthnet, change_detection
    from manyseg_bench.datasets.base import check_native_size

    check_native_size(None, 120, "x")
    check_native_size(120, 120, "x")
    with pytest.raises(ValueError, match="fixed 120"):
        bigearthnet.prepare(tmp_path / "raw", tmp_path / "cache", size=64)
    with pytest.raises(ValueError, match="fixed 256"):
        ai4boundaries_s2.prepare(tmp_path / "raw", tmp_path / "cache", size=128)
    with pytest.raises(ValueError, match="fixed 512"):
        change_detection.prepare_second(tmp_path / "raw", tmp_path / "cache", size=256)


def test_pastis_ts_rejects_other_size(tmp_path):
    from manyseg_bench.datasets import pastis, pastis_ts
    from manyseg_bench.datasets.base import save_split

    src = tmp_path / "cache" / pastis.CACHE_NAME
    save_split(src, "train", {"s2": np.zeros((1, 16, 16, 2), np.float16)},
               np.zeros((1, 16, 16), np.uint8), ["p_0"])
    with pytest.raises(ValueError, match="fixed 16"):
        pastis_ts.prepare(tmp_path / "raw", tmp_path / "cache" / "pastis_ts", size=32)


# --- B10: an empty training loader raises -----------------------------------------------------

def test_empty_training_loader_raises():
    from manyseg_bench.datasets import GPULoader

    imgs, m = np.zeros((3, 16, 16, 3), np.uint8), np.zeros((3, 16, 16), np.uint8)
    kw = dict(mean=[0, 0, 0], std=[1, 1, 1], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="training loader is empty"):
        GPULoader(imgs, m, 16, train=True, **kw)
    assert len(GPULoader(imgs, m, 16, train=False, **kw)) == 1  # evaluation keeps the batch
    assert len(GPULoader(imgs, m, 2, train=True, **kw)) == 1


# --- B11: semantic change detection respects ignore_index -------------------------------------

def test_semantic_change_ignore_index():
    task = build_task("semantic_change", 3, ignore_index=255)
    y = torch.zeros(1, 3, 4, 4, dtype=torch.long)
    y[:, 0, :2], y[:, 1, :2], y[:, 2, :2] = 1, 2, 1
    out = {k: torch.randn(1, c, 4, 4) for k, c in (("change", 2), ("lc1", 3), ("lc2", 3))}
    clean = task.evaluator()
    clean.update(out, y)
    y_ign = y.clone()
    y_ign[:, :, 3] = 255  # an ignored row in every channel
    t = task.targets(y_ign)
    assert bool((t["change"][:, 3] == -100).all()) and bool((t["lc1"][:, 3] == -100).all())
    assert math.isfinite(float(task.loss(out, t)))
    ev = task.evaluator()
    ev.update(out, y_ign)  # crashed with 255 labels
    assert int(ev.scd.sum()) == 2 * 12 and int(ev.bin.sum()) == 12
    # the ignored row is unchanged and land-cover-free in y: dropping it removes its cells
    sub = task.evaluator()
    sub.update({k: v[..., :3, :] for k, v in out.items()}, y[..., :3, :])
    assert torch.equal(ev.scd, sub.scd) and torch.equal(ev.bin, sub.bin)
    # all-ignored batch: a finite (zero) loss
    allign = task.targets(torch.full((1, 3, 4, 4), 255))
    assert float(task.loss(out, allign)) == 0.0


def test_manyseg_stem_note_uses_resolved_input_settings():
    from manyseg_bench.providers.manyseg_provider import _input_stem_note

    assert _input_stem_note({"in_channels": 4}, True, {}) != []
    assert _input_stem_note({"in_channels": 4, "pretrained": False}, True, {}) == []
    assert _input_stem_note({"in_channels": 4, "backbone": "shared:a",
                             "channel_adapter": "projection"}, True, {}) == []
