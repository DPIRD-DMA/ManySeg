"""Regression tests for the pre-release sweep: optional dependencies, normalisation of pooled
splits under remote-sensing weights, scipy-independent point matching, the CLI's evaluate
printout, report grouping and the metric conventions (absent classes, edge-free images,
validation loss per valid pixel)."""

import json
import math
import pathlib
import subprocess
import sys
import tomllib

import numpy as np
import pytest
import torch
from manyseg_bench.evaluation import ConfusionMatrix, boundary_f1
from manyseg_bench.evaluation.objects import PointMetrics, match_points
from manyseg_bench.reporting.summary import group_key, training_table
from manyseg_bench.tasks import build_task

PKG = pathlib.Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]


# --- B13: the bench runs semantic / image tasks without ManySeg -----------------------------

def test_semantic_and_image_tasks_run_without_manyseg():
    code = """
import sys
for m in ("manyseg", "manyseg.losses", "manyseg.utils", "manyseg.instances"):
    sys.modules[m] = None
import torch
from manyseg_bench.tasks import build_task
from manyseg_bench import losses
t = build_task("semantic", 3, 255)
y = torch.randint(0, 3, (2, 8, 8)); out = torch.randn(2, 3, 8, 8)
assert torch.isfinite(t.loss(out, y))
ev = t.evaluator(); ev.update(out, y); assert 0 <= ev.compute()["miou"] <= 1
c = build_task("classification", 3)
ev = c.evaluator(); ev.update(torch.randn(4, 3), torch.tensor([0, 1, 2, 0])); ev.compute()
try:
    losses.dice_bce
except ImportError as e:
    assert "manyseg" in str(e)
else:
    raise AssertionError("dice_bce resolved without ManySeg")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_losses_resolve_with_manyseg():
    pytest.importorskip("manyseg.losses")
    import manyseg.losses as ml
    from manyseg_bench import losses

    assert losses.dice_bce is ml.dice_bce and "soft_dice" in dir(losses)
    with pytest.raises(AttributeError):
        losses.not_a_loss  # noqa: B018


# --- B16, extras, classifiers -------------------------------------------------------------

def test_pyproject_extras_and_classifiers():
    meta = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]
    extras = meta["optional-dependencies"]
    assert any(d.startswith("pandas") for d in extras["geo"])  # datasets/bigearthnet.py
    assert any(d.startswith("timm") for d in extras["timm"])   # providers/timm_provider.py
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    tested = {"3.11", "3.12", "3.14"}
    assert all(f'"{v}"' in ci for v in tested)
    declared = {c.rsplit(" ", 1)[1] for c in meta["classifiers"]
                if c.startswith("Programming Language :: Python :: 3.")}
    assert tested <= declared
    assert "geo" in ci.split("manyseg-bench[", 1)[1].split("]", 1)[0]


# --- B14: registry weights keep their normalisation on pooled (event) splits ----------------

def test_event_split_keeps_weights_normalisation(tmp_path, monkeypatch):
    pytest.importorskip("manyseg.weights")
    from manyseg_bench import datasets
    from manyseg_bench.datasets.base import DatasetInfo, save_split
    from manyseg_bench.experiments import run_training
    from manyseg_bench.providers import ModelSpec
    from test_weights_bench import DS_BANDS, fake_hub

    fake_hub(monkeypatch)

    class Fake:
        INFO = DatasetInfo(name="fake_s2_ev", num_classes=2, in_channels=10,
                           class_names=("a", "b"), mean={"s2": [0.1] * 10},
                           std={"s2": [0.05] * 10}, inputs={"s2": 10})
        BAND_NAMES = {"s2": tuple(DS_BANDS)}
        DN_PER_UNIT = {"s2": 1e4}

        @staticmethod
        def event_folds(names, k, fold):
            idx = list(range(len(names)))
            return {"train": idx[:4], "val": idx[4:6], "test": idx[6:]}

    monkeypatch.setitem(datasets.DATASETS, "fake_s2_ev", Fake)
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        save_split(tmp_path / "cache" / "fake_s2_ev", split,
                   {"s2": rng.random((n, 32, 32, 10)).astype(np.float16) * 0.3},
                   rng.integers(0, 2, (n, 32, 32)).astype(np.uint8),
                   [f"{split}{i}" for i in range(n)])
    spec = ModelSpec("unet", "resnet18", weights="ssl4eo_s2_moco")
    rec = run_training("manyseg", spec, "fake_s2_ev", "smoke", device="cpu", data=tmp_path,
                       out_dir=tmp_path / "runs", split="events:3:0",
                       recipe_overrides={"batch_size": 2, "precision": "fp32"})
    assert rec["split_scheme"] == "events:3:0"
    assert rec["norm_stats"]["source"].startswith("weights resnet18.ssl4eo_s2_moco")
    assert rec["norm_stats"]["mean"]["s2"] == [0.0] * 10  # not the fold's data statistics


def test_split_train_stats_dict_and_array():
    from manyseg_bench.experiments.runner import split_train_stats

    a = np.stack([np.zeros((4, 4)), np.full((4, 4), 2.0)], -1)[None].astype(np.float32)
    m, s = split_train_stats(a)
    assert m == pytest.approx([0.0, 2.0])
    md, _ = split_train_stats({"x": a})
    assert md["x"] == pytest.approx([0.0, 2.0])


# --- B15: point matching is maximum-cardinality, with or without scipy ---------------------

def test_match_points_maximum_matching_example():
    pred = torch.tensor([[0.0, 0.0], [0.0, 4.0]])
    gt = torch.tensor([[0.0, 2.0], [0.0, -2.5]])
    # greedy by distance would take pred1-gt0 (2.0) first and leave pred0-gt1 (2.5) only:
    # pred0-gt0 (2.0) is taken first, then pred1 has nothing -> 1 TP. The optimum is 2.
    assert match_points(pred, gt, 3) == (2, 0, 0)


def test_match_points_long_chain_no_recursion_error():
    n = 5000
    pred = torch.stack([torch.arange(n) * 2.0, torch.zeros(n)], 1)
    gt = pred + torch.tensor([1.0, 0.0])
    # each pred reaches gt i-1 and i; dropping the last gt forces a long augmenting chain
    assert match_points(pred, gt[:-1], 1.5) == (n - 1, 1, 0)
    assert match_points(pred.flip(0), gt[1:], 1.5) == (n - 1, 1, 0)


def test_match_points_without_scipy_matches_scipy(monkeypatch):
    sopt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(300):
        n, m = rng.integers(0, 20, 2)
        cases.append((torch.tensor(rng.random((n, 2)) * 15), torch.tensor(rng.random((m, 2)) * 15),
                      float(rng.uniform(0.5, 4))))
    for p, g, r in cases:
        ref = 0
        if len(p) and len(g):
            d = torch.cdist(p, g)
            cost = torch.where(d <= r, d, torch.full_like(d, 1e9)).numpy()
            rows, cols = sopt.linear_sum_assignment(cost)
            ref = int(sum(cost[i, j] < 1e9 for i, j in zip(rows, cols)))
        assert match_points(p, g, r) == (ref, len(p) - ref, len(g) - ref)
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    pm = PointMetrics(3.0)
    pm.update(torch.tensor([[0.0, 0.0], [0.0, 4.0]]), torch.tensor([[0.0, 2.0], [0.0, -2.5]]))
    assert pm.compute()["f1"] == 1.0


# --- B17: evaluate printout with agreement None --------------------------------------------

def test_cli_evaluate_prints_missing_agreement(monkeypatch, capsys):
    from manyseg_bench import cli, experiments

    rec = {"variants": {"fp32": {"device": "cpu",
                                 "val": {"miou": 0.51234, "agreement": None}}},
           "path": "x.json"}
    monkeypatch.setattr(experiments, "evaluate_run", lambda *a, **k: rec)
    assert cli.main(["evaluate", "run.json", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "n/a" in out and "0.5123" in out


# --- B18: report grouping covers every config field but the seed ---------------------------

def _run(seed=0, **over):
    r = {"kind": "train", "tag": "", "provider": "smp", "seed": seed,
         "spec": {"arch": "unet", "encoder": "resnet34", "weights": "tv_in1k",
                  "input_mode": "stack", "options": {}},
         "dataset": {"name": "oxford_pet"},
         "recipe": {"name": "standard", "epochs": 30, "batch_size": 16},
         "task_kwargs": {"label_smoothing": 0.0}, "subset": None, "split_scheme": "official",
         "weights_fingerprint": "f", "params": 1e6, "primary_metric": "miou",
         "final_val": {"miou": 0.5 + seed / 100}, "train_time_s": 1.0}
    for k, v in over.items():
        if k in ("weights", "input_mode"):
            r["spec"] = {**r["spec"], k: v}
        elif k == "epochs":
            r["recipe"] = {**r["recipe"], "epochs": v}
        else:
            r[k] = v
    return r


@pytest.mark.parametrize("over", [{"weights": None}, {"input_mode": "dict"}, {"epochs": 5},
                                  {"subset": 8}, {"split_scheme": "events:5:0"},
                                  {"task_kwargs": {"label_smoothing": 0.1}}])
def test_group_key_separates_configs(over):
    assert group_key(_run(0)) == group_key(_run(1))  # seeds of one config
    assert group_key(_run(0)) != group_key(_run(0, **over))


def test_group_key_old_records_group_with_defaults():
    old = _run(1)
    del old["split_scheme"], old["subset"]
    old["spec"] = {k: v for k, v in old["spec"].items() if k != "input_mode"}
    assert group_key(old) == group_key(_run(0))
    table = training_table([_run(0), old, _run(0, epochs=5)])
    rows = table.splitlines()[2:]
    assert len(rows) == 2 and any("| 2 |" in ln for ln in rows)


# --- X9: metric conventions -----------------------------------------------------------------

def test_confusion_matrix_absent_classes_are_nan_and_excluded():
    cm = ConfusionMatrix(3)
    y = torch.tensor([[0, 0, 1, 1]])
    p = torch.tensor([[0, 2, 1, 1]])   # class 2 predicted, absent from the ground truth
    cm.update(p, y)
    m = cm.compute()
    assert math.isnan(m["iou"][2]) and math.isnan(m["dice"][2])
    assert m["miou"] == pytest.approx(np.nanmean(m["iou"]))
    assert m["miou"] == pytest.approx((0.5 + 1.0) / 2)  # headline rule unchanged
    assert m["pred_only_classes"] == [2]


def test_boundary_f1_edge_free_images_are_nan_and_skipped():
    z = torch.zeros(1, 16, 16, dtype=torch.long)
    assert math.isnan(boundary_f1(z, z).item())
    t = build_task("semantic", 2)
    ev = t.evaluator()
    y = torch.zeros(2, 16, 16, dtype=torch.long)
    y[1, :, 8:] = 1
    out = torch.stack([1 - y, y], 1).float() * 10  # perfect: image 0 has no edge at all
    ev.update(out, y)
    assert ev.compute()["boundary_f1"] == 1.0
    ev0 = t.evaluator()
    ev0.update(out[:1], y[:1])
    assert math.isnan(ev0.compute()["boundary_f1"])


def test_semantic_val_loss_is_per_valid_pixel():
    t = build_task("semantic", 2, 255)
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    y[0, :2] = 255                       # half the pixels ignored
    out = torch.randn(1, 2, 4, 4)
    ev = t.evaluator()
    ev.update(out, y)
    ref = torch.nn.functional.cross_entropy(out, y, ignore_index=255)  # mean over valid
    assert ev.compute()["loss"] == pytest.approx(float(ref), rel=1e-5)


def test_classification_absent_class_f1_is_nan():
    t = build_task("classification", 3)
    ev = t.evaluator()
    ev.update(torch.tensor([[5.0, 0, 0], [0, 5, 0], [0, 0, 5]]), torch.tensor([0, 1, 1]))
    m = ev.compute()
    assert math.isnan(m["f1"][2])
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3) / 2)


def test_multilabel_seg_miou_over_classes_with_positives():
    t = build_task("multilabel_seg", 2)
    y = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    y[0, 0, 0] = 1                       # class 0: top row; class 1: no positives
    logits = torch.full((1, 2, 2, 2), -5.0)
    logits[0, 0, 0] = 5.0
    ev = t.evaluator()
    ev.update(logits, y)
    m = ev.compute()
    assert m["miou"] == 1.0 and math.isnan(m["iou"][1]) and math.isnan(m["recall"][1])
    assert math.isnan(m["precision"][1])
    assert json.loads(json.dumps(m["iou"]))[0] == 1.0
