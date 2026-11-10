"""Regressions for the third library sweep of manyseg-bench (B1-B12 and evaluate's vectors).
Synthetic data only, no downloads."""

import json
import math
import sys

import numpy as np
import pytest
import torch
from manyseg_bench.providers import ModelSpec

# --- B1: `evaluate` prints errors and the task's primary metric ---------------------------------


def test_cli_evaluate_prints_errors_and_the_primary_metric(monkeypatch, capsys):
    import manyseg_bench.experiments as E
    from manyseg_bench.cli import main

    def fake(run, **kw):
        return {"primary_metric": "f1_change", "splits": ["val", "test"], "path": "x.json",
                "variants": {
                    "fp32": {"device": "cpu", "size_bytes": 1,
                             "val": {"miou": 0.5, "f1_change": 0.7, "agreement": 0.99},
                             "test": {"miou": 0.4, "f1_change": 0.6, "agreement": None}},
                    "int8_pt2e": {"device": "cpu", "error": "RuntimeError: export failed"}}}

    monkeypatch.setattr(E, "evaluate_run", fake)
    assert main(["evaluate", "r.json"]) == 0
    out = capsys.readouterr().out
    assert "int8_pt2e error: RuntimeError: export failed" in out
    fp32 = next(ln for ln in out.splitlines() if ln.startswith("fp32"))
    assert "'f1_change': 0.7" in fp32 and "'f1_change': 0.6" in fp32
    assert "miou" not in fp32 and "'agreement': 'n/a'" in fp32


# --- B2: floating-point variants need no ManySeg -------------------------------------------------


def test_float_variant_size_does_not_import_manyseg_quantize(monkeypatch):
    from manyseg_bench.experiments.evaluate import _score
    from manyseg_bench.profiling import serialized_size_bytes
    from manyseg_bench.tasks import build_task

    monkeypatch.setitem(sys.modules, "manyseg.quantize", None)  # import raises
    model = torch.nn.Conv2d(3, 2, 1)
    x, y = torch.randn(2, 3, 8, 8), torch.randint(0, 2, (2, 8, 8))
    task = build_task("semantic", 2)
    info = type("I", (), {"ignore_index": None})()
    res = _score("fp32", model, lambda split: [(x, y)], ("val",), task, info,
                 torch.device("cpu"), "fp32", 1)
    assert res["size_bytes"] == serialized_size_bytes(model)
    assert 0.0 <= res["val"]["miou"] <= 1.0 and res["val"]["agreement"] == 1.0


# --- B3: SMP / fastai refuse image-level output --------------------------------------------------


@pytest.mark.parametrize("provider", ["smp", "fastai"])
def test_segmentation_providers_refuse_image_output(provider):
    from manyseg_bench.providers import get_provider

    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=5, output="image")
    with pytest.raises(ValueError, match="segmentation models only.*timm or manyseg"):
        get_provider(provider).build(spec, img_size=(32, 32))


# --- B4: --subset also subsets the series extras -------------------------------------------------


@pytest.mark.parametrize("name", ["pastis_ts", "ai4boundaries_s2"])
def test_subset_applies_to_series_extras(tmp_path, name):
    from manyseg_bench.datasets import GPULoader, get_dataset
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments.runner import split_getter

    ds = get_dataset(name)
    (inp, c), = ds.INFO.inputs.items()
    t = ds.INFO.timeseries[inp]
    n, cache = 6, tmp_path / "cache"
    for s in ("train", "val", "test"):
        save_split(cache, s, {inp: np.random.rand(n, t, 16, 16, c).astype(np.float16)},
                   np.random.randint(0, 2, (n, 16, 16)).astype(np.uint8),
                   [f"p{i}" for i in range(n)])
        np.save(cache / f"{s}_dates.npy", np.tile(np.arange(float(t)), (n, 1)))
        np.save(cache / f"{s}_valid.npy", np.ones((n, t), bool))
    sp = split_getter(ds, cache, cache, "official", subset=2)("train")
    assert len(sp.names) == 2 and all(len(v) == 2 for v in sp.vectors.values())
    loader = GPULoader(sp.images, sp.masks, 2, mean={inp: [0.0] * c}, std={inp: [1.0] * c},
                       device=torch.device("cpu"), train=True, stack=False, vectors=sp.vectors)
    x, _ = next(iter(loader))
    assert x[f"{inp}_dates"].shape == (2, t)


# --- B5: default task kwargs group with omitted ones ---------------------------------------------


def _train_record(dataset="bigearthnet", task_kwargs=None, seed=0, pm="map_macro", **extra):
    return {"kind": "train", "dataset": {"name": dataset, "test_splits": ["test"]},
            "recipe": {"name": "standard"}, "provider": "timm",
            "spec": {"arch": "none", "encoder": "resnet18"}, "primary_metric": pm,
            "final_val": {pm: 0.5}, "params": 1e6, "train_time_s": 1,
            "weights_fingerprint": "a", "task_kwargs": task_kwargs, "seed": seed, **extra}


def test_group_key_canonicalises_default_task_kwargs():
    from manyseg_bench.reporting import training_table
    from manyseg_bench.reporting.summary import _config_label, group_key

    old = _train_record(task_kwargs={})
    new = _train_record(task_kwargs={"label_smoothing": 0.0}, seed=1)
    smooth = _train_record(task_kwargs={"label_smoothing": 0.1}, seed=2)
    assert group_key(old) == group_key(new) != group_key(smooth)
    assert "task=" not in _config_label(group_key(new))
    assert 'task={"label_smoothing": 0.1}' in _config_label(group_key(smooth))
    assert "| 2 |" in training_table([old, new])
    # dataset TASK_KWARGS are effective defaults too (ai4boundaries: boundary_width=2.0)
    a = _train_record("ai4boundaries", {}, pm="pq")
    b = _train_record("ai4boundaries", {"boundary_width": 2.0, "head_scale": 1}, 1, pm="pq")
    c = _train_record("ai4boundaries", {"boundary_width": 1.0}, 2, pm="pq")
    assert group_key(a) == group_key(b) != group_key(c)


# --- B6: seg_multilabel val loss is the training objective ---------------------------------------


def test_seg_multilabel_loss_is_the_training_objective():
    from manyseg_bench.tasks import build_task

    task = build_task("seg_multilabel", 2, 255, cls_weight=3.0)
    y = torch.zeros(2, 8, 8, dtype=torch.long)
    y[:, :4] = 1
    out = {"seg": torch.zeros(2, 2, 8, 8), "logits": torch.zeros(2, 2)}
    r = task.evaluator()
    r.update(out, y)
    r = r.compute()
    objective = float(task.loss(out, task.targets(y)))
    assert r["loss"] == pytest.approx(objective) == pytest.approx(4 * math.log(2))
    assert r["seg"]["loss"] == pytest.approx(math.log(2))
    assert r["cls"]["loss"] == pytest.approx(math.log(2))


# --- B7: device-aware, failure-proof GPU contention ----------------------------------------------


def _fake_smi(monkeypatch, gpus: str, apps: str):
    import subprocess

    from manyseg_bench import env

    def run(cmd, **kw):
        out = gpus if any(a.startswith("--query-gpu") for a in cmd) else apps
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(env.subprocess, "run", run)
    return env


def test_gpu_contention_reads_the_runs_gpu_only(monkeypatch):
    env = _fake_smi(monkeypatch,
                    "0, GPU-aaa, 90, 9000, 24000\n1, GPU-bbb, 2, 100, 24000\n",
                    "GPU-aaa, 111, 500\nGPU-aaa, 112, 500\n")
    monkeypatch.setattr(env, "_cuda_uuid", lambda i: "bbb" if i == 1 else "aaa")
    r = env.gpu_contention("cuda:1")
    assert r["gpu_index"] == 1 and r["gpu_util_pct"] == 2 and r["mem_used_mib"] == 100
    assert r["other_compute_processes"] == 0 and r["contended"] is False
    r0 = env.gpu_contention("cuda:0")
    assert r0["gpu_util_pct"] == 90 and r0["other_compute_processes"] == 2
    monkeypatch.setattr(env, "_cuda_uuid", lambda i: None)  # no UUID: matched by index
    assert env.gpu_contention("cuda:1")["gpu_util_pct"] == 2


@pytest.mark.parametrize("gpus", ["0, GPU-aaa, [N/A], 1, 2\n", "", "garbage\n"])
def test_gpu_contention_malformed_output_returns_none(monkeypatch, gpus):
    env = _fake_smi(monkeypatch, gpus, "")
    monkeypatch.setattr(env, "_cuda_uuid", lambda i: "aaa")
    assert env.gpu_contention("cuda:0") is None


def test_gpu_contention_none_off_cuda_and_without_nvidia_smi(monkeypatch):
    from manyseg_bench import env

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(env.subprocess, "run", boom)
    assert env.gpu_contention("cpu") is None and env.gpu_contention("mps") is None
    assert env.gpu_contention("cuda:0") is None


# --- B8 (+B7): profile builds image models for timm and skips GPU telemetry off CUDA -------------


def test_profile_timm_defaults_to_image_output_on_cpu(tmp_path, monkeypatch, capsys):
    pytest.importorskip("timm")
    from manyseg_bench import env
    from manyseg_bench.cli import main

    def no_call(*a, **k):
        raise AssertionError("gpu_contention called for a CPU profile")

    monkeypatch.setattr(env, "gpu_contention", no_call)
    rc = main(["profile", "--provider", "timm", "--encoder", "resnet18", "--weights", "none",
               "--device", "cpu", "--size", "32,32", "--batch-sizes", "1",
               "--train-batch-size", "0", "--precision", "fp32", "--out", str(tmp_path)])
    assert rc == 0
    rec = json.loads(next(tmp_path.glob("*_profile_timm_*.json")).read_text())
    assert rec["spec"]["output"] == "image" and rec["gpu_contention_at_end"] is None


def test_profile_output_flag_overrides_the_default(tmp_path):
    from manyseg_bench.cli import main

    with pytest.raises(ValueError, match="output must be 'image'"):
        main(["profile", "--provider", "timm", "--encoder", "resnet18", "--weights", "none",
              "--device", "cpu", "--size", "32,32", "--output", "pixel",
              "--out", str(tmp_path)])


# --- B9: fixed-size datasets refuse --size before downloading ------------------------------------


@pytest.mark.parametrize("name", ["sen1floods11", "eurosat", "ai4boundaries", "pastis"])
def test_fixed_size_prepare_refuses_size_before_download(tmp_path, monkeypatch, name):
    import subprocess

    from manyseg_bench.datasets import get_dataset

    ds = get_dataset(name)

    def no_download(*a, **k):
        raise AssertionError("downloaded before checking size")

    if hasattr(ds, "download"):
        monkeypatch.setattr(ds, "download", no_download)
    monkeypatch.setattr(subprocess, "run", no_download)
    with pytest.raises(ValueError, match=f"fixed {ds.SIZE} x {ds.SIZE}"):
        ds.prepare(tmp_path / "raw", tmp_path / "cache", size=ds.SIZE // 2)


def test_prepare_help_lists_every_fixed_size_dataset(capsys):
    from manyseg_bench.cli import main

    with pytest.raises(SystemExit):
        main(["prepare", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    for name in ("sen1floods11", "eurosat", "ai4boundaries,", "ai4boundaries_s2", "pastis,",
                 "pastis_ts", "bigearthnet", "second"):
        assert name in out


# --- B10: float pixel labels stay float in training augmentation ---------------------------------


def test_train_augmentation_keeps_float_pixel_labels():
    from manyseg_bench.datasets import GPULoader

    imgs = np.zeros((4, 16, 16, 3), np.uint8)
    masks = np.full((4, 16, 16), 0.25, np.float32)
    loader = GPULoader(imgs, masks, 2, mean=[0.0] * 3, std=[1.0] * 3,
                       device=torch.device("cpu"), train=True)
    _, y = next(iter(loader))
    assert y.dtype == torch.float32 and torch.allclose(y, torch.full_like(y, 0.25))
    ints = GPULoader(imgs, np.ones((4, 16, 16), np.uint8), 2, mean=[0.0] * 3, std=[1.0] * 3,
                     device=torch.device("cpu"), train=True)
    _, y = next(iter(ints))
    assert y.dtype == torch.int64 and (y == 1).all()


# --- B11: onecycle refuses a warmup it would ignore ----------------------------------------------


def test_onecycle_refuses_warmup_epochs():
    from manyseg_bench.training.recipe import RECIPES, Recipe, get_recipe
    from manyseg_bench.training.trainer import lr_factor

    for name in RECIPES:  # no built-in recipe trips the check
        get_recipe(name)
    with pytest.raises(ValueError, match="fixed warmup.*30%"):
        get_recipe("fastai_like", warmup_epochs=1)
    with pytest.raises(ValueError, match="warmup_epochs=0"):
        Recipe(schedule="onecycle")  # the default warmup_epochs=1.0
    Recipe(schedule="onecycle", warmup_epochs=0)
    # Switching schedule on an existing recipe drops its inherited warmup ...
    assert get_recipe("standard", schedule="onecycle").warmup_epochs == 0
    # ... but an explicit warmup alongside onecycle is still refused.
    with pytest.raises(ValueError, match="warmup_epochs=0"):
        get_recipe("standard", schedule="onecycle", warmup_epochs=2)
    with pytest.raises(ValueError, match="onecycle"):
        lr_factor(0, 100, 5, "onecycle")


# --- B12: every declared test split in the training table ----------------------------------------


def test_training_table_reports_every_test_split():
    from manyseg_bench.reporting import training_table

    rs = [{**_train_record("sen1floods11", {}, s, pm="miou"),
           "dataset": {"name": "sen1floods11", "test_splits": ["test", "bolivia"]},
           "test": {"miou": 0.8}, "bolivia": {"miou": 0.6 + s / 10}} for s in (0, 1)]
    table = training_table(rs)
    assert "test: 0.8000 ± 0.0000" in table and "bolivia: 0.6500 ± 0.0707" in table
    no_test = training_table([_train_record("sen1floods11", {}, pm="miou")])
    assert no_test.splitlines()[-1].split("|")[13].strip() == "n/a"  # test primary column


# --- evaluate_run passes vector inputs to the loader ---------------------------------------------


def test_evaluate_run_passes_vectors_to_the_loader(tmp_path, monkeypatch):
    pytest.importorskip("segmentation_models_pytorch")
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import evaluate as ev_mod
    from manyseg_bench.experiments import evaluate_run, run_training

    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8)
    masks = rng.integers(0, 3, (4, 32, 32)).astype(np.uint8)
    cache = tmp_path / "cache" / "oxford_pet"
    for split in ("train", "val", "test"):
        save_split(cache, split, imgs, masks, [str(i) for i in range(4)])
    rec = run_training("smp", ModelSpec("unet", "resnet18", weights=None), "oxford_pet",
                       "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                       recipe_overrides={"precision": "fp32", "batch_size": 2},
                       save_model=True)
    real_getter, seen = ev_mod.split_getter, []

    def getter(*a, **k):
        get = real_getter(*a, **k)

        def with_vectors(name):
            s = get(name)
            s.vectors = {"meta": np.zeros((len(s.names), 2), np.float32)}
            return s
        return with_vectors

    class Loader(ev_mod.GPULoader):
        def __init__(self, *a, vectors=None, **k):
            seen.append(vectors)
            super().__init__(*a, vectors=vectors, **k)

    monkeypatch.setattr(ev_mod, "split_getter", getter)
    monkeypatch.setattr(ev_mod, "GPULoader", Loader)
    out = evaluate_run(rec["path"], variants=("fp32",), device="cpu", data=tmp_path,
                       out_dir=tmp_path / "eval")
    assert "error" not in out["variants"]["fp32"] and out["primary_metric"] == "miou"
    assert seen and all(v is not None and "meta" in v for v in seen)
