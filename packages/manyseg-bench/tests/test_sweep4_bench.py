"""Regressions for the fourth library sweep of manyseg-bench (B1-B7). Synthetic data only, no
downloads."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from manyseg_bench.providers import ModelSpec

REPO = Path(__file__).resolve().parents[3]

# --- B1: LEVIR-CD edge padding is ignored ---------------------------------------------------------


def _png_bytes(a: np.ndarray) -> dict:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def test_levircd_padding_is_ignored_in_prepare_and_change_frac(tmp_path, monkeypatch):
    from manyseg_bench.datasets import change_detection as cd

    assert cd.LEVIR_INFO.ignore_index == 255
    rgb = np.zeros((1024, 1024, 3), np.uint8)
    row = {"image_name": "a.png", "t1_image": _png_bytes(rgb), "t2_image": _png_bytes(rgb),
           "change_mask": _png_bytes(np.full((1024, 1024), 255, np.uint8))}  # all changed
    monkeypatch.setattr(cd, "_parquet_rows", lambda raw, split: iter([row]))
    summary = cd.prepare_levircd(tmp_path / "raw", tmp_path / "cache", size=300,
                                 expected={"train": 1, "val": 1, "test": 1})
    masks = np.load(tmp_path / "cache" / "train_masks.npy")
    assert (masks == 255).any()  # 300 does not divide 1024: edge tiles padded
    assert summary["train"]["change_frac"] == 1.0  # over valid pixels only (was 0.728)


def test_levircd_trains_and_evaluates_with_padded_tiles(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import evaluate_run, run_training

    rng = np.random.default_rng(0)
    for s in ("train", "val", "test"):
        m = rng.integers(0, 2, (4, 32, 32)).astype(np.uint8)
        m[:, 24:, :] = 255  # padded edge, as prepare writes it
        save_split(tmp_path / "cache" / "levircd", s,
                   {k: rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8) for k in ("t1", "t2")},
                   m, [str(i) for i in range(4)])
    rec = run_training("smp", ModelSpec("unet", "resnet18", weights=None), "levircd", "smoke",
                       device="cpu", data=tmp_path, out_dir=tmp_path / "runs",
                       evaluate_test=True, save_model=True,
                       recipe_overrides={"batch_size": 2, "precision": "fp32"})
    assert rec["dataset"]["ignore_index"] == 255
    assert 0.0 <= rec["test"][rec["primary_metric"]] <= 1.0
    ev = evaluate_run(rec["path"], variants=("fp32",), device="cpu", data=tmp_path,
                      out_dir=tmp_path / "eval")
    assert ev["variants"]["fp32"]["test"]["agreement"] == 1.0
    # evaluate records hold the DatasetInfo dict, as train records do
    assert ev["dataset"] == rec["dataset"]


# --- B2: contention means other processes, not this run's own utilisation -------------------------


def _fake_smi(monkeypatch, gpus: str, apps: str):
    from manyseg_bench import env

    def run(cmd, **kw):
        out = gpus if any(a.startswith("--query-gpu") for a in cmd) else apps
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(env.subprocess, "run", run)
    monkeypatch.setattr(env, "_cuda_uuid", lambda i: "aaa")
    return env


def test_gpu_contention_ignores_own_utilisation(monkeypatch):
    me = os.getpid()
    env = _fake_smi(monkeypatch, "0, GPU-aaa, 99, 20000, 24000\n", f"GPU-aaa, {me}, 19000\n")
    r = env.gpu_contention("cuda:0")
    assert r["gpu_util_pct"] == 99 and r["mem_used_mib"] == 20000  # telemetry kept
    assert r["other_compute_processes"] == 0 and r["other_mem_mib"] == 0
    assert r["contended"] is False
    env = _fake_smi(monkeypatch, "0, GPU-aaa, 1, 600, 24000\n",
                    f"GPU-aaa, {me}, 100\nGPU-aaa, 1, 500\nGPU-bbb, 2, 900\n")
    r = env.gpu_contention("cuda:0")
    assert r["other_compute_processes"] == 1 and r["other_mem_mib"] == 500
    assert r["contended"] is True


# --- B3: code_version checks the whole repository ------------------------------------------------


def test_code_version_dirty_anywhere_in_the_repository(tmp_path, monkeypatch):
    from manyseg_bench import env

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        *a], check=True, capture_output=True)

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "core.py").write_text("x = 1\n")
    pkg = tmp_path / "packages" / "bench" / "src"
    pkg.mkdir(parents=True)
    (pkg / "env.py").write_text("")
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "init")
    monkeypatch.delenv("MANYSEG_CODE_SHA", raising=False)
    monkeypatch.setattr(env, "__file__", str(pkg / "env.py"))
    clean = env.code_version()
    assert clean and not clean.endswith("-dirty")
    (tmp_path / "untracked.txt").write_text("scratch")  # untracked files do not count
    assert env.code_version() == clean
    (tmp_path / "lib" / "core.py").write_text("x = 2\n")  # outside the package's directory
    assert env.code_version() == f"{clean}-dirty"
    monkeypatch.setenv("MANYSEG_CODE_SHA", "abc1234")  # explicit override still wins
    assert env.code_version() == "abc1234"


# --- B4: analyse_runs.py summarises train records only --------------------------------------------


def _train_record(tag: str, val: float) -> dict:
    return {"kind": "train", "tag": tag, "spec": {"arch": "unet"}, "primary_metric": "miou",
            "dataset": {"name": "d", "test_splits": ["test"]}, "final_val": {"miou": val},
            "test": {"miou": val}}


def _analyse(*args):
    return subprocess.run([sys.executable, str(REPO / "experiments" / "analyse_runs.py"), *args],
                          capture_output=True, text=True)


def test_analyse_runs_skips_profile_evaluate_and_non_object_json(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for i, (tag, v) in enumerate([("a", 0.5), ("a", 0.6), ("b", 0.7)]):
        (runs / f"t{i}.json").write_text(json.dumps(_train_record(tag, v)))
    # evaluate (dataset is the DatasetInfo dict now, formerly a string) and profile records
    (runs / "0_evaluate.json").write_text(json.dumps(
        {"kind": "evaluate", "tag": "a", "spec": {"arch": "unet"}, "dataset": "d"}))
    (runs / "0_profile.json").write_text(json.dumps({"kind": "profile", "tag": "a"}))
    (runs / "log.json").write_text("[1, 2]")
    r = _analyse(str(runs), "--ref", "a")
    assert r.returncode == 0, r.stderr
    rows = [ln for ln in r.stdout.splitlines() if ln.startswith("| a ") or ln.startswith("| b ")]
    assert len(rows) == 2 and "| a | 2 |" in r.stdout and "| b | 1 |" in r.stdout


def test_analyse_runs_without_train_records_says_so(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"kind": "profile"}))
    r = _analyse(str(tmp_path), "--ref", "a")
    assert r.returncode != 0 and "no training records" in r.stderr
    assert "Traceback" not in r.stderr


# --- B5: load_runs skips non-object JSON ----------------------------------------------------------


def test_load_runs_skips_non_dict_json(tmp_path):
    from manyseg_bench.reporting import load_runs, training_table

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "log.json").write_text(json.dumps([{"epoch": 0}]))
    (tmp_path / "n.json").write_text("3")
    (tmp_path / "p.json").write_text(json.dumps({"kind": "profile"}))
    runs = load_runs([tmp_path])
    assert runs == [{"kind": "profile"}]
    assert training_table(runs).count("\n") == 1  # header only, no crash


# --- B6: resolution_info looks through the ManySeg provider's wrappers ---------------------------


def test_resolution_info_unwraps_provider_wrappers():
    pytest.importorskip("manyseg")
    from manyseg_bench.experiments.runner import resolution_info
    from manyseg_bench.providers import get_provider
    from manyseg_bench.providers.manyseg_provider import WeightsInput, WeightsInputModel

    p = get_provider("manyseg")
    opts = {"head": {"scale": 0.5}}
    plain = p.build(ModelSpec("unet", "resnet18", weights=None, in_channels=4,
                              options=opts), img_size=(64, 64)).model
    expected = resolution_info(plain)
    assert expected is not None and expected["head_scales"] == {"logits": 0.5}
    bg = p.build(ModelSpec("unet", "resnet18", weights=None, in_channels=4, options={
        **opts, "band_groups": {"a": {"bands": [0, 1, 2]}, "b": {"bands": [3], "factor": 2}}}),
        img_size=(64, 64)).model
    assert type(bg).__name__ == "BandGroupModel"
    assert resolution_info(bg)["head_scales"] == {"logits": 0.5}
    wrapped = WeightsInputModel(plain, WeightsInput([0, 1, 2, 3], clip=False))
    assert resolution_info(wrapped) == expected


# --- B7: image-level runs record the effective arch "none" ----------------------------------------


def test_image_level_runs_record_arch_none_and_group_together(tmp_path, monkeypatch):
    pytest.importorskip("manyseg")
    from manyseg_bench import datasets
    from manyseg_bench.datasets.base import DatasetInfo, save_split
    from manyseg_bench.experiments import run_training
    from manyseg_bench.reporting.summary import group_key

    class Fake:
        INFO = DatasetInfo(name="fake_cls4", num_classes=3, in_channels=3,
                           class_names=("a", "b", "c"), mean=(0.0,) * 3, std=(1.0,) * 3)
        TASK = "classification"

    monkeypatch.setitem(datasets.DATASETS, "fake_cls4", Fake)
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("val", 4)):
        save_split(tmp_path / "cache" / "fake_cls4", split,
                   rng.normal(size=(n, 32, 32, 3)).astype(np.float16),
                   rng.integers(0, 3, n).astype(np.int64), [f"{split}{i}" for i in range(n)])
    recs = [run_training("manyseg", ModelSpec(arch, "resnet18", weights=None), "fake_cls4",
                         "smoke", device="cpu", data=tmp_path, out_dir=tmp_path / "runs",
                         recipe_overrides={"batch_size": 2, "precision": "fp32"})
            for arch in ("unet", "fpn")]
    assert [r["spec"]["arch"] for r in recs] == ["none", "none"]
    assert [r["requested_arch"] for r in recs] == ["unet", "fpn"]
    assert "_none_resnet18_" in Path(recs[0]["path"]).name
    assert group_key(recs[0]) == group_key(recs[1])
    assert recs[0]["params"] == recs[1]["params"]


def test_profile_image_output_records_arch_none(tmp_path):
    pytest.importorskip("timm")
    from manyseg_bench.cli import main

    assert main(["profile", "--provider", "timm", "--arch", "fpn", "--encoder", "resnet18",
                 "--weights", "none", "--device", "cpu", "--size", "32,32", "--batch-sizes",
                 "1", "--train-batch-size", "0", "--precision", "fp32",
                 "--out", str(tmp_path)]) == 0
    rec = json.loads(next(tmp_path.glob("*_profile_timm_none_*.json")).read_text())
    assert rec["spec"]["arch"] == "none" and rec["requested_arch"] == "fpn"
