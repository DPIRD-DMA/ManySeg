"""Re-evaluating a saved run under precision / int8 variants (E008)."""

import json
import shutil
import sys

import numpy as np
import pytest
import torch
from manyseg_bench.providers import ModelSpec


def _synthetic(n=8, h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, h, w), np.uint8)
    imgs = rng.integers(0, 60, (n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        y0, x0 = rng.integers(0, h // 2, 2)
        masks[i, y0:y0 + h // 3, x0:x0 + w // 3] = 1
        imgs[i][masks[i] > 0] += np.uint8(120)
    return imgs, masks


@pytest.fixture
def saved_run(tmp_path):
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import run_training

    imgs, masks = _synthetic()
    cache = tmp_path / "cache" / "oxford_pet"
    for split in ("train", "val", "test"):
        save_split(cache, split, imgs, masks, [str(i) for i in range(len(imgs))])
    rec = run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), "oxford_pet",
                       "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                       evaluate_test=True, recipe_overrides={"precision": "fp32"},
                       save_model=True)
    return rec, tmp_path


def test_fp32_variant_reproduces_the_training_record(saved_run):
    from manyseg_bench.experiments import evaluate_run

    rec, root = saved_run
    ev = evaluate_run(rec["path"], variants=("fp32",), device="cpu", data=root,
                      out_dir=root / "eval")
    v = ev["variants"]["fp32"]
    assert v["test"]["miou"] == pytest.approx(rec["test"]["miou"], abs=1e-6)
    assert v["val"]["miou"] == pytest.approx(rec["final_val"]["miou"], abs=1e-6)
    assert v["test"]["agreement"] == 1.0  # the reference agrees with itself
    assert ev["source_run"] == rec["run_id"] and ev["kind"] == "evaluate"
    assert json.loads(next((root / "eval").glob("*.json")).read_text())["variants"]


def test_int8_variants_run_on_cpu_and_report_agreement(saved_run):
    pytest.importorskip("torchao")
    from manyseg_bench.experiments import evaluate_run

    rec, root = saved_run
    ev = evaluate_run(rec["path"], variants=("int8_weight_only", "int8_pt2e"), device="cpu",
                      data=root, out_dir=root / "eval", calib_batches=2)
    for name in ("int8_weight_only", "int8_pt2e"):
        t = ev["variants"][name]["test"]
        assert 0.0 <= t["miou"] <= 1.0 and 0.5 < t["agreement"] <= 1.0
        assert ev["variants"][name]["device"] == "cpu"
        assert ev["variants"][name]["size_bytes"] > 0


def test_checkpoint_hash_is_verified(saved_run):
    from manyseg_bench.experiments import evaluate_run

    rec, root = saved_run
    ck = rec["checkpoint"]["path"]
    sd = torch.load(ck, weights_only=True)
    next(iter(sd.values())).add_(1.0)
    torch.save(sd, ck)
    with pytest.raises(ValueError, match="sha256"):
        evaluate_run(rec["path"], variants=("fp32",), device="cpu", data=root,
                     out_dir=root / "eval")


def test_unknown_variant_is_rejected(saved_run):
    from manyseg_bench.experiments import evaluate_run

    rec, root = saved_run
    with pytest.raises(ValueError, match="variant"):
        evaluate_run(rec["path"], variants=("int4",), device="cpu", data=root,
                     out_dir=root / "eval")


def test_cli_evaluate(saved_run, capsys):
    from manyseg_bench.cli import main

    rec, root = saved_run
    assert main(["--data", str(root), "evaluate", rec["path"], "--variants", "fp32",
                 "--device", "cpu", "--out", str(root / "eval")]) == 0
    assert "fp32" in capsys.readouterr().out


def test_int8_pt2e_handles_a_smaller_final_batch(saved_run, monkeypatch):
    """Regression: calibration batches fixed the exported batch size, so the last (smaller)
    eval batch failed the export guard."""
    pytest.importorskip("torchao")
    from manyseg_bench.datasets import oxford_pet
    from manyseg_bench.experiments import evaluate_run

    rec, root = saved_run
    monkeypatch.setattr(oxford_pet, "EVAL_BATCH_SIZE", 3, raising=False)  # 8 = 3 + 3 + 2
    ev = evaluate_run(rec["path"], variants=("int8_pt2e",), device="cpu", data=root,
                      out_dir=root / "eval", calib_batches=2)
    assert 0.5 < ev["variants"]["int8_pt2e"]["test"]["agreement"] <= 1.0


@pytest.mark.skipif(sys.platform == "win32" and shutil.which("cl") is None,
                    reason="inductor's CPU backend needs MSVC (cl) on Windows")
def test_compiled_int8_with_uneven_batches_and_failures_are_recorded(saved_run, monkeypatch):
    """Regression: torch.compile + inductor freezing on a dynamic-batch PT2E program crashed in
    dynamo, and one failing variant discarded every other variant's results."""
    pytest.importorskip("torchao")
    from manyseg_bench.datasets import oxford_pet
    from manyseg_bench.experiments import evaluate, evaluate_run

    rec, root = saved_run
    monkeypatch.setattr(oxford_pet, "EVAL_BATCH_SIZE", 3, raising=False)
    ev = evaluate_run(rec["path"], variants=("int8_pt2e_compiled",), device="cpu", data=root,
                      out_dir=root / "eval", calib_batches=2)
    res = ev["variants"]["int8_pt2e_compiled"]
    assert "test" in res, res.get("error")
    assert 0.5 < res["test"]["agreement"] <= 1.0

    def boom(*a, **k):
        raise RuntimeError("quantiser exploded")

    monkeypatch.setattr(evaluate, "_quantised", boom)
    ev = evaluate_run(rec["path"], variants=("int8_weight_only", "fp32"), device="cpu",
                      data=root, out_dir=root / "eval")
    assert "quantiser exploded" in ev["variants"]["int8_weight_only"]["error"]
    assert ev["variants"]["fp32"]["test"]["agreement"] == 1.0


def test_pad_batch_and_agreement_helpers():
    """Codex review: compiled int8 must handle dict (multi-input) batches, and agreement is
    only defined for single full-resolution logits."""
    from manyseg_bench.experiments.evaluate import _agreement, _pad_batch

    d = _pad_batch({"a": torch.ones(2, 3, 4, 4), "b": torch.zeros(2, 1, 4, 4)}, 5)
    assert d["a"].shape[0] == 5 and d["b"].shape[0] == 5
    y = torch.zeros(2, 4, 4, dtype=torch.long)
    y[0, 0, 0] = 255
    logits = torch.randn(2, 3, 4, 4)
    assert _agreement(logits, logits, y, 255) == (31, 31)
    assert _agreement(logits[..., :2, :2], logits[..., :2, :2], y, 255) is None  # scaled head
    assert _agreement({"h": logits}, {"h": logits}, y, 255) is None  # multi-head task
