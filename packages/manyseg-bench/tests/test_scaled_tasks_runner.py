"""run_training with scaled instance / point heads (regression: the runner replaced
spec.options["heads"] with task.heads(), silently discarding user scale/decoder overrides)."""

import numpy as np
import pytest
from manyseg_bench.datasets import oam_tcd
from manyseg_bench.datasets.base import save_split
from manyseg_bench.experiments import run_training
from manyseg_bench.providers import ModelSpec
from manyseg_bench.tasks import InstanceTask, PointTask


def instance_cache(root, name, n=6, size=48):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 40, (n, size, size, 3)).astype(np.uint8)
    masks = np.zeros((n, size, size), np.int32)
    for i in range(n):
        for k, (y, x) in enumerate([(4, 4), (4, 26), (26, 10), (30, 30)]):
            masks[i, y:y + 12, x:x + 12] = k + 1
            imgs[i, y:y + 12, x:x + 12] += 150
    for split in ("train", "val"):
        save_split(root / "cache" / name, split, imgs, masks, [str(i) for i in range(n)])


def run(tmp_path, dataset, **task_overrides):
    return run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), dataset,
                        "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                        recipe_overrides={"precision": "fp32", "batch_size": 2},
                        task_overrides=task_overrides)


def test_instance_task_head_scale_end_to_end(tmp_path):
    instance_cache(tmp_path, "ai4boundaries")
    rec = run(tmp_path, "ai4boundaries", head_scale=0.5,
              decoders={"boundary": {"type": "unet", "out_stride": 2}})
    heads = rec["spec"]["options"]["heads"]
    assert {k: h["scale"] for k, h in heads.items()} == {
        "extent": 0.5, "boundary": 0.5, "distance": 0.5}
    assert rec["resolution"]["head_scales"] == {"extent": 0.5, "boundary": 0.5,
                                                "distance": 0.5}
    assert rec["resolution"]["task_decoder_out_strides"] == {"boundary": 2}
    assert "pq" in rec["final_val"]


@pytest.mark.parametrize("kw", [{"offsets": True, "head_scale": 0.5},
                                {"density": True, "head_scale": 0.25}])
def test_point_task_head_scale_end_to_end(tmp_path, monkeypatch, kw):
    monkeypatch.setattr(oam_tcd, "LOADER_KWARGS", {"device_resident": False})
    instance_cache(tmp_path, "oam_tcd")
    rec = run(tmp_path, "oam_tcd", **kw)
    scales = set(rec["resolution"]["head_scales"].values())
    assert scales == {kw["head_scale"]}
    assert rec["primary_metric"] in rec["final_val"]


def test_user_heads_with_a_task_that_defines_heads_is_an_error(tmp_path):
    instance_cache(tmp_path, "ai4boundaries")
    spec = ModelSpec("unet", "resnet18", weights=None,
                     options={"heads": {"extent": {"type": "segmentation", "channels": 1,
                                                   "scale": 0.5}}})
    with pytest.raises(ValueError, match="head_scale"):
        run_training("manyseg", spec, "ai4boundaries", "smoke", device="cpu",
                     out_dir=tmp_path / "runs", data=tmp_path)


def test_task_head_options():
    assert {h["scale"] for h in InstanceTask(head_scale=0.25).heads().values()} == {0.25}
    assert all("scale" not in h for h in InstanceTask().heads().values())
    heads = PointTask(offsets=True, head_scale=0.5, decoders="separate").heads()
    assert {h["scale"] for h in heads.values()} == {0.5}
    assert all(h["decoder"] == "unet" for h in heads.values())
    with pytest.raises(ValueError, match="scale"):
        PointTask(head_scale=0.3)
