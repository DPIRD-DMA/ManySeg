"""Reduced-resolution models in the benchmark: scaled heads, decoder out_stride, profiling."""

import pytest
import torch
import torch.nn.functional as F
from manyseg.utils import downsample_labels, upsample_output
from manyseg_bench.providers import ModelSpec, get_provider
from manyseg_bench.tasks import InstanceTask, PointTask, SemanticTask
from manyseg_bench.training import get_recipe, train
from test_training import loader, synthetic


def build(options, num_classes=3):
    return get_provider("manyseg").build(
        ModelSpec("unet", "resnet18", weights=None, num_classes=num_classes, options=options),
        img_size=(64, 64))


def test_provider_head_option_scales_the_logits_head():
    b = build({"head": {"scale": 0.25}, "decoder_kwargs": {"out_stride": 4}})
    out = b.model.eval()(torch.randn(1, 3, 65, 97))
    assert isinstance(out, torch.Tensor) and out.shape == (1, 3, 17, 25)


def test_provider_head_option_conflicts_with_heads():
    with pytest.raises(ValueError, match="head"):
        build({"head": {"scale": 0.5}, "heads": {"y": {"type": "segmentation", "channels": 2}}})


def test_semantic_loss_at_head_scale_uses_mode_downsampled_labels():
    task = SemanticTask(3, ignore_index=255)
    y = torch.randint(0, 3, (2, 33, 47))
    y[:, :5] = 255
    logits = torch.randn(2, 3, 9, 12)
    ref = F.cross_entropy(logits, downsample_labels(y, 0.25, 255), ignore_index=255)
    torch.testing.assert_close(task.loss(logits, y), ref)


def test_semantic_eval_upsamples_to_full_resolution():
    """Perfect block-level logits: full-resolution metrics see the block quantisation, the
    optional native-scale metrics do not."""
    task = SemanticTask(2, native_metrics=True)
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[0, 5:21, 9:27] = 1          # edges not on the 4-pixel grid
    small = downsample_labels(y, 0.25)
    logits = F.one_hot(small, 2).permute(0, 3, 1, 2).float() * 10
    ev = task.evaluator()
    ev.update(logits, y)
    r = ev.compute()
    assert r["native"]["miou"] == pytest.approx(1.0)
    full = upsample_output(logits, (32, 32), 0.25).argmax(1)
    inter = ((full == 1) & (y == 1)).sum() / ((full == 1) | (y == 1)).sum()
    assert r["iou"][1] == pytest.approx(float(inter))
    assert r["head_factor"] == 4 and r["iou"][1] < 1.0


def test_semantic_eval_full_resolution_unchanged():
    ev = SemanticTask(2).evaluator()
    y = torch.randint(0, 2, (1, 16, 16))
    ev.update(F.one_hot(y, 2).permute(0, 3, 1, 2).float(), y)
    r = ev.compute()
    assert r["miou"] == 1.0 and r["head_factor"] == 1 and "native" not in r


def test_head_factor_rejects_unrelated_sizes():
    task = SemanticTask(2)
    with pytest.raises(ValueError, match="grid"):
        task.loss(torch.randn(1, 2, 10, 10), torch.zeros(1, 32, 32, dtype=torch.long))


@pytest.mark.parametrize("options", [
    {"head": {"scale": 0.5}}, {"decoder_kwargs": {"out_stride": 2}},
    {"head": {"scale": 0.25}, "decoder_kwargs": {"out_stride": 4}}])
def test_training_with_reduced_resolution(options):
    torch.manual_seed(0)
    b = build(options)
    rec = get_recipe("standard", epochs=3, batch_size=4, lr=1e-3, precision="fp32",
                     warmup_epochs=0)
    res = train(b.model, b.encoder_params, b.decoder_params, loader(True), loader(False),
                num_classes=3, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
    losses = [h["train_loss"] for h in res["history"]]
    assert losses[-1] < losses[0]
    assert res["final_val"]["head_factor"] == round(1 / options.get("head", {}).get("scale", 1))
    assert res["train_images_per_s"] > 0


@pytest.mark.parametrize("task", [InstanceTask(), PointTask(offsets=True), PointTask(density=True)])
def test_instance_and_point_tasks_accept_scaled_heads(task):
    """Instance/point tasks compare at input resolution: scaled outputs are upsampled."""
    from test_tasks import loaders

    heads = {k: {**v, "scale": 0.5} for k, v in task.heads().items()}
    torch.manual_seed(0)
    b = get_provider("manyseg").build(ModelSpec("unet", "resnet18", weights=None, num_classes=1,
                                                options={"heads": heads}))
    rec = get_recipe("standard", epochs=1, batch_size=4, lr=1e-3, precision="fp32",
                     warmup_epochs=0)
    tr, va = loaders()
    res = train(b.model, b.encoder_params, b.decoder_params, tr, va, num_classes=1, recipe=rec,
                device=torch.device("cpu"), task=task, log=lambda *_: None)
    assert task.primary_metric in res["final_val"]


def test_profile_handles_dict_outputs():
    from manyseg_bench.profiling import profile_model

    b = build({"heads": {"a": {"type": "segmentation", "channels": 2, "scale": 0.5},
                         "b": {"type": "regression", "channels": 1}}})
    r = profile_model(b.model, in_channels=3, img_size=(64, 64), device="cpu", batch_sizes=(1,),
                      precision="fp32", warmup=1, iters=2, train_batch_size=2)
    assert r["train"]["images_per_s"] > 0 and r["flops_per_image"] > 0


def test_runner_records_flops_and_resolution(tmp_path):
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import run_training

    imgs, masks = synthetic(n=8)
    cache = tmp_path / "cache" / "oxford_pet"
    for split in ("train", "val"):
        save_split(cache, split, imgs, (masks > 0).astype("uint8"), [str(i) for i in range(8)])
    spec = ModelSpec("unet", "resnet18", weights=None,
                     options={"decoder_kwargs": {"out_stride": 2}, "head": {"scale": 0.5}})
    rec = run_training("manyseg", spec, "oxford_pet", "smoke", device="cpu",
                       out_dir=tmp_path / "runs", data=tmp_path,
                       recipe_overrides={"precision": "fp32"})
    assert rec["flops_per_image"] > 0 and rec["params"] > 0
    assert rec["resolution"] == {"decoder_out_stride": 2, "head_scales": {"logits": 0.5},
                                 "eval_resolution": "input"}
    assert rec["train_images_per_s"] > 0
    assert "peak_train_mem_mib" in rec


def test_profile_cli_records_resolution_and_training_memory(tmp_path):
    import json

    from manyseg_bench.cli import main

    main(["profile", "--provider", "manyseg", "--arch", "unet", "--encoder", "resnet18",
          "--weights", "none", "--device", "cpu", "--size", "64,64", "--batch-sizes", "1",
          "--precision", "fp32", "--train-batch-size", "2", "--out", str(tmp_path),
          "--options", '{"decoder_kwargs": {"out_stride": 2}, "head": {"scale": 0.5}}'])
    (path,) = tmp_path.glob("*_profile_*.json")
    rec = json.loads(path.read_text())
    assert rec["resolution"]["decoder_out_stride"] == 2
    assert rec["resolution"]["head_scales"] == {"logits": 0.5}
    assert rec["train"]["batch_size"] == 2 and "peak_mem_mib" in rec["train"]
    assert "gpu_contention_at_end" in rec
