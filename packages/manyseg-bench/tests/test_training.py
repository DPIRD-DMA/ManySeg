import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader
from manyseg_bench.providers import ModelSpec, get_provider
from manyseg_bench.training import build_optimizer, get_recipe, lr_factor, train
from torch import nn


def synthetic(n=12, h=64, w=64, k=3, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, h, w), np.uint8)
    imgs = rng.integers(0, 60, (n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        y0, x0 = rng.integers(0, h // 2, 2)
        masks[i, y0:y0 + h // 3, x0:x0 + w // 3] = 1 + i % (k - 1)
        imgs[i][masks[i] > 0] += np.uint8(120)
    return imgs, masks


def loader(train, seed=0, bs=4):
    imgs, masks = synthetic()
    return GPULoader(imgs, masks, bs, (0.5,) * 3, (0.25,) * 3, torch.device("cpu"), train,
                     seed=seed)


def test_loader_deterministic_given_seed():
    a = [y for _, y in loader(True, seed=3)]
    b = [y for _, y in loader(True, seed=3)]
    c = [y for _, y in loader(True, seed=4)]
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not all(torch.equal(u, v) for u, v in zip(a, c))


def test_eval_loader_is_identity_and_ordered():
    imgs, masks = synthetic()
    got = torch.cat([y for _, y in loader(False)])
    assert torch.equal(got, torch.from_numpy(masks).long())


def test_augmentation_keeps_image_and_mask_aligned():
    """Bright pixels (object) must coincide with non-zero mask after augmentation."""
    for x, y in loader(True, seed=1):
        bright = x.mean(1) > 0.5 * ((120 / 255 - 0.5) / 0.25)
        agree = (bright == (y > 0)).float().mean()
        assert agree > 0.97


def test_lr_schedules():
    assert lr_factor(0, 100, 10, "cosine") == pytest.approx(0.1)
    assert lr_factor(10, 100, 10, "cosine") == pytest.approx(1.0)
    assert lr_factor(99, 100, 10, "cosine") < 0.01
    peak = max(lr_factor(s, 100, 0, "onecycle") for s in range(100))
    assert peak == pytest.approx(1.0, abs=1e-3)
    assert lr_factor(0, 100, 0, "onecycle") == pytest.approx(1 / 25)


def test_optimizer_groups_no_decay_on_norm_and_bias():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    enc = list(m[0].parameters())
    dec = list(m[1].parameters())
    opt = build_optimizer(enc, dec, get_recipe("fastai_like"))
    for g in opt.param_groups:
        if any(p.ndim <= 1 for p in g["params"]):
            assert g["weight_decay"] == 0
    enc_lrs = {g["base_lr"] for g in opt.param_groups if g["tag"] == "encoder"}
    assert enc_lrs == {1e-4}


@pytest.mark.parametrize("provider", ["manyseg", "smp", "fastai"])
def test_training_reduces_loss_on_synthetic(provider):
    torch.manual_seed(0)
    b = get_provider(provider).build(
        ModelSpec("unet", "resnet18", weights=None, num_classes=3), img_size=(64, 64))
    rec = get_recipe("standard", epochs=6, batch_size=4, lr=1e-3, precision="fp32",
                     warmup_epochs=0)
    res = train(b.model, b.encoder_params, b.decoder_params, loader(True), loader(False),
                num_classes=3, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
    losses = [h["train_loss"] for h in res["history"]]
    assert losses[-1] < losses[0]
    assert 0 <= res["final_val"]["miou"] <= 1


def test_freeze_encoder_epoch_leaves_encoder_conv_weights_unchanged():
    b = get_provider("manyseg").build(
        ModelSpec("unet", "resnet18", weights=None, num_classes=3), img_size=(64, 64))
    before = [p.detach().clone() for p in b.encoder_params if p.ndim > 1]
    rec = get_recipe("fastai_like", epochs=1, batch_size=4, precision="fp32")
    train(b.model, b.encoder_params, b.decoder_params, loader(True), loader(False),
          num_classes=3, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
    after = [p for p in b.encoder_params if p.ndim > 1]
    assert all(torch.equal(u, v) for u, v in zip(before, after))


def test_run_training_saves_checkpoint(tmp_path, monkeypatch):
    """End-to-end runner on a tiny prepared dataset; --save-model writes a loadable state_dict."""
    import json

    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import run_training

    imgs, masks = synthetic(n=8)
    cache = tmp_path / "cache" / "oxford_pet"
    for split in ("train", "val"):
        save_split(cache, split, imgs, (masks > 0).astype("uint8"), [str(i) for i in range(8)])
    rec = run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), "oxford_pet",
                       "smoke", device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                       recipe_overrides={"precision": "fp32"}, save_model=True)
    ck = rec["checkpoint"]["path"]
    sd = torch.load(ck, weights_only=True)
    assert any(k.startswith("backbone.") for k in sd)
    with open(rec["path"]) as f:
        saved = json.load(f)
    assert saved["checkpoint"]["sha256"] == rec["checkpoint"]["sha256"]


def test_runner_passes_label_smoothing_to_semantic_task(tmp_path, monkeypatch):
    """Regression (Codex review): runner-built SemanticTask must keep recipe.label_smoothing."""
    from manyseg_bench import tasks
    from manyseg_bench.datasets.base import save_split
    from manyseg_bench.experiments import run_training

    seen = {}
    orig = tasks.SemanticTask.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        seen["ls"] = self.label_smoothing

    monkeypatch.setattr(tasks.SemanticTask, "__init__", spy)
    imgs, masks = synthetic(n=8)
    cache = tmp_path / "cache" / "oxford_pet"
    for split in ("train", "val"):
        save_split(cache, split, imgs, masks, [str(i) for i in range(8)])
    run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), "oxford_pet", "smoke",
                 device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                 recipe_overrides={"precision": "fp32", "label_smoothing": 0.1})
    assert seen["ls"] == 0.1


def test_cpu_storage_yields_identical_batches():
    """device_resident=False (memmap-friendly) must give exactly the same batches."""
    imgs, masks = synthetic(n=10)
    kw = dict(batch_size=4, mean=(0.5,) * 3, std=(0.25,) * 3, device=torch.device("cpu"),
              seed=5)
    for is_train in (True, False):
        a = list(GPULoader(imgs, masks, train=is_train, **kw))
        b = list(GPULoader(imgs, masks, train=is_train, device_resident=False, **kw))
        assert len(a) == len(b)
        for (xa, ya), (xb, yb) in zip(a, b):
            assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_instance_id_masks_survive_augmentation():
    """Large integer ids (int32 masks) pass through nearest resampling exactly."""
    imgs, _ = synthetic(n=4)
    ids = np.zeros((4, 64, 64), np.int32)
    ids[:, 10:30, 10:30] = 70000
    ids[:, 40:60, 5:25] = 123
    for _, y in GPULoader(imgs, ids, 2, (0.5,) * 3, (0.25,) * 3, torch.device("cpu"), True):
        assert set(y.unique().tolist()) <= {0, 123, 70000}


@pytest.mark.parametrize("provider", ["manyseg", "smp"])
def test_freeze_encoder_bn_keeps_encoder_running_stats(provider):
    """E007: freeze_encoder_bn keeps the encoder's BN in eval mode (running statistics fixed;
    affine weights still train); decoder BN keeps updating."""
    pytest.importorskip("segmentation_models_pytorch") if provider == "smp" else None
    b = get_provider(provider).build(
        ModelSpec("unet", "resnet18", weights=None, num_classes=3), img_size=(64, 64))
    enc = {id(p) for p in b.encoder_params}
    bns = [m for m in b.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    enc_bn = [m for m in bns if id(m.weight) in enc]
    dec_bn = [m for m in bns if id(m.weight) not in enc]
    assert enc_bn and dec_bn
    before_enc = [m.running_mean.clone() for m in enc_bn]
    before_dec = [m.running_mean.clone() for m in dec_bn]
    before_w = [m.weight.detach().clone() for m in enc_bn]
    rec = get_recipe("standard", epochs=1, batch_size=4, precision="fp32",
                     freeze_encoder_bn=True)
    train(b.model, b.encoder_params, b.decoder_params, loader(True), loader(False),
          num_classes=3, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
    assert all(torch.equal(u, m.running_mean) for u, m in zip(before_enc, enc_bn))
    assert not all(torch.equal(u, m.running_mean) for u, m in zip(before_dec, dec_bn))
    assert not all(torch.equal(u, m.weight) for u, m in zip(before_w, enc_bn))
