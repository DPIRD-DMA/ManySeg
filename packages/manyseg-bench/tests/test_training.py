import numpy as np
import pytest
import torch

from manyseg_bench.datasets import GPULoader
from manyseg_bench.providers import ModelSpec, get_provider
from manyseg_bench.training import build_optimizer, get_recipe, lr_factor, train


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
