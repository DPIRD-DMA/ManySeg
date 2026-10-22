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


def test_grad_accumulation_steps_and_equivalence(monkeypatch):
    """E007b: grad_accum_steps=k takes one optimiser step per k micro-batches (the lr schedule
    counts optimiser steps), and k micro-batches of b match one batch of k*b (no BN)."""
    from manyseg_bench.training import trainer

    imgs, masks = synthetic(n=8)

    def fixed_loader(bs):  # no augmentation: identical images in both runs
        return GPULoader(imgs, masks, bs, (0.5,) * 3, (0.25,) * 3, torch.device("cpu"), False)

    def build():
        torch.manual_seed(0)
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1))

    steps = []
    orig = torch.optim.AdamW.step

    def counting(self, *a, **k):
        steps.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting)
    results = []
    for bs, k in ((4, 1), (2, 2)):
        m = build()
        params = list(m.parameters())
        rec = get_recipe("standard", epochs=1, batch_size=bs, precision="fp32", warmup_epochs=0,
                         schedule="constant", grad_accum_steps=k)
        steps.clear()
        trainer.train(m, params[:2], params[2:], fixed_loader(bs), fixed_loader(4),
                      num_classes=3, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
        results.append(([p.detach().clone() for p in m.parameters()], len(steps)))
    (pa, na), (pb, nb) = results
    assert na == nb == 2  # 8 images: 2 steps of 4, or 4 micro-batches of 2 accumulated in pairs
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(pa, pb))


def test_grad_accumulation_weights_micro_batches_by_size(monkeypatch):
    """Codex review: a short final micro-batch must not count as much as a full one. 6 images,
    batch 4 + accumulate 2 (micro-batches of 4 and 2) must equal one batch of 6. (Training
    loaders drop the last partial batch, so this only arises with drop_last=False.) Plain SGD,
    because AdamW's first step normalises gradient magnitude and would hide the difference."""
    from manyseg_bench.training import trainer

    monkeypatch.setattr(trainer, "build_optimizer", lambda enc, dec, rec: torch.optim.SGD(
        [{"params": enc + dec, "lr": 0.5, "base_lr": 0.5}], lr=0.5))

    imgs, masks = synthetic(n=6)

    def fixed_loader(bs):
        return GPULoader(imgs, masks, bs, (0.5,) * 3, (0.25,) * 3, torch.device("cpu"), False)

    def build():
        torch.manual_seed(0)
        return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1))

    got = []
    for bs, k in ((6, 1), (4, 2)):
        m = build()
        params = list(m.parameters())
        rec = get_recipe("standard", epochs=1, batch_size=bs, precision="fp32", warmup_epochs=0,
                         schedule="constant", grad_accum_steps=k)
        trainer.train(m, params[:2], params[2:], fixed_loader(bs), fixed_loader(6),
                      num_classes=3, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
        got.append([p.detach().clone() for p in m.parameters()])
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(*got))


def test_freeze_encoder_bn_includes_non_affine_bn():
    """Codex review: an encoder BN with affine=False has no weight to match on."""
    from manyseg_bench.training import trainer

    enc = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4, affine=False))
    dec = nn.Conv2d(4, 3, 1)
    model = nn.Sequential(enc, dec)
    before = enc[1].running_mean.clone()
    rec = get_recipe("standard", epochs=1, batch_size=4, precision="fp32",
                     freeze_encoder_bn=True)
    trainer.train(model, list(enc.parameters()), list(dec.parameters()), loader(True),
                  loader(False), num_classes=3, recipe=rec, device=torch.device("cpu"),
                  log=lambda *_: None)
    assert torch.equal(before, enc[1].running_mean)


def test_reported_train_loss_weights_batches_by_image_count(monkeypatch):
    """Codex review: with a short last batch, the reported epoch loss must be the per-image
    mean, as the optimisation is (lr 0 so the model is fixed during the epoch)."""
    from manyseg_bench.training import trainer

    imgs, masks = synthetic(n=6)
    ld = GPULoader(imgs, masks, 4, (0.5,) * 3, (0.25,) * 3, torch.device("cpu"), False)
    monkeypatch.setattr(trainer, "build_optimizer", lambda enc, dec, rec: torch.optim.SGD(
        [{"params": enc + dec, "lr": 0.0, "base_lr": 0.0}], lr=0.0))
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1))
    params = list(m.parameters())
    rec = get_recipe("standard", epochs=1, batch_size=4, precision="fp32", warmup_epochs=0,
                     schedule="constant")
    out = trainer.train(m, params[:2], params[2:], ld, ld, num_classes=3, recipe=rec,
                        device=torch.device("cpu"), log=lambda *_: None)
    with torch.no_grad():
        per_image = [float(nn.functional.cross_entropy(m(x[i:i + 1]), y[i:i + 1].long()))
                     for x, y in ld for i in range(x.shape[0])]
    assert out["history"][0]["train_loss"] == pytest.approx(sum(per_image) / 6, rel=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_side_nonfinite_skip_equals_dropping_the_batch():
    """E026: on CUDA (fused AdamW, one micro-batch per step) a non-finite loss skips the update
    on the device, without a host sync. It must equal training without that batch."""
    dev = torch.device("cuda")
    imgs, masks = synthetic(n=12)
    batches = list(GPULoader(imgs, masks, 4, (0.5,) * 3, (0.25,) * 3, dev, False))
    bad = (batches[1][0].clone(), batches[1][1])
    bad[0][0, 0, 0, 0] = float("nan")
    got = []
    for data in ([batches[0], bad, batches[2]], [batches[0], batches[2]]):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1))
        params = list(m.parameters())
        rec = get_recipe("standard", epochs=1, batch_size=4, precision="fp32", warmup_epochs=0,
                         schedule="constant")
        out = train(m, params[:2], params[2:], data, batches[:1], num_classes=3, recipe=rec,
                    device=dev, log=lambda *_: None)
        got.append(([p.detach().clone() for p in m.parameters()], out["history"][0]))
    (pa, ha), (pb, hb) = got
    assert ha["nonfinite_steps"] == 1 and hb["nonfinite_steps"] == 0
    assert ha["train_loss"] == pytest.approx(hb["train_loss"], rel=1e-6)
    # a failed skip would leave NaN weights; cuDNN algorithm choice (benchmark mode, set by
    # other tests) allows rounding-level differences between the two runs
    assert all(torch.isfinite(u).all() for u in pa)
    assert all(torch.allclose(u, v, rtol=1e-5, atol=1e-6) for u, v in zip(pa, pb))


def test_evaluate_pad_last_gives_the_same_metrics():
    """E027: padding the short last batch (for a static compiled model) must not change the
    metrics: the padded rows are dropped before the evaluator sees them."""
    from manyseg_bench.training.trainer import evaluate

    imgs, masks = synthetic(n=10)
    ld = GPULoader(imgs, masks, 4, (0.5,) * 3, (0.25,) * 3, torch.device("cpu"), False)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1)).eval()
    seen = []
    m.register_forward_hook(lambda _m, i, _o: seen.append(i[0].shape[0]) and None)
    a = evaluate(m, ld, 3, torch.device("cpu"), "fp32")
    b = evaluate(m, ld, 3, torch.device("cpu"), "fp32", pad_last=True)
    assert seen == [4, 4, 2, 4, 4, 4]  # the padded pass runs the short batch at full size
    assert a == pytest.approx(b)
