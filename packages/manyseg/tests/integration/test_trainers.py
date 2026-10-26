"""ManySeg models train under the frameworks the examples document: fastai, Lightning,
Accelerate and TorchGeo.

Each test runs a short fit on CPU and checks the point where that framework could break the
recipe: dict outputs, bf16 autocast, checkpoint round trips, unwrapping before ``manyseg.save``,
TorchGeo's freezing and binary paths. The Lightning module and the TorchGeo task are loaded from
the example scripts themselves, so the tests cover the documented code.

A missing framework skips its tests, unless ``MANYSEG_REQUIRE_TRAINERS=1`` (set in CI), where
it fails: a skipped integration test there would hide a broken install.
"""

import importlib
import importlib.util
import os
from pathlib import Path

import manyseg
import pytest
import torch
import torch.nn.functional as F
from manyseg import create_model
from torch.utils.data import DataLoader, Dataset

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"


def _need(module: str):
    if os.environ.get("MANYSEG_REQUIRE_TRAINERS"):
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _example(name: str):
    """Import ``examples/<name>.py`` as a module (its ``main`` doesn't run)."""
    path = EXAMPLES / f"{name}.py"
    if not path.is_file():
        pytest.skip("not a repository checkout (no examples/)")
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Squares(Dataset):
    """Tiny batches of 3-band images with one bright square; items are dicts or tuples."""

    def __init__(self, n=8, size=32, as_dict=True, channels=3):
        g = torch.Generator().manual_seed(0)
        self.items = []
        for _ in range(n):
            mask = torch.zeros(size, size, dtype=torch.long)
            y, x = torch.randint(4, size - 4, (2,), generator=g).tolist()
            mask[y - 3:y + 4, x - 3:x + 4] = 1
            img = torch.rand(channels, size, size, generator=g) * 0.3 + mask
            self.items.append({"image": img, "mask": mask} if as_dict else (img, mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().float() for p in module.parameters()]).clone()


def _trainer(L, tmp_path, **kw):
    return L.Trainer(accelerator="cpu", devices=1, default_root_dir=tmp_path, logger=False,
                     enable_progress_bar=False, enable_model_summary=False, **kw)


# fastai


def test_fastai_dict_output_fit_and_get_preds(tmp_path):
    _need("fastai")
    import fastai.callback.schedule  # noqa: F401
    from fastai.data.core import DataLoaders
    from fastai.learner import Learner

    class Pairs(Squares):
        n_inp = 1

    model = create_model(backbone="resnet18", pretrained=False,
                         heads={"seg": {"type": "segmentation", "out_channels": 2},
                                "aux": {"type": "regression", "out_channels": 1}})
    dls = DataLoaders.from_dsets(Pairs(as_dict=False), Pairs(4, as_dict=False), bs=4,
                                 device="cpu")

    def loss(out, mask):
        return F.cross_entropy(out["seg"], mask) + out["aux"].pow(2).mean()

    learn = Learner(dls, model, loss_func=loss, path=tmp_path)
    before = _params(model)
    with learn.no_logging(), learn.no_bar():
        learn.fit_one_cycle(1, lr_max=1e-3)
    assert not torch.equal(before, _params(model))

    out, mask = learn.get_preds(reorder=False)
    assert set(out) == {"seg", "aux"}
    assert out["seg"].shape == (4, 2, 32, 32) and mask.shape == (4, 32, 32)
    path = manyseg.save(learn.model, tmp_path / "m.safetensors", normalize=[],
                        activations={"seg": "softmax"})
    assert set(manyseg.load(path)(torch.rand(1, 3, 32, 32))) == {"seg", "aux"}


# Lightning


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_lightning_fit_checkpoint_and_save(tmp_path, precision):
    L = _need("lightning")
    ex = _example("11_lightning")
    L.seed_everything(0, verbose=False)
    train = DataLoader(ex.Blobs(8, size=32), batch_size=4)
    trainer = _trainer(L, tmp_path, max_epochs=1, precision=precision)
    module = ex.SegModule()
    before = _params(module.model)
    trainer.fit(module, train, DataLoader(ex.Blobs(4, size=32, seed=1), batch_size=4))
    assert not torch.equal(before, _params(module.model))
    assert torch.isfinite(trainer.callback_metrics["train_loss"])

    # the checkpoint rebuilds the same model from the saved hyperparameters
    restored = ex.SegModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                                 map_location="cpu").eval()
    module.eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        for name, out in module(x).items():
            torch.testing.assert_close(restored(x)[name], out)
        path = manyseg.save(restored.model, tmp_path / "m.safetensors", normalize=[],
                            activations={"seg": "softmax"})
        torch.testing.assert_close(manyseg.load(path)(x)["seg"], module(x)["seg"])


# Accelerate


def test_accelerate_prepare_unwrap_and_save(tmp_path):
    accelerate = _need("accelerate")
    accelerator = accelerate.Accelerator(cpu=True)
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(Squares(as_dict=False), batch_size=4)
    valid = DataLoader(Squares(5, as_dict=False), batch_size=2)  # uneven last batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=len(loader))
    prepared, opt, sched, loader, valid = accelerator.prepare(model, opt, sched, loader, valid)
    before = _params(model)
    for x, y in loader:
        loss = F.cross_entropy(prepared(x), y)
        opt.zero_grad()
        accelerator.backward(loss)
        opt.step()
        sched.step()
    assert not torch.equal(before, _params(model))

    prepared.eval()
    with torch.no_grad():
        preds = [accelerator.gather_for_metrics((prepared(x).argmax(1), y)) for x, y in valid]
    assert sum(len(p) for p, _ in preds) == 5
    assert all(p.shape == y.shape == (len(y), 32, 32) for p, y in preds)

    unwrapped = accelerator.unwrap_model(prepared)
    assert isinstance(unwrapped, manyseg.DenseModel)
    path = manyseg.save(unwrapped, tmp_path / "m.safetensors", normalize=[],
                        activations={"logits": "softmax"})
    x = torch.rand(1, 3, 32, 32)
    unwrapped.eval()
    with torch.no_grad():
        torch.testing.assert_close(manyseg.load(path)(x), unwrapped(x))


# TorchGeo


def _torchgeo_task(**kw):
    _need("torchgeo")
    kw = {"backbone": "resnet18", "weights": False, "in_channels": 4, "lr": 1e-3, **kw}
    return _need("lightning"), _example("13_torchgeo").ManySegSegmentation(**kw)


@pytest.mark.parametrize("task_kw", [
    {"task": "multiclass", "num_classes": 2, "loss": "ce"},
    {"task": "multiclass", "num_classes": 2, "loss": "dice", "model": "fpn"},
    {"task": "binary", "loss": "bce"},  # one output channel, squeezed by TorchGeo
], ids=["multiclass-ce", "multiclass-dice-fpn", "binary-bce"])
def test_torchgeo_task_fit_and_predict(tmp_path, task_kw):
    L, task = _torchgeo_task(**task_kw)
    assert isinstance(task.model, manyseg.DenseModel)
    data = DataLoader(Squares(channels=4), batch_size=4)
    before = _params(task.model)
    trainer = _trainer(L, tmp_path, max_epochs=1)
    trainer.fit(task, data, data)
    assert not torch.equal(before, _params(task.model))
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
    probs = trainer.predict(task, data)[0]["probabilities"]
    assert probs.shape == (4, task_kw.get("num_classes", 1), 32, 32)


def test_torchgeo_freeze_backbone(tmp_path):
    L, task = _torchgeo_task(num_classes=2, freeze_backbone=True)
    backbone, decoder = _params(task.model.backbone), _params(task.model.decoder)
    data = DataLoader(Squares(channels=4), batch_size=4)
    _trainer(L, tmp_path, max_epochs=1).fit(task, data, data)  # its scheduler needs val_loss
    assert not any(p.requires_grad for p in task.model.backbone.parameters())
    torch.testing.assert_close(_params(task.model.backbone), backbone)
    assert not torch.equal(_params(task.model.decoder), decoder)


def test_torchgeo_refuses_smp_weights(tmp_path):
    ResNet18_Weights = _need("torchgeo.models").ResNet18_Weights
    ckpt = tmp_path / "smp.ckpt"
    ckpt.touch()
    for weights in (ResNet18_Weights.SENTINEL2_ALL_MOCO, str(ckpt)):
        with pytest.raises(ValueError, match="manyseg.weights tag"):
            _torchgeo_task(num_classes=2, weights=weights)
