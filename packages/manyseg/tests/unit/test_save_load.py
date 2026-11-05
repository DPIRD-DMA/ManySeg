"""manyseg.save / manyseg.load: rebuild from the stored create_model arguments, identical
outputs, metadata for inference (normalisation, activations, classes, bands)."""

import json
import random

import manyseg
import numpy as np
import pytest
import torch
from manyseg import create_model
from manyseg.io import _weights_normalize
from safetensors import safe_open


def outputs(model, x):
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out if isinstance(out, dict) else {"": out}


def activations(model):
    """softmax for every head that needs one stated (the tests' default)."""
    return {n: "softmax" for n, h in model.heads.items() if h.kind != "regression"}


def normalize_for(model):
    """The tests' default spec, per raster input for a model built with inputs=."""
    step = [{"type": "scale", "value": 10000}]
    inputs = model.build_config.get("inputs")
    if not inputs:
        return step
    return {n: step for n, c in inputs.items() if c.get("kind") != "vector"}


def roundtrip(model, x, tmp_path, **kw):
    kw.setdefault("activations", activations(model))
    kw.setdefault("normalize", normalize_for(model))
    path = manyseg.save(model.eval(), tmp_path / "m.safetensors", **kw)
    loaded = manyseg.load(path)
    a, b = outputs(model, x), outputs(loaded, x)
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), name
    return loaded


SAR = {"in_channels": 2, "backbone": "resnet18", "alignment": "same_extent"}
FIELDS = {"doy": {"type": "cyclic", "period": 365.25}}

CASES = {
    "single": (dict(in_channels=4, num_classes=3), lambda: torch.randn(2, 4, 64, 64)),
    "heads": (dict(in_channels=3, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "height": {"type": "regression", "out_channels": 1, "scale": 0.25, "uncertainty": True},
        "scene": {"type": "classification", "out_channels": 5, "scope": "image"}}),
        lambda: torch.randn(2, 3, 64, 64)),
    "two_inputs": (dict(inputs={"s2": {"in_channels": 4, "backbone": "resnet18"}, "s1": SAR},
                        fusion={"type": "intermediate", "op": "gated"}, num_classes=2),
                   lambda: {"s2": torch.randn(2, 4, 64, 64), "s1": torch.randn(2, 2, 32, 32)}),
    "shared_backbone": (dict(inputs={"t1": {"in_channels": 3, "backbone": "resnet18"},
                                     "t2": {"in_channels": 3, "backbone": "shared:t1"}},
                             fusion={"type": "intermediate", "op": "concat"}, num_classes=2),
                        lambda: {"t1": torch.randn(2, 3, 64, 64),
                                 "t2": torch.randn(2, 3, 64, 64)}),
    # shared modules appear under several names (found by the config fuzzer: safetensors'
    # save_model / load_model chose different names for shared BatchNorm buffers)
    "shared_heads": (dict(inputs={"t1": {"in_channels": 3}, "t2": {"in_channels": 3}},
                          heads={"c": {"type": "segmentation", "out_channels": 2},
                                 "s1": {"type": "segmentation", "out_channels": 4, "input": "t1"},
                                 "s2": {"type": "segmentation", "out_channels": 4, "input": "t2",
                                        "share": "s1"}}, decoder="unetplusplus"),
                     lambda: {"t1": torch.randn(2, 3, 64, 64), "t2": torch.randn(2, 3, 64, 64)}),
    "shared_norms": (dict(inputs={"a": {"in_channels": 1, "backbone": "efficientnet_b0"},
                                  "b": {"in_channels": 2, "backbone": "shared:a",
                                        "share_norm": True, "channel_adapter": "projection"}},
                          num_classes=2, decoder="unetplusplus"),
                     lambda: {"a": torch.randn(2, 1, 64, 64), "b": torch.randn(2, 2, 64, 64)}),
    "time_series": (dict(inputs={"ts": {"kind": "timeseries", "in_channels": 3,
                                        "temporal": "attention"}}, num_classes=2),
                    lambda: {"ts": torch.randn(2, 4, 3, 64, 64)}),
    "metadata": (dict(inputs={"img": {"in_channels": 3}, "meta": {"kind": "vector",
                                                                "fields": FIELDS}},
                      num_classes=2),
                 lambda: {"img": torch.randn(2, 3, 64, 64), "meta": torch.rand(2, 1) * 365}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_roundtrip_gives_identical_outputs(case, tmp_path):
    kwargs, make_input = CASES[case]
    torch.manual_seed(0)
    model = create_model(backbone="resnet18", pretrained=False, **kwargs)
    roundtrip(model, make_input(), tmp_path)


def test_selected_heads_survive(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "aux": {"type": "regression", "out_channels": 1}})
    model.select_heads(["seg"])
    loaded = roundtrip(model, torch.randn(1, 3, 64, 64), tmp_path)
    assert list(loaded.heads) == ["seg"]


def test_metadata_and_tiling_spec(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "height": {"type": "regression", "out_channels": 1}})
    norm = [{"type": "scale", "value": 10000}, {"type": "zscore", "scope": "tile"}]
    path = manyseg.save(model, tmp_path / "m.safetensors", normalize=norm,
                        activations={"seg": "softmax"}, classes={"seg": ["a", "b", "c"]},
                        bands=["B04", "B03", "B02"], extra={"trained_on": "test"})
    meta = manyseg.read_metadata(path)
    assert meta["normalize"] == norm and meta["classes"] == {"seg": ["a", "b", "c"]}
    assert meta["activations"] == {"seg": "softmax", "height": "none"}
    assert meta["create_model"]["pretrained"] is False and meta["extra"]["trained_on"] == "test"
    spec = manyseg.load(path).tiling_spec()
    assert spec["normalize"] == norm
    assert spec["outputs"]["seg"]["activation"] == "softmax"
    assert spec["outputs"]["height"]["activation"] == "none"
    json.dumps(spec)


def test_pretrained_flags_are_not_used_on_load(tmp_path, monkeypatch):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2)
    model.build_config["pretrained"] = True  # as if built with downloaded weights
    path = manyseg.save(model, tmp_path / "m.safetensors", activations=activations(model),
                        normalize=[])
    with safe_open(str(path), framework="pt") as f:
        assert json.loads(f.metadata()["manyseg"])["create_model"]["pretrained"] is True
    import timm

    created = []
    original = timm.create_model
    monkeypatch.setattr(timm, "create_model",
                        lambda *a, **k: created.append(k.get("pretrained")) or original(*a, **k))
    manyseg.load(path)
    assert created and not any(created)


def test_weights_normalisation_on_request():
    """SSL4EO-S12 S2 weights: DN / 10000 clipped to [0, 1] (mean 0, std 10000, clip)."""
    steps = _weights_normalize({"backbone": "resnet50", "pretrained": "ssl4eo_s2_moco"})
    assert steps == [{"type": "zscore", "mean": [0.0] * 13, "std": [10000.0] * 13},
                     {"type": "clip", "min": 0.0, "max": 1.0}]
    from manyseg.weights import WEIGHTS

    for w in WEIGHTS.values():  # every entry maps to (x - mean) / std, then clip if flagged
        got = _weights_normalize({"backbone": w.model, "pretrained": w.tag})
        assert got[0] == {"type": "zscore", "mean": list(w.mean), "std": list(w.std)}
        assert (len(got) == 2) == w.clip
    assert _weights_normalize({"backbone": "resnet18", "pretrained": True}) is None


def test_no_silent_defaults(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "height": {"type": "regression", "out_channels": 1}})
    with pytest.raises(ValueError, match="'seg'"):  # softmax or sigmoid must be stated
        manyseg.save(model, tmp_path / "a.safetensors", normalize=[])
    with pytest.warns(UserWarning, match="No normalisation"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"seg": "softmax"})
    with pytest.raises(ValueError, match="weights"):  # not built with a weights tag
        manyseg.save(model, tmp_path / "a.safetensors", activations={"seg": "softmax"},
                     normalize="weights")


def test_load_honours_device_and_dtype(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2)
    model = model.to(torch.float64)
    path = manyseg.save(model, tmp_path / "m.safetensors", activations={"logits": "softmax"},
                        normalize=[])
    loaded = manyseg.load(path, map_location="cpu")
    assert next(loaded.parameters()).dtype == torch.float64
    if torch.cuda.is_available():
        assert next(manyseg.load(path, map_location="cuda").parameters()).is_cuda


def test_models_changed_after_creation_fail_at_save(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2)
    model.extra = torch.nn.Linear(2, 2)  # added by training code
    with pytest.raises(ValueError, match="differs from what its create_model"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"logits": "softmax"},
                     normalize=[])
    with pytest.raises(TypeError, match="DenseModel"):
        manyseg.save(torch.nn.DataParallel(model), tmp_path / "a.safetensors")


def test_save_rejects_replaced_modules_without_weights(tmp_path):
    """Swapping a parameterless module leaves the state dict alone but changes predictions."""
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2)
    name = next(n for n, m in model.named_modules() if isinstance(m, torch.nn.ReLU))
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent), child, torch.nn.Sigmoid())
    with pytest.raises(ValueError, match="differs from what its create_model"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"logits": "softmax"},
                     normalize=[])


def test_errors(tmp_path):
    from manyseg.models import DenseModel

    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2)
    kw = {"normalize": []}
    with pytest.raises(ValueError, match="unknown head"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"nope": "softmax"}, **kw)
    with pytest.raises(ValueError, match="one of"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"logits": "relu"}, **kw)
    with pytest.raises(ValueError, match="plain JSON"):
        manyseg.save(model, tmp_path / "a.safetensors", extra={"x": object()},
                     activations={"logits": "softmax"}, **kw)
    bare = DenseModel(model.backbone, model.decoder, dict(model.heads))
    with pytest.raises(ValueError, match="build_config"):
        manyseg.save(bare, tmp_path / "a.safetensors", **kw)
    from safetensors.torch import save_file

    save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"))
    with pytest.raises(ValueError, match="not written by manyseg.save"):
        manyseg.load(tmp_path / "other.safetensors")
    for meta in ({"format": "1"}, {"format": 99, "create_model": {}, "heads": [],
                                   "activations": {}}, {"format": 1}):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "bad.safetensors"),
                  metadata={"manyseg": json.dumps(meta)})
        with pytest.raises(ValueError, match="format|lacks"):
            manyseg.read_metadata(tmp_path / "bad.safetensors")


def test_files_from_before_the_first_release_are_refused_clearly(tmp_path):
    """Format 1 never shipped, so nothing migrates it; the error says what to do."""
    from safetensors.torch import load_file, save_file

    model = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    path = manyseg.save(model, tmp_path / "new.safetensors", activations={"logits": "softmax"},
                        normalize=[])
    with safe_open(str(path), framework="pt") as f:
        stored = dict(f.metadata())
    meta = json.loads(stored["manyseg"])
    meta["format"] = 1
    old = tmp_path / "old.safetensors"
    save_file(load_file(str(path)), str(old), metadata={**stored, "manyseg": json.dumps(meta)})
    with pytest.raises(ValueError, match="predates ManySeg 0.1.0"):
        manyseg.load(old)
    with pytest.raises(ValueError, match="predates ManySeg 0.1.0"):
        manyseg.read_metadata(old)


@pytest.mark.parametrize("config", [
    {"inputs": {"a": {"channels": 3}}, "num_classes": 2},
    {"inputs": {"a": {"in_channels": 3}, "m": {"kind": "vector", "channels": 2}}, "num_classes": 2},
    {"heads": {"h": {"type": "segmentation", "channels": 2}}},
], ids=["input", "vector", "head"])
def test_the_old_channels_key_is_an_ordinary_config_error(config):
    with pytest.raises(ValueError, match="channels|unknown"):
        create_model("resnet18", pretrained=False, **config)


def test_save_names_the_model_inside_a_wrapper(tmp_path):
    from manyseg.utils import concat_heads

    model = create_model(backbone="resnet18", pretrained=False, heads={
        "a": {"type": "segmentation", "out_channels": 2},
        "b": {"type": "regression", "out_channels": 1}})
    with pytest.raises(TypeError, match=r"It wraps one as \.model: save that"):
        manyseg.save(concat_heads(model, ["a", "b"]), tmp_path / "a.safetensors")


def _rng_state():
    return torch.get_rng_state(), random.getstate(), np.random.get_state()


def _assert_same_rng(before, after):
    assert torch.equal(before[0], after[0]), "torch RNG moved"
    assert before[1] == after[1], "random RNG moved"
    assert before[2][0] == after[2][0] and np.array_equal(before[2][1], after[2][1]) \
        and before[2][2:] == after[2][2:], "numpy RNG moved"


RNG_MODELS = {
    "plain": {"in_channels": 4, "num_classes": 2},
    "metadata": {"inputs": {"image": {"in_channels": 3},
                            "meta": {"kind": "vector", "fields": FIELDS, "dropout": 0.5}},
                 "num_classes": 2},
    "shared_encoder": {"inputs": {"a": {"in_channels": 3},
                                  "b": {"in_channels": 3, "backbone": "shared:a"}},
                       "num_classes": 2},
}


@pytest.mark.parametrize("config", RNG_MODELS.values(), ids=RNG_MODELS)
def test_save_and_load_leave_the_rng_alone(config, tmp_path):
    """Regression: both rebuild the model, and its weight init used the caller's random stream,
    so a checkpoint changed all later dropout, augmentation and sampling."""
    model = create_model("resnet18", pretrained=False, **config)
    before = _rng_state()
    path = manyseg.save(model, tmp_path / "m.safetensors", **{
        "activations": activations(model), "normalize": normalize_for(model)})
    _assert_same_rng(before, _rng_state())
    manyseg.load(path)
    _assert_same_rng(before, _rng_state())


def test_save_and_load_keep_the_rng_of_a_quantized_model(tmp_path):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_weights

    model = quantize_weights(create_model("resnet18", pretrained=False, num_classes=2))
    before = _rng_state()
    path = manyseg.save(model, tmp_path / "q.safetensors", activations={"logits": "softmax"},
                        normalize=[{"type": "scale", "value": 1}])
    _assert_same_rng(before, _rng_state())
    manyseg.load(path)
    _assert_same_rng(before, _rng_state())


def test_a_checkpoint_does_not_change_a_seeded_training_run(tmp_path):
    def run(checkpoint):
        torch.manual_seed(0)
        model = create_model("resnet18", pretrained=False, num_classes=2)
        opt = torch.optim.SGD(model.parameters(), 0.01)
        for step in range(3):
            x, y = torch.randn(2, 3, 64, 64), torch.randint(0, 2, (2, 64, 64))
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if checkpoint and step == 0:
                path = manyseg.save(model, tmp_path / "c.safetensors",
                                    activations={"logits": "softmax"},
                                    normalize=[{"type": "scale", "value": 1}])
                manyseg.load(path)
        return loss.item()

    assert run(False) == run(True)


def test_syncbatchnorm_models_save_and_load_as_batchnorm(tmp_path):
    """Multi-GPU training converts BatchNorm to SyncBatchNorm: same weights and, outside
    training, the same computation, so the file rebuilds the model it holds."""
    plain = create_model("resnet18", pretrained=False, num_classes=2)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(plain)
    assert any(isinstance(m, torch.nn.SyncBatchNorm) for m in model.modules())
    loaded = roundtrip(model, torch.randn(2, 3, 64, 64), tmp_path)
    assert not any(isinstance(m, torch.nn.SyncBatchNorm) for m in loaded.modules())


def test_timm_syncbatchnorm_models_save(tmp_path):
    """timm's own conversion keeps the activation inside BatchNormAct2d."""
    from timm.layers import convert_sync_batchnorm

    model = convert_sync_batchnorm(
        create_model("efficientnet_b0", pretrained=False, num_classes=2))
    roundtrip(model, torch.randn(2, 3, 64, 64), tmp_path)


def test_torch_syncbatchnorm_conversion_that_drops_an_activation_is_refused(tmp_path):
    """torch's conversion turns timm's BatchNormAct2d into a bare SyncBatchNorm, losing its
    activation: the model computes something the file would not rebuild."""
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        create_model("efficientnet_b0", pretrained=False, num_classes=2))
    with pytest.raises(ValueError, match="differs from what its create_model"):
        manyseg.save(model, tmp_path / "m.safetensors", activations={"logits": "softmax"},
                     normalize=[{"type": "scale", "value": 1}])


def test_a_lookalike_class_named_syncbatchnorm_is_refused(tmp_path):
    """The exception is for the real classes: this one would not be rebuilt by the file."""

    class SyncBatchNorm(torch.nn.BatchNorm2d):
        def forward(self, x):
            return super().forward(x) * 0

    model = create_model("resnet18", pretrained=False, num_classes=2)
    name = next(n for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d))
    parent, _, child = name.rpartition(".")
    original = getattr(model.get_submodule(parent), child)
    setattr(model.get_submodule(parent), child, SyncBatchNorm(original.num_features))
    SyncBatchNorm.__qualname__ = "SyncBatchNorm"  # as a class defined at module level would be
    with pytest.raises(ValueError, match="differs from what its create_model"):
        manyseg.save(model, tmp_path / "m.safetensors", activations={"logits": "softmax"},
                     normalize=[{"type": "scale", "value": 1}])
