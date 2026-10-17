"""manyseg.save / manyseg.load: rebuild from the stored create_model arguments, identical
outputs, metadata for inference (normalisation, activations, classes, bands)."""

import json

import manyseg
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


def roundtrip(model, x, tmp_path, **kw):
    path = manyseg.save(model.eval(), tmp_path / "m.safetensors", **kw)
    loaded = manyseg.load(path)
    a, b = outputs(model, x), outputs(loaded, x)
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), name
    return loaded


SAR = {"channels": 2, "backbone": "resnet18", "alignment": "same_extent"}
FIELDS = {"doy": {"type": "cyclic", "period": 365.25}}

CASES = {
    "single": (dict(in_channels=4, num_classes=3), lambda: torch.randn(2, 4, 64, 64)),
    "heads": (dict(in_channels=3, heads={
        "seg": {"type": "segmentation", "channels": 3},
        "height": {"type": "regression", "channels": 1, "scale": 0.25, "uncertainty": True},
        "scene": {"type": "classification", "channels": 5, "scope": "image"}}),
        lambda: torch.randn(2, 3, 64, 64)),
    "two_inputs": (dict(inputs={"s2": {"channels": 4, "backbone": "resnet18"}, "s1": SAR},
                        fusion={"type": "intermediate", "op": "gated"}, num_classes=2),
                   lambda: {"s2": torch.randn(2, 4, 64, 64), "s1": torch.randn(2, 2, 32, 32)}),
    "shared_backbone": (dict(inputs={"t1": {"channels": 3, "backbone": "resnet18"},
                                     "t2": {"channels": 3, "backbone": "shared:t1"}},
                             fusion={"type": "intermediate", "op": "concat"}, num_classes=2),
                        lambda: {"t1": torch.randn(2, 3, 64, 64),
                                 "t2": torch.randn(2, 3, 64, 64)}),
    "time_series": (dict(inputs={"ts": {"kind": "timeseries", "channels": 3,
                                        "temporal": "attention"}}, num_classes=2),
                    lambda: {"ts": torch.randn(2, 4, 3, 64, 64)}),
    "metadata": (dict(inputs={"img": {"channels": 3}, "meta": {"kind": "vector",
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
        "seg": {"type": "segmentation", "channels": 3},
        "aux": {"type": "regression", "channels": 1}})
    model.select_heads(["seg"])
    loaded = roundtrip(model, torch.randn(1, 3, 64, 64), tmp_path)
    assert list(loaded.heads) == ["seg"]


def test_metadata_and_tiling_spec(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, heads={
        "seg": {"type": "segmentation", "channels": 3},
        "height": {"type": "regression", "channels": 1}})
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
    path = manyseg.save(model, tmp_path / "m.safetensors")
    with safe_open(str(path), framework="pt") as f:
        assert json.loads(f.metadata()["manyseg"])["create_model"]["pretrained"] is True
    import timm

    created = []
    original = timm.create_model
    monkeypatch.setattr(timm, "create_model",
                        lambda *a, **k: created.append(k.get("pretrained")) or original(*a, **k))
    manyseg.load(path)
    assert created and not any(created)


def test_weights_tag_gives_the_default_normalisation():
    config = {"backbone": "resnet50", "pretrained": "ssl4eo_s2_moco", "inputs": None}
    steps = _weights_normalize(config)
    assert steps[0]["type"] == "zscore" and len(steps[0]["mean"]) == 13
    assert _weights_normalize({"backbone": "resnet18", "pretrained": True}) is None


def test_errors(tmp_path):
    from manyseg.models import DenseModel

    model = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2)
    with pytest.raises(ValueError, match="unknown head"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"nope": "softmax"})
    with pytest.raises(ValueError, match="one of"):
        manyseg.save(model, tmp_path / "a.safetensors", activations={"logits": "relu"})
    with pytest.raises(ValueError, match="plain JSON"):
        manyseg.save(model, tmp_path / "a.safetensors", extra={"x": object()})
    bare = DenseModel(model.backbone, model.decoder, dict(model.heads))
    with pytest.raises(ValueError, match="build_config"):
        manyseg.save(bare, tmp_path / "a.safetensors")
    from safetensors.torch import save_file

    save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"))
    with pytest.raises(ValueError, match="not written by manyseg.save"):
        manyseg.load(tmp_path / "other.safetensors")
