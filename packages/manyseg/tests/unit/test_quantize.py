"""manyseg.quantize: torchao weight-only / dynamic int8, size and agreement reports.

The PT2E (static int8, exported) path is in ``integration/test_quantize_pt2e.py`` (slow)."""

import json
import sys

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg.quantize import compare_outputs, model_size, quantize_pt2e, quantize_weights
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn


def _model(backbone="resnet18", **kw):
    return create_model(backbone=backbone, pretrained=False, num_classes=3, **kw).eval()


def _batches(n=2, hw=(64, 96), c=3):
    return [torch.randn(1, c, *hw) for _ in range(n)]


# --- reports -------------------------------------------------------------------------------

def test_compare_outputs_identical_model():
    model = _model()
    r = compare_outputs(model, model, _batches())
    assert r == {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "argmax_agreement": 1.0}


def test_compare_outputs_detects_change():
    a = nn.Conv2d(3, 4, 1)
    b = nn.Conv2d(3, 4, 1)
    with torch.no_grad():
        b.weight.copy_(a.weight)
        b.bias.copy_(a.bias + 0.5)
    r = compare_outputs(a, b, _batches())
    assert r["max_abs_diff"] == pytest.approx(0.5, abs=1e-5)
    assert r["mean_abs_diff"] == pytest.approx(0.5, abs=1e-5)
    assert r["argmax_agreement"] == 1.0  # a constant shift never changes the argmax


def test_compare_outputs_dict_heads_are_prefixed():
    model = create_model(backbone="resnet18", pretrained=False,
                         heads={"seg": {"type": "segmentation", "out_channels": 3},
                                "height": {"type": "regression", "out_channels": 1}}).eval()
    r = compare_outputs(model, model, _batches(1))
    assert r["seg/argmax_agreement"] == 1.0
    assert r["height/max_abs_diff"] == 0.0
    assert "height/argmax_agreement" not in r  # one channel: argmax is meaningless


def test_compare_outputs_rejects_empty_inputs():
    with pytest.raises(ValueError, match="at least one"):
        compare_outputs(nn.Identity(), nn.Identity(), [])


class _Labels(nn.Module):
    """Returns an integer label map (B, H, W), optionally as another dtype."""

    def __init__(self, dtype=torch.long, offset=0):
        super().__init__()
        self.dtype, self.offset = dtype, offset

    def forward(self, x):
        return (x[:, 0] > 0).to(self.dtype) + self.offset


def test_compare_outputs_integer_labels_use_exact_match():
    """No argmax over dim 1 of a (B, H, W) label map: exact-match agreement instead."""
    r = compare_outputs(_Labels(), _Labels(offset=1), _batches())
    assert "argmax_agreement" not in r
    assert r["exact_agreement"] == 0.0
    assert compare_outputs(_Labels(), _Labels(), _batches())["exact_agreement"] == 1.0


def test_compare_outputs_flags_dtype_change():
    with pytest.warns(UserWarning, match="dtype"):
        r = compare_outputs(_Labels(), _Labels(torch.int32), _batches())
    assert r["dtype_changed"] == 1.0
    assert "dtype_changed" not in compare_outputs(_Labels(), _Labels(), _batches())


def test_compare_outputs_argmax_only_for_logit_shapes():
    """(B, C) and (B, C, H, W) floats get argmax agreement; a (B, N, D) float does not."""
    assert "argmax_agreement" in compare_outputs(nn.Flatten(), nn.Flatten(), _batches(1))
    tokens = nn.Flatten(2)  # (B, C, H*W): not a logit shape
    assert "argmax_agreement" not in compare_outputs(tokens, tokens, _batches(1))


def test_model_size_counts_state_dict_bytes():
    m = nn.Linear(100, 10)  # 1010 float32 values
    assert 1010 * 4 <= model_size(m) < 1010 * 4 + 2048


# --- torchao weight-only / dynamic ---------------------------------------------------------

@pytest.mark.parametrize("mode", ["int8_weight_only", "int8_dynamic"])
@pytest.mark.parametrize("backbone", ["resnet18", "mobilenetv3_small_100", "caformer_s18"])
def test_quantize_weights_odd_sizes_agreement_and_size(mode, backbone):
    """MobileNetV3 has depthwise convs (left in float); CAFormer is mostly Linear layers."""
    pytest.importorskip("torchao")
    model = _model(backbone)
    q = quantize_weights(model, mode)
    for hw in [(64, 96), (97, 131), (75, 33)]:
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            assert q(x).shape == model(x).shape == (1, 3, *hw)
    r = compare_outputs(model, q, _batches(2, (97, 131)))
    assert r["argmax_agreement"] > 0.95
    assert model_size(q) < 0.35 * model_size(model)


def test_quantize_weights_copies_unless_inplace():
    pytest.importorskip("torchao")
    model = _model()
    w = model.backbone.model.conv1.weight
    quantize_weights(model)
    assert model.backbone.model.conv1.weight is w and type(w) is nn.Parameter
    q = quantize_weights(model, inplace=True)
    assert q is model and type(model.backbone.model.conv1.weight) is not nn.Parameter


def test_quantize_weights_conv_false_leaves_convs():
    pytest.importorskip("torchao")
    model = _model("caformer_s18")
    q = quantize_weights(model, conv=False)
    for m in q.modules():
        if isinstance(m, nn.Conv2d):
            assert type(m.weight) is nn.Parameter
    assert any(type(m.weight) is not nn.Parameter for m in q.modules()
               if isinstance(m, nn.Linear))


def test_quantize_weights_rejects_unknown_mode():
    pytest.importorskip("torchao")
    with pytest.raises(ValueError, match="int8_weight_only"):
        quantize_weights(_model(), "int4")


def test_quantize_weights_device(device):
    pytest.importorskip("torchao")
    model = _model().to(device)
    q = quantize_weights(model)
    x = torch.randn(1, 3, 97, 131, device=device)
    with torch.no_grad():
        assert q(x).shape == (1, 3, 97, 131)
    assert compare_outputs(model, q, [x])["argmax_agreement"] > 0.95


# --- PT2E helpers (no export needed) -------------------------------------------------------

@pytest.mark.parametrize("backbone", ["swin_tiny_patch4_window7_224", "swinv2_tiny_window8_256",
                                      "davit_tiny"])
def test_windowed_detects_timm_window_attention(backbone):
    from manyseg.quantize import _windowed

    assert _windowed(_model(backbone)) == "WindowAttention"


def test_windowed_ignores_unrelated_window_size_attribute():
    from manyseg.quantize import _windowed

    model = _model()
    model.decoder.window_size = 7  # e.g. a user module with an unrelated attribute
    assert _windowed(model) is None


def test_drop_unused_attrs_keeps_params_of_called_submodules():
    """A parameter reached through a call_module (target ``body``, param ``body.weight``) is
    used; only attributes no node reaches are dropped."""
    from manyseg.quantize import _drop_unused_attrs

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Conv2d(3, 4, 1)
            self.scale = nn.Parameter(torch.ones(1))
            self.unused = nn.Parameter(torch.ones(5))

        def forward(self, x):
            return self.body(x) * self.scale

    gm = torch.fx.symbolic_trace(M())
    x = torch.randn(1, 3, 8, 8)
    ref = gm(x)
    _drop_unused_attrs(gm)
    assert set(dict(gm.named_parameters())) == {"body.weight", "body.bias", "scale"}
    torch.testing.assert_close(gm(x), ref)


# --- missing optional dependency -----------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda m: quantize_weights(m),
    lambda m: quantize_pt2e(m, torch.randn(1, 3, 64, 64), [torch.randn(1, 3, 64, 64)]),
])
def test_missing_torchao_error(monkeypatch, call):
    monkeypatch.setitem(sys.modules, "torchao", None)  # makes `import torchao` raise
    with pytest.raises(ImportError, match=r"manyseg\[quant\]"):
        call(_model())


def test_pt2e_names_dynamic_batch_when_calibration_batch_differs():
    """Rough edge (examples review): a different calibration batch size failed deep in an
    export guard; it is now a clear error up front."""
    pytest.importorskip("torchao")
    from manyseg import create_model
    from manyseg.quantize import quantize_pt2e

    m = create_model(backbone="resnet18", pretrained=False, in_channels=4, num_classes=3).eval()
    with pytest.raises(ValueError, match="dynamic_batch=True"):
        quantize_pt2e(m, torch.randn(1, 4, 64, 64), [torch.randn(2, 4, 64, 64)])


@pytest.mark.parametrize("vnni,expected_max", [(True, 255), (False, 127)])
def test_default_x86_quantizer_reduces_range_without_vnni(monkeypatch, vnni, expected_max):
    """E008: on CPUs without VNNI (e.g. AMD Zen 3), compiled u8 x s8 kernels saturate and the
    deployed int8 model lost ~6 mIoU on Pet; reduce_range (7-bit activations) fixes it."""
    pytest.importorskip("torchao")
    from manyseg import quantize as mq

    monkeypatch.setattr(torch.cpu, "_is_vnni_supported", lambda: vnni)
    q = mq._x86_quantizer(None)
    assert q.global_config.input_activation.quant_max == expected_max
    assert mq._x86_quantizer(True).global_config.input_activation.quant_max == 127
    assert mq._x86_quantizer(False).global_config.input_activation.quant_max == 255


def test_reduce_range_falls_back_to_safe_when_undetectable(monkeypatch):
    pytest.importorskip("torchao")
    from manyseg import quantize as mq

    monkeypatch.delattr(torch.cpu, "_is_vnni_supported", raising=False)
    assert mq._x86_quantizer(None).global_config.input_activation.quant_max == 127


def _tiny_int8(monkeypatch, vnni, **kw):
    from manyseg import quantize as mq

    monkeypatch.setattr(mq, "_has_vnni", lambda: vnni)
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 2, 1)).eval()
    x = torch.randn(2, 3, 16, 16)
    return mq, mq.quantize_pt2e(m, x, [x], dynamic=False, **kw)


def test_full_range_int8_warns_and_records_vnni_requirement(monkeypatch, tmp_path):
    """E008: a full-range (non-reduce_range) int8 program is only accurate on VNNI CPUs, so the
    risky choice warns at quantisation time and is recorded in the program, and
    check_deployment refuses a non-VNNI CPU (survives torch.export.save / load)."""
    pytest.importorskip("torchao")
    with pytest.warns(UserWarning, match="VNNI"):
        mq, ep = _tiny_int8(monkeypatch, vnni=True)
    path = tmp_path / "m.pt2"
    torch.export.save(ep, str(path))
    ep = torch.export.load(str(path))
    mq.check_deployment(ep)  # this (VNNI) CPU is fine
    monkeypatch.setattr(mq, "_has_vnni", lambda: False)
    with pytest.raises(RuntimeError, match="reduce_range"):
        mq.check_deployment(ep)


def test_reduce_range_int8_is_silent_and_deploys_anywhere(monkeypatch, recwarn):
    pytest.importorskip("torchao")
    mq, ep = _tiny_int8(monkeypatch, vnni=False)
    assert not [w for w in recwarn if "VNNI" in str(w.message)]
    mq.check_deployment(ep)
    monkeypatch.setattr(mq, "_has_vnni", lambda: True)
    mq.check_deployment(ep)


def test_check_deployment_warns_when_unrecorded():
    from manyseg.quantize import check_deployment

    m = nn.Conv2d(3, 2, 1)
    ep = torch.export.export(m, (torch.randn(1, 3, 4, 4),))
    with pytest.warns(UserWarning, match="not produced by"):
        check_deployment(ep)


# --- save / load ---------------------------------------------------------------------------

_NORM = [{"type": "scale", "value": 10000}]


def _save_args(model):
    inputs = model.build_config.get("inputs")
    norm = ({n: _NORM for n, c in inputs.items() if c.get("kind") != "vector"} if inputs
            else _NORM)
    return {"activations": {h: "softmax" for h in model.heads}, "normalize": norm}


def _same_outputs(a, b, x):
    with torch.no_grad():
        out_a, out_b = a(x), b(x)
    if isinstance(out_a, dict):
        assert set(out_a) == set(out_b)
        for k in out_a:
            assert torch.equal(out_a[k], out_b[k]), k
    else:
        assert torch.equal(out_a, out_b)


@pytest.mark.parametrize("mode", ["int8_weight_only", "int8_dynamic"])
@pytest.mark.parametrize("kw", [{"backbone": "resnet18"},
                                {"backbone": "resnet18", "decoder": "segformer"},
                                {"backbone": "mobilenetv3_small_100"},
                                {"backbone": "pvt_v2_b0"}],
                         ids=lambda k: "-".join(map(str, k.values())))
def test_weights_round_trip_same_outputs(tmp_path, mode, kw):
    pytest.importorskip("torchao")
    q = quantize_weights(_model(**kw), mode)
    path = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))
    loaded = manyseg.load(path)
    assert isinstance(loaded, manyseg.DenseModel) and not loaded.training
    _same_outputs(q, loaded, torch.randn(1, 3, 97, 131))
    # int8 for real, and the same layers as before saving
    want, have = q.state_dict(), loaded.state_dict()
    assert {k: type(v) for k, v in want.items()} == {k: type(v) for k, v in have.items()}
    for k, v in want.items():  # torchao keeps the real values in attributes, not in .data
        for part in getattr(v, "tensor_data_names", ()):
            a, b = getattr(v, part), getattr(have[k], part)
            assert a.dtype == b.dtype and torch.equal(a, b), f"{k}:{part}"
    assert any(getattr(v, "qdata", None) is not None and v.qdata.dtype == torch.int8
               for v in have.values())


def test_weights_conv_false_round_trip(tmp_path):
    pytest.importorskip("torchao")
    q = quantize_weights(_model("pvt_v2_b0"), conv=False)
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q)))
    _same_outputs(q, loaded, torch.randn(1, 3, 65, 97))
    assert loaded.saved_metadata["quantization"]["conv"] is False


def test_weights_file_size_under_target(tmp_path):
    pytest.importorskip("torchao")
    model = _model()
    q = quantize_weights(model)
    a = manyseg.save(model, tmp_path / "f.safetensors", **_save_args(model))
    b = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))
    assert b.stat().st_size < 0.35 * a.stat().st_size


def test_float_files_keep_format_2(tmp_path):
    model = _model()
    path = manyseg.save(model, tmp_path / "f.safetensors", **_save_args(model))
    meta = manyseg.read_metadata(path)
    assert meta["format"] == 2 and "quantization" not in meta


def test_weights_file_is_format_2_without_pickle(tmp_path):
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    path = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))
    meta = manyseg.read_metadata(path)
    assert meta["format"] == 2  # the schema version; the quantization entry marks the kind
    assert meta["quantization"]["route"] == "weights"
    assert meta["quantization"]["mode"] == "int8_weight_only"
    with safe_open(str(path), framework="pt") as f:  # safetensors: plain tensors only
        assert "backbone.model.conv1.weight:qdata" in set(f.keys())
        assert f.get_tensor("backbone.model.conv1.weight:qdata").dtype == torch.int8


def _reduced_head_model():
    return create_model(backbone="resnet18", pretrained=False, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "coarse": {"type": "segmentation", "out_channels": 2, "scale": 0.25},
        "label": {"type": "classification", "scope": "image", "out_channels": 4},
        "height": {"type": "regression", "out_channels": 1}}).eval()


def test_weights_tiling_spec_matches_float_after_load(tmp_path):
    pytest.importorskip("torchao")
    model = _reduced_head_model()
    args = {"normalize": _NORM, "activations": {"seg": "softmax", "coarse": "sigmoid",
                                                "label": "softmax"},
            "classes": {"seg": ["a", "b", "c"]}, "bands": ["r", "g", "b"]}
    float_loaded = manyseg.load(manyseg.save(model, tmp_path / "f.safetensors", **args))
    q = quantize_weights(model)
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **args))
    spec = loaded.tiling_spec()
    assert spec == float_loaded.tiling_spec()
    assert spec["normalize"] == _NORM
    assert spec["outputs"]["coarse"]["scale"] == 0.25
    assert spec["outputs"]["coarse"]["activation"] == "sigmoid"
    assert loaded.saved_metadata["classes"]["seg"] == ["a", "b", "c"]
    x = torch.randn(1, 3, 97, 131)
    _same_outputs(q, loaded, x)
    assert loaded(x)["coarse"].shape[-2:] == (25, 33)


def test_weights_single_output_round_trip(tmp_path):
    pytest.importorskip("torchao")
    model = _reduced_head_model()
    model.select_heads(["coarse"])
    q = quantize_weights(model)
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q)))
    _same_outputs(q, loaded, torch.randn(1, 3, 97, 131))
    assert loaded.tiling_spec() == {**model.tiling_spec(), "normalize": _NORM, "outputs": {
        "coarse": {**model.tiling_spec()["outputs"]["coarse"], "activation": "softmax"}}}


@pytest.mark.parametrize("share", [False, True], ids=["two-encoders", "shared-encoder"])
def test_weights_multi_input_round_trip(tmp_path, share):
    pytest.importorskip("torchao")
    inputs = ({"a": {"in_channels": 3}, "b": {"in_channels": 3, "backbone": "shared:a"}}
              if share else {"a": {"in_channels": 4}, "b": {"in_channels": 2}})
    model = create_model(inputs=inputs, backbone="resnet18", pretrained=False,
                         num_classes=2).eval()
    q = quantize_weights(model)
    args = {**_save_args(q), "normalize": {"a": _NORM, "b": _NORM}}
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **args))
    c = (3, 3) if share else (4, 2)
    x = {"a": torch.randn(1, c[0], 97, 131), "b": torch.randn(1, c[1], 97, 131)}
    _same_outputs(q, loaded, x)
    assert loaded.tiling_spec()["inputs"] == model.tiling_spec()["inputs"]


def test_weights_one_feature_vector_input_round_trip(tmp_path):
    """torchao can't quantise nn.Linear(1, n), so that layer stays float."""
    pytest.importorskip("torchao")
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"in_channels": 4}, "doy": {"kind": "vector", "num_features": 1}}).eval()
    q = quantize_weights(model)
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q)))
    _same_outputs(q, loaded, {"img": torch.randn(1, 4, 97, 131), "doy": torch.rand(1, 1)})
    assert loaded.tiling_spec()["inputs"]["doy"] == {"kind": "vector"}


def test_weights_window_attention_round_trip(tmp_path):
    pytest.importorskip("torchao")
    q = quantize_weights(_model("swin_tiny_patch4_window7_224"))
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q)))
    _same_outputs(q, loaded, torch.randn(1, 3, 224, 224))
    assert loaded.tiling_spec()["max_size"] == q.tiling_spec()["max_size"]


def test_weights_loaded_model_moves_and_resaves(tmp_path):
    """ManyPred calls .to(device=..., dtype=...) on a module; a second save still works."""
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q)))
    x = torch.randn(1, 3, 65, 97)
    _same_outputs(q, loaded.to(device="cpu", dtype=torch.float32).eval(), x)
    again = manyseg.load(manyseg.save(loaded, tmp_path / "r.safetensors", **_save_args(q)))
    _same_outputs(q, again, x)


def test_weights_changed_model_is_refused(tmp_path):
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    q.decoder.extra = nn.Conv2d(1, 1, 1)
    with pytest.raises(ValueError, match="differs from what its create_model"):
        manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))


def test_save_refuses_weights_quantised_outside_manyseg(tmp_path):
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    del q.quantization
    with pytest.raises(ValueError, match="did not come from"):
        manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))


def test_load_quantised_file_without_torchao_names_the_extra(tmp_path, monkeypatch):
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    path = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))
    monkeypatch.setitem(sys.modules, "torchao", None)
    assert "quantization" in manyseg.read_metadata(path)  # the metadata needs no torchao
    with pytest.raises(ImportError, match=r"manyseg\[quant\]"):
        manyseg.load(path)


def test_older_torchao_part_mismatch_is_a_clear_error(tmp_path):
    """A file whose quantised layers don't line up names the versions, not a KeyError."""
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    path = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))
    with safe_open(str(path), framework="pt") as f:
        meta, tensors = f.metadata(), {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    del tensors["backbone.model.conv1.weight:scale"]
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad), metadata=meta)
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        manyseg.load(bad)


def test_save_refuses_weights_that_do_not_split(tmp_path):
    """Older torchao's Linear weights are a subclass without plain parts."""
    pytest.importorskip("torchao")

    class Opaque(torch.Tensor):
        pass

    q = quantize_weights(_model())
    q.backbone.model.bn1.weight = nn.Parameter(q.backbone.model.bn1.weight.as_subclass(Opaque))
    with pytest.raises(ValueError, match="upgrade it"):
        manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))


def _rewrite(path, tmp_path, edit):
    """A copy of saved file ``path`` with its manyseg metadata changed by ``edit(meta)``."""
    with safe_open(str(path), framework="pt") as f:
        meta, tensors = f.metadata(), {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    stored = json.loads(meta["manyseg"])
    edit(stored)
    out = tmp_path / f"edited_{path.name}"
    save_file(tensors, str(out), metadata={**meta, "manyseg": json.dumps(stored)})
    return out


def test_float_file_with_format_3_asks_for_an_upgrade(tmp_path):
    """Format 3 is retired (pre-release quantised files): without a quantization entry it is
    a future format, and an upgrade is the right advice."""
    model = _model()
    path = manyseg.save(model, tmp_path / "f.safetensors", **_save_args(model))
    bad = _rewrite(path, tmp_path, lambda m: m.update(format=3))
    with pytest.raises(ValueError, match="Upgrade ManySeg"):
        manyseg.load(bad)


def test_pre_release_format_3_quantised_file_still_loads(tmp_path):
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    path = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))
    old = _rewrite(path, tmp_path, lambda m: m.update(format=3))
    assert manyseg.read_metadata(old)["format"] == 2
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(manyseg.load(old)(x), manyseg.load(path)(x))


def test_invalid_quantization_entry_is_refused(tmp_path):
    model = _model()
    path = manyseg.save(model, tmp_path / "f.safetensors", **_save_args(model))
    bad = _rewrite(path, tmp_path, lambda m: m.update(quantization="int8"))
    with pytest.raises(ValueError, match="invalid quantisation metadata"):
        manyseg.read_metadata(bad)


def test_quantised_file_from_newer_manyseg_blames_manyseg_not_torchao(tmp_path):
    """A create_model option this ManySeg doesn't know is a version problem, whatever torchao
    is installed."""
    pytest.importorskip("torchao")
    q = quantize_weights(_model())
    path = manyseg.save(q, tmp_path / "q.safetensors", **_save_args(q))

    def newer(m):
        m["manyseg_version"] = "99.0.0"
        m["create_model"]["future_option"] = 1

    with pytest.raises(ValueError, match=r"saved by ManySeg 99\.0\.0") as err:
        manyseg.load(_rewrite(path, tmp_path, newer))
    assert "torchao" not in str(err.value)


def test_save_ep_needs_source_and_module_is_rejected(tmp_path):
    pytest.importorskip("torchao")
    ep = torch.export.export(nn.Conv2d(3, 1, 1).eval(), (torch.randn(1, 3, 8, 8),))
    with pytest.raises(TypeError, match="source="):
        manyseg.save(ep, tmp_path / "m.pt2")
    with pytest.raises(TypeError, match="ExportedProgram itself"):
        manyseg.save(ep.module(), tmp_path / "m.pt2")
    with pytest.raises(TypeError, match="source= is only"):
        manyseg.save(_model(), tmp_path / "m.safetensors", source=_model())
