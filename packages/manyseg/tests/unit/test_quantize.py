"""manyseg.quantize: torchao weight-only / dynamic int8, size and agreement reports.

The PT2E (static int8, exported) path is in ``integration/test_quantize_pt2e.py`` (slow)."""

import sys

import pytest
import torch
from manyseg import create_model
from manyseg.quantize import compare_outputs, model_size, quantize_pt2e, quantize_weights
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
                         heads={"seg": {"type": "segmentation", "channels": 3},
                                "height": {"type": "regression", "channels": 1}}).eval()
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
    model = _model()
    w = model.backbone.model.conv1.weight
    quantize_weights(model)
    assert model.backbone.model.conv1.weight is w and type(w) is nn.Parameter
    q = quantize_weights(model, inplace=True)
    assert q is model and type(model.backbone.model.conv1.weight) is not nn.Parameter


def test_quantize_weights_conv_false_leaves_convs():
    model = _model("caformer_s18")
    q = quantize_weights(model, conv=False)
    for m in q.modules():
        if isinstance(m, nn.Conv2d):
            assert type(m.weight) is nn.Parameter
    assert any(type(m.weight) is not nn.Parameter for m in q.modules()
               if isinstance(m, nn.Linear))


def test_quantize_weights_rejects_unknown_mode():
    with pytest.raises(ValueError, match="int8_weight_only"):
        quantize_weights(_model(), "int4")


def test_quantize_weights_device(device):
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
