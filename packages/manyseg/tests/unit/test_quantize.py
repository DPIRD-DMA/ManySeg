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


# --- missing optional dependency -----------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda m: quantize_weights(m),
    lambda m: quantize_pt2e(m, torch.randn(1, 3, 64, 64), [torch.randn(1, 3, 64, 64)]),
])
def test_missing_torchao_error(monkeypatch, call):
    monkeypatch.setitem(sys.modules, "torchao", None)  # makes `import torchao` raise
    with pytest.raises(ImportError, match=r"manyseg\[quant\]"):
        call(_model())
