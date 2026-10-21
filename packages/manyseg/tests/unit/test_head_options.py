"""Pixel-head options (hidden layers, dilation, norm, output activation), registered custom
head classes and the single-tensor wrapper."""

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg.features import FeatureSpec
from manyseg.heads import HEADS, PixelHead
from manyseg.utils import concat_heads
from torch import nn

HIDDEN = {"type": "regression", "out_channels": 1, "hidden": [16, 8, 4],
          "dilation": [8, 16, 32]}


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def model(**heads):
    return create_model(backbone="resnet18", pretrained=False, heads=heads)


def test_default_head_unchanged():
    head = PixelHead(FeatureSpec(16, 1), 2)
    assert head.hidden is None and head.activation is None
    assert nparams(head) == 16 * 2 * 9 + 2  # one 3x3 conv, as in SMP
    assert [k for k in head.state_dict()] == ["conv.weight", "conv.bias"]


@pytest.mark.parametrize("size", [(67, 93), (32, 32), (41, 160)])
def test_hidden_head_odd_sizes(size):
    m = model(seg={"type": "segmentation", "out_channels": 2}, hm=HIDDEN).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, *size))
    assert out["hm"].shape == (2, 1, *size)
    convs = [c for c in m.heads["hm"].hidden if isinstance(c, nn.Conv2d)]
    assert [c.dilation[0] for c in convs] == [8, 16, 32]
    assert [c.out_channels for c in convs] == [16, 8, 4]
    assert m.heads["hm"].conv.in_channels == 4
    bn = [b for b in m.heads["hm"].hidden if isinstance(b, nn.BatchNorm2d)]
    assert len(bn) == 3 and all(c.bias is None for c in convs)  # bias-free conv before norm


def test_hidden_params_and_norms():
    spec = FeatureSpec(16, 1)
    base = nparams(PixelHead(spec, 1))
    plain = PixelHead(spec, 1, hidden=[8], norm=None)
    assert nparams(plain) == (16 * 8 * 9 + 8) + (8 * 9 + 1)
    assert nparams(PixelHead(spec, 1, hidden=[8], norm="gn")) == 16 * 8 * 9 + 2 * 8 + 8 * 9 + 1
    assert nparams(plain) > base
    head = PixelHead(spec, 1, hidden=[8, 8], dilation=4, bias_init=-2.0)
    assert torch.all(head.conv.bias == -2.0)  # bias_init: still the final conv


def test_hidden_scaled_head():
    m = model(hm={**HIDDEN, "scale": 0.25}).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 70, 90))["hm"].shape == (1, 1, 18, 23)


@pytest.mark.parametrize("bad, msg", [
    ({"hidden": 16}, "list of positive"),
    ({"hidden": [16, 0]}, "list of positive"),
    ({"hidden": [16, 8], "dilation": [2]}, "one per hidden layer"),
    ({"hidden": [16], "norm": "ln"}, "norm must be one of"),
    ({"dilation": 4}, "give 'hidden' too"),
    ({"norm": "bn"}, "give 'hidden' too"),
    ({"activation": "sigmoid"}, "activation must be one of"),
])
def test_hidden_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        model(hm={"type": "regression", "out_channels": 1, **bad})


def test_activation_rules():
    with pytest.raises(ValueError, match="only for regression"):
        model(s={"type": "segmentation", "out_channels": 2, "activation": "softplus"})
    with pytest.raises(ValueError, match="log-variance"):
        model(r={"type": "regression", "out_channels": 1, "uncertainty": True,
                 "activation": "softplus"})
    with pytest.raises(ValueError, match="pixel heads only"):
        model(r={"type": "regression", "out_channels": 1, "scope": "image", "hidden": [8]})


def test_softplus_nonnegative():
    m = model(d={"type": "regression", "out_channels": 1, "activation": "softplus",
                 "bias_init": -20.0, "scale": 0.5})
    out = m(torch.randn(2, 3, 50, 61))["d"]
    assert out.min() >= 0
    assert "activation: softplus" in m.describe()


def test_describe_shows_hidden():
    text = model(hm={**HIDDEN, "norm": "gn"}).describe()
    assert "hidden: 16 (d8) -> 8 (d16) -> 4 (d32), gn + ReLU" in text


def test_save_load_round_trip(tmp_path):
    m = model(seg={"type": "segmentation", "out_channels": 2}, hm=HIDDEN,
              d={"type": "regression", "out_channels": 1, "activation": "softplus"}).eval()
    x = torch.randn(1, 3, 48, 56)
    with torch.no_grad():
        ref = m(x)
    path = manyseg.save(m, tmp_path / "m.safetensors", normalize=[],
                        activations={"seg": "softmax"})
    back = manyseg.load(path).eval()
    with torch.no_grad():
        got = back(x)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k])
    assert manyseg.io.read_metadata(path)["activations"]["d"] == "none"
    with pytest.raises(ValueError, match="applies softplus itself"):
        manyseg.save(m, tmp_path / "x.safetensors", normalize=[],
                     activations={"seg": "softmax", "d": "sigmoid"})


@pytest.fixture
def custom_head():
    class DoubleConv(nn.Module):
        def __init__(self, spec, channels, width=8):
            super().__init__()
            self.out_channels = channels
            self.net = nn.Sequential(nn.Conv2d(spec.channels, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, channels, 1))

        def forward(self, x):
            return self.net(x)

    HEADS.register("double_conv_test")(DoubleConv)
    yield DoubleConv
    del HEADS["double_conv_test"]


def test_custom_head_class(custom_head, tmp_path):
    m = model(seg={"type": "segmentation", "out_channels": 2},
              c={"type": "double_conv_test", "out_channels": 3, "width": 5}).eval()
    assert isinstance(m.heads["c"], custom_head) and m.heads["c"].kind == "double_conv_test"
    x = torch.randn(1, 3, 45, 70)
    with torch.no_grad():
        ref = m(x)
    assert ref["c"].shape == (1, 3, 45, 70)
    assert "c: double_conv_test, pixel, 3 ch" in m.describe()
    with pytest.raises(ValueError, match="turn into predictions"):
        manyseg.save(m, tmp_path / "m.safetensors", normalize=[],
                     activations={"seg": "softmax"})
    path = manyseg.save(m, tmp_path / "m.safetensors", normalize=[],
                        activations={"seg": "softmax", "c": "none"})
    with torch.no_grad():
        torch.testing.assert_close(manyseg.load(path).eval()(x)["c"], ref["c"])
    assert concat_heads(m, ["seg", "c"]).channels == {"seg": slice(0, 2), "c": slice(2, 5)}


def test_concat_heads():
    m = model(seg={"type": "segmentation", "out_channels": 2}, hm=HIDDEN,
              cls={"type": "classification", "out_channels": 4, "scope": "image"},
              coarse={"type": "regression", "out_channels": 1, "scale": 0.5})
    w = concat_heads(m, ["seg", "hm"])
    x = torch.randn(2, 3, 40, 52)
    out, raw = w(x), m(x)
    assert out.shape == (2, 3, 40, 52)
    torch.testing.assert_close(out[:, w.channels["seg"]], raw["seg"])
    torch.testing.assert_close(out[:, w.channels["hm"]], raw["hm"])
    assert set(w.state_dict()) == {f"model.{k}" for k in m.state_dict()}
    with pytest.raises(ValueError, match="different shapes"):
        concat_heads(m, ["seg", "cls"])
    with pytest.raises(ValueError, match="different grids"):
        concat_heads(m, ["seg", "coarse"])
    with pytest.raises(ValueError, match="distinct heads"):
        concat_heads(m, ["seg", "nope"])
    with pytest.raises(TypeError, match="list of head names"):
        concat_heads(m, "seg")


def test_empty_hidden_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        model(hm={"type": "regression", "out_channels": 1, "hidden": []})


def test_softplus_after_metadata_head_bias():
    m = create_model(
        inputs={"x": {"in_channels": 3},
                "meta": {"kind": "vector", "num_features": 2,
                         "conditioning": {"mode": "head", "pixel": True}}},
        backbone="resnet18", pretrained=False,
        heads={"d": {"type": "regression", "out_channels": 1, "activation": "softplus"}})
    bias = m.conditioner.head_bias["d"]
    with torch.no_grad():
        bias.bias.fill_(-50.0)  # a large negative metadata bias
    out = m({"x": torch.randn(1, 3, 40, 40), "meta": torch.randn(1, 2)})["d"]
    assert out.min() >= 0


def test_softplus_export_dynamic():
    from torch.export import Dim, export

    m = model(d={"type": "regression", "out_channels": 1, "activation": "softplus",
                 "hidden": [8]}).eval()
    ep = export(m, (torch.randn(1, 3, 64, 96),),
                dynamic_shapes=({2: Dim("h", min=64, max=512), 3: Dim("w", min=64, max=512)},))
    y = torch.randn(1, 3, 77, 130)
    with torch.no_grad():
        torch.testing.assert_close(ep.module()(y)["d"], m(y)["d"])


def test_relu_removed():
    """ReLU density heads died on the drone data (see density_head); it is not an option."""
    from manyseg.points import density_head

    with pytest.raises(ValueError, match="activation must be one of"):
        model(d={"type": "regression", "out_channels": 1, "activation": "relu"})
    with pytest.raises(ValueError, match="activation must be one of"):
        model(**density_head(activation="relu"))


def test_default_head_outputs_unchanged():
    """The default head is still exactly conv -> bilinear upsampling (the pre-option code)."""
    torch.manual_seed(0)
    head = PixelHead(FeatureSpec(8, 4), 3)
    x = torch.randn(2, 8, 5, 7)
    want = torch.nn.functional.interpolate(head.conv(x), size=(20, 28), mode="bilinear",
                                           align_corners=False)
    torch.testing.assert_close(head(x), want, rtol=0, atol=0)
    assert nparams(head) == 8 * 3 * 9 + 3


def test_softplus_onnx_matches_onnxruntime(tmp_path):
    act = "softplus"
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    m = create_model(backbone="resnet18", pretrained=False, heads={
        "d": {"type": "regression", "out_channels": 1, "activation": act, "hidden": [8],
              "dilation": 4}}).eval()
    path = tmp_path / "m.onnx"
    h, w = torch.export.Dim("h", min=33, max=2048), torch.export.Dim("w", min=33, max=2048)
    torch.onnx.export(m, (torch.randn(1, 3, 64, 96),), str(path), dynamo=True,
                      dynamic_shapes={"x": {2: h, 3: w}}, input_names=["x"],
                      output_names=["d"])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 65, 97)
    (got,) = sess.run(None, {"x": x.numpy()})
    with torch.no_grad():
        want = m(x)["d"]
    torch.testing.assert_close(torch.from_numpy(got), want, atol=1e-3, rtol=1e-3)
    assert got.min() >= 0
