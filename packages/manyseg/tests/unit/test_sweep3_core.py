"""Regression tests for the third core sweep: repeated deepest strides (DeepLabV3+, image heads,
U-Net++), keyword-only dice_bce options, head area-average edge blocks, upsample_output grid
checks, raster checks with a vector input, one norm normaliser, module backbones per input,
invalid head decoders, FiLM ``on``."""

from __future__ import annotations

import inspect

import pytest
import torch
from manyseg import create_model
from manyseg.decoders import DeepLabV3PlusDecoder, UNetDecoder, UNetPlusPlusDecoder
from manyseg.features import FeatureSpec
from manyseg.heads import PixelHead
from manyseg.layers import ConvNormAct, normalize_norm
from manyseg.losses import dice_bce
from manyseg.utils import downsample_mean, upsample_output
from torch import nn


class Dilated(nn.Module):
    """Three stages at strides 4, 8, 8 (the last dilated, as timm ``output_stride``): the last
    stride-8 level is the deepest."""

    input_divisor = 8

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_specs = (FeatureSpec(8, 4), FeatureSpec(16, 8), FeatureSpec(32, 8))
        self.s4 = nn.Sequential(nn.Conv2d(in_channels, 8, 4, stride=4), nn.ReLU())
        self.s8 = nn.Sequential(nn.Conv2d(8, 16, 2, stride=2), nn.ReLU())
        self.s8d = nn.Sequential(nn.Conv2d(16, 32, 3, padding=2, dilation=2), nn.ReLU())

    def forward(self, x):
        a = self.s4(x)
        b = self.s8(a)
        return [a, b, self.s8d(b)]


def _no_grad_params(model, x):
    out = model(x)
    sum(v.float().sum() for v in (out.values() if isinstance(out, dict) else [out])).backward()
    return [n for n, p in model.named_parameters() if p.grad is None]


# C1: the deepest level is the LAST one with the largest stride
def test_repeated_deepest_stride_reads_last_level():
    m = create_model(Dilated(), decoder="deeplabv3plus",
                     heads={"seg": {"out_channels": 2},
                            "cls": {"type": "classification", "out_channels": 3}})
    assert m.decoder.high == 2 and m._deepest == 2
    assert m.heads["cls"].fc.in_features == 32
    assert DeepLabV3PlusDecoder(Dilated().feature_specs).high == 2
    assert _no_grad_params(m, torch.randn(2, 3, 32, 32)) == []


# C2: U-Net++ keeps the last feature per stride, so a dilated encoder builds and trains fully
def test_unetplusplus_repeated_deepest_stride():
    specs = Dilated().feature_specs
    dec = UNetPlusPlusDecoder(specs)
    assert dec.read_levels == UNetDecoder(specs).read_levels == [0, 2]
    m = create_model("resnet18", pretrained=False, backbone_kwargs={"output_stride": 16},
                     decoder="unetplusplus", num_classes=2)
    nog = _no_grad_params(m, torch.randn(2, 3, 64, 64))
    assert not [n for n in nog if n.startswith("backbone")], nog


# C3: everything after the target is keyword-only
def test_dice_bce_options_keyword_only():
    x = torch.randn(2, 1, 4, 4)
    t = (torch.rand(2, 1, 4, 4) > 0.5).float()
    m = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    with pytest.raises(TypeError):
        dice_bce(x, t, m)
    params = list(inspect.signature(dice_bce).parameters.values())
    assert all(p.kind is p.KEYWORD_ONLY for p in params[2:])
    torch.testing.assert_close(dice_bce(x, t, mask=m), dice_bce(x, t))


# C4: a partial edge block averages its in-image cells, as downsample_mean does for targets
def test_head_area_average_partial_blocks():
    h = PixelHead(FeatureSpec(1, 1), 1, kernel_size=1, scale=0.125)
    with torch.no_grad():
        h.conv.weight.fill_(1)
        h.conv.bias.zero_()
    x = torch.rand(1, 1, 70, 45)
    torch.testing.assert_close(h(x), downsample_mean(x, 0.125))


# C5: the head grid must be (ceil(H*s), ceil(W*s)), in both directions
def test_upsample_output_checks_grid():
    y = torch.ones(1, 1, 4, 4)
    for mode in ("bilinear", "nearest"):
        assert upsample_output(y, (7, 8), 0.5, mode=mode).shape[-2:] == (7, 8)
        with pytest.raises(ValueError, match="grid"):
            upsample_output(y, (10, 10), 0.5, mode=mode)  # grid too small
        with pytest.raises(ValueError, match="grid"):
            upsample_output(y, (6, 6), 1, mode=mode)  # grid too large


# C6: one raster + one vector input runs the same raster checks as one raster alone
@pytest.mark.parametrize("extra, match", [({"share_norm": True}, "share_norm"),
                                          ({"temporal": "mean"}, "temporal"),
                                          ({"backbone": "shared:meta"}, "shares")])
def test_single_raster_with_vector_runs_raster_checks(extra, match):
    meta = {"kind": "vector", "num_features": 2}
    with pytest.raises(ValueError, match=match):
        create_model("resnet18", pretrained=False, num_classes=2,
                     inputs={"img": {"in_channels": 3, **extra}, "meta": meta})


# C7: decoders and head hidden layers take the same norm values, aliases included
@pytest.mark.parametrize("norm, layer", [("batch", nn.BatchNorm2d), ("bn", nn.BatchNorm2d),
                                         ("group", nn.GroupNorm), ("gn", nn.GroupNorm),
                                         (None, None), ("none", None)])
def test_one_norm_normaliser(norm, layer):
    canon = {"bn": "batch", "gn": "group", "none": None}.get(norm, norm)
    assert normalize_norm(norm) == canon
    block = ConvNormAct(4, 8, norm=norm)
    assert (block[0].bias is None) == (layer is not None)
    m = create_model("resnet18", pretrained=False, decoder_kwargs={"norm": norm},
                     heads={"a": {"out_channels": 2, "hidden": [8], "norm": norm}})
    hidden = m.heads["a"].hidden
    assert (hidden[0].bias is None) == (layer is not None)
    if layer is not None:
        assert isinstance(hidden[1], layer)
        assert any(isinstance(mod, layer) for mod in m.decoder.modules())
    else:
        assert not any(isinstance(mod, nn.BatchNorm2d | nn.GroupNorm)
                       for mod in [*hidden, *m.decoder.modules()])
    for bad in ("layer", "BN"):
        with pytest.raises(ValueError, match="norm"):
            normalize_norm(bad)


# C8: a per-input module backbone accepts the no-op values the top level accepts
def test_per_input_module_backbone_noop_options():
    def bb():
        from manyseg.backbones import TimmBackbone
        return TimmBackbone("resnet18", pretrained=False)

    for extra in ({"pretrained": False}, {"pretrained": True}, {"channel_adapter": "repeat"},
                  {"out_indices": None}, {"backbone_kwargs": {}}):
        create_model(num_classes=2, inputs={"a": {"in_channels": 3, "backbone": bb(), **extra}})
    for extra, match in (({"pretrained": "ssl4eo_s2_dino"}, "pretrained"),
                         ({"channel_adapter": "projection"}, "channel_adapter"),
                         ({"out_indices": [1, 2]}, "out_indices"),
                         ({"backbone_kwargs": {"drop_path_rate": 0.1}}, "backbone_kwargs")):
        with pytest.raises(ValueError, match=match):
            create_model(num_classes=2,
                         inputs={"a": {"in_channels": 3, "backbone": bb(), **extra}})


# C9: a head "decoder" that is not a name, mapping or callable names the head
@pytest.mark.parametrize("bad", [None, 3, ["unet"]])
def test_head_decoder_invalid_type(bad):
    with pytest.raises(ValueError, match="head 'a': 'decoder'"):
        create_model("resnet18", pretrained=False,
                     heads={"a": {"out_channels": 2, "decoder": bad}})


# C10: FiLM 'on' takes one name, and its order does not matter across vector inputs
def test_film_on_string_and_order():
    def build(on_a, on_b=None):
        inputs = {"img": {"in_channels": 3},
                  "a": {"kind": "vector", "num_features": 2,
                        "conditioning": {"mode": "film", "on": on_a}}}
        if on_b is not None:
            inputs["b"] = {"kind": "vector", "num_features": 1,
                           "conditioning": {"mode": "film", "on": on_b}}
        return create_model("resnet18", pretrained=False, num_classes=2, inputs=inputs)

    film = build("output").conditioner.film["decoder"]
    assert film.output is not None and film.levels == ()
    build(["features", "output"], ["output", "features"])
    with pytest.raises(ValueError, match="same 'on'"):
        build(["output"], ["features", "output"])
    with pytest.raises(ValueError, match="film 'on'"):
        build("outputs")
    with pytest.raises(ValueError, match="film 'on'"):
        build([])


def test_model_decoder_invalid_type():
    with pytest.raises(ValueError, match="decoder must be a decoder name"):
        create_model("resnet18", pretrained=False, num_classes=2, decoder=None)
