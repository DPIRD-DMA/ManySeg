"""Regression tests for the core sweep: fixed-size padding (single input and early fusion),
backbone_kwargs rgb_indices, module backbones, head norm names, classification scope default,
per-input heads with one raster input, image heads pooling the valid region, odd head kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from manyseg import create_model
from manyseg.features import FeatureSpec
from manyseg.heads import PixelHead
from torch import nn


class Tiny(nn.Module):
    """Two conv stages (strides 4, 8); optional fixed input size."""

    input_divisor = 8

    def __init__(self, in_channels: int = 3, fixed_size=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.fixed_size = fixed_size
        self.feature_specs = (FeatureSpec(8, 4), FeatureSpec(16, 8))
        self.s4 = nn.Sequential(nn.Conv2d(in_channels, 8, 4, stride=4), nn.ReLU())
        self.s8 = nn.Sequential(nn.Conv2d(8, 16, 2, stride=2), nn.ReLU())

    def forward(self, x):
        if self.fixed_size is not None:
            assert tuple(x.shape[-2:]) == tuple(self.fixed_size), x.shape
        a = self.s4(x)
        return [a, self.s8(a)]


def _zero_pad(x, size):
    return F.pad(x, (0, size - x.shape[-1], 0, size - x.shape[-2]))


# B1: a fixed-size backbone crops its outputs whatever pad_to_stride says
def test_fixed_size_backbone_crops_without_pad_to_stride():
    torch.manual_seed(0)
    bb = Tiny(fixed_size=(32, 32))
    a = create_model(bb, num_classes=2, decoder="fpn", pad_to_stride=True).eval()
    b = create_model(bb, num_classes=2, decoder="fpn", pad_to_stride=False).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 3, 20, 27)
    with torch.no_grad():
        ya, yb = a(x), b(x)
        full = a(_zero_pad(x, 32))
    assert yb.shape == (1, 2, 20, 27)
    torch.testing.assert_close(ya, yb)
    torch.testing.assert_close(yb, full[..., :20, :27])  # cropped, not resized


# B2: early fusion pads to the wrapped backbone's fixed size, and crops
def test_early_fusion_pads_to_fixed_size():
    torch.manual_seed(0)
    m = create_model(Tiny(5, fixed_size=(32, 32)), num_classes=2, decoder="fpn",
                     inputs={"a": {"in_channels": 3}, "b": {"in_channels": 2}},
                     fusion={"type": "early"}).eval()
    x = {"a": torch.randn(1, 3, 20, 27), "b": torch.randn(1, 2, 20, 27)}
    with torch.no_grad():
        y = m(x)
        full = m({k: _zero_pad(v, 32) for k, v in x.items()})
    assert y.shape == (1, 2, 20, 27)
    torch.testing.assert_close(y, full[..., :20, :27])
    with pytest.raises(ValueError, match="fixed input size"):
        m({k: torch.randn(1, v.shape[1], 40, 40) for k, v in x.items()})


# B3: rgb_indices through backbone_kwargs with inputs=; a per-input one wins
def test_rgb_indices_in_backbone_kwargs_with_inputs():
    kw = dict(backbone="resnet18", pretrained=False, num_classes=2,
              channel_adapter="rgb_plus_zero")
    m = create_model(inputs={"s2": {"in_channels": 4}}, backbone_kwargs={"rgb_indices": [2, 1, 0]},
                     **kw)
    assert m.backbone.encoders["s2"].rgb_indices == (2, 1, 0)
    m = create_model(inputs={"s2": {"in_channels": 4, "rgb_indices": [0, 1, 2]}},
                     backbone_kwargs={"rgb_indices": [2, 1, 0]}, **kw)
    assert m.backbone.encoders["s2"].rgb_indices == (0, 1, 2)
    with pytest.raises(ValueError, match="both"):
        create_model(inputs={"s2": {"in_channels": 4, "rgb_indices": [0, 1, 2],
                                    "backbone_kwargs": {"rgb_indices": [2, 1, 0]}}}, **kw)


# B4: a module backbone rejects timm options instead of ignoring them
@pytest.mark.parametrize("kw", [{"out_indices": (1, 2)}, {"channel_adapter": "mean"},
                                {"backbone_kwargs": {"drop_path_rate": 0.1}},
                                {"pretrained": "ssl4eo_s2_dino"}, {"in_channels": 4}])
def test_module_backbone_rejects_timm_options(kw):
    with pytest.raises(ValueError, match="nn.Module|channels"):
        create_model(Tiny(), num_classes=2, decoder="fpn", **kw)


def test_module_backbone_accepts_matching_options():
    create_model(Tiny(), num_classes=2, decoder="fpn", in_channels=3, pretrained=False)


def test_module_backbone_with_inputs_checks_channels_up_front():
    with pytest.raises(ValueError, match="4 channels.*takes 3"):
        create_model(Tiny(), num_classes=2, decoder="fpn",
                     inputs={"a": {"in_channels": 3}, "b": {"in_channels": 4}})
    with pytest.raises(ValueError, match="configure a timm backbone"):
        create_model(Tiny(), num_classes=2, decoder="fpn",
                     inputs={"a": {"in_channels": 3, "out_indices": (1,)}})
    with pytest.raises(ValueError, match="configure a timm backbone"):
        create_model(Tiny(), num_classes=2, decoder="fpn", out_indices=(1,),
                     inputs={"a": {"in_channels": 3}})
    # one module for several inputs: allowed (documented), the same object
    bb = Tiny()
    m = create_model(bb, num_classes=2, decoder="fpn",
                     inputs={"a": {"in_channels": 3}, "b": {"in_channels": 3}})
    assert m.backbone.encoders["a"] is m.backbone.encoders["b"] is bb
    # with a vector input too (the single-raster path)
    vec = {"kind": "vector", "num_features": 2}
    with pytest.raises(ValueError, match="4 channels.*takes 3"):
        create_model(Tiny(), num_classes=2, decoder="fpn",
                     inputs={"a": {"in_channels": 4}, "meta": vec})
    create_model(Tiny(), num_classes=2, decoder="fpn",
                 inputs={"a": {"in_channels": 3}, "meta": vec})


# X1: heads take the decoders' norm names, with the decoders' group rule
def test_head_norm_names_match_decoders():
    spec = FeatureSpec(16, 1)
    assert isinstance(PixelHead(spec, 1, hidden=[8])
                      .hidden[1], nn.BatchNorm2d)  # default "batch"
    for name, alias in (("batch", "bn"), ("group", "gn")):
        a = PixelHead(spec, 1, hidden=[48], norm=name).hidden[1]
        b = PixelHead(spec, 1, hidden=[48], norm=alias).hidden[1]
        assert type(a) is type(b)
    assert PixelHead(spec, 1, hidden=[48], norm="group").hidden[1].num_groups == 24
    with pytest.raises(ValueError, match="'batch', 'group'"):
        PixelHead(spec, 1, hidden=[8], norm="layer")


# X3: classification defaults to image scope
def test_classification_defaults_to_image_scope():
    m = create_model(Tiny(), heads={"cls": {"type": "classification", "out_channels": 3}})
    assert m.heads["cls"].scope == "image"
    assert m(torch.randn(2, 3, 32, 32))["cls"].shape == (2, 3)
    with pytest.raises(ValueError, match="image-scope"):
        create_model(Tiny(), heads={"cls": {"type": "classification", "out_channels": 3,
                                            "scope": "pixel"}})


# X5: "input" naming the one raster input next to a vector input
def test_input_head_with_one_raster_and_a_vector_input():
    vec = {"kind": "vector", "num_features": 2}
    inputs = {"s2": {"in_channels": 3}, "meta": vec}
    m = create_model(Tiny(), decoder="fpn", inputs=inputs,
                     heads={"seg": {"out_channels": 2, "input": "s2"}}).eval()
    assert "seg" in m.task_decoders  # its own decoder, as a per-input head has
    y = m({"s2": torch.randn(1, 3, 32, 32), "meta": torch.randn(1, 2)})["seg"]
    assert y.shape == (1, 2, 32, 32)
    with pytest.raises(ValueError, match="vector"):
        create_model(Tiny(), decoder="fpn", inputs=inputs,
                     heads={"seg": {"out_channels": 2, "input": "meta"}})
    with pytest.raises(ValueError, match="unknown input"):
        create_model(Tiny(), decoder="fpn", inputs=inputs,
                     heads={"seg": {"out_channels": 2, "input": "s1"}})


# A2: image heads pool over the valid region only
@pytest.mark.parametrize("source", ["encoder", "decoder"])
def test_image_head_ignores_padding(source):
    torch.manual_seed(0)
    m = create_model(Tiny(), decoder="fpn",
                     heads={"cls": {"type": "classification", "out_channels": 3,
                                    "source": source}}).eval()
    x = torch.randn(1, 3, 20, 27)  # padded to 24 x 32
    with torch.no_grad():
        y = m(x)["cls"]
        if source == "encoder":
            f, stride = m.features(x)[-1], m.feature_specs[-1].stride
        else:
            f, stride = m.embed(x, source="decoder"), m.decoder.out_spec.stride
        expect = m.heads["cls"].fc(f.mean(dim=(-2, -1)))
    assert f.shape[-2:] == (-(-20 // stride), -(-27 // stride))  # the valid region
    torch.testing.assert_close(y, expect)


# A3: even head kernels are rejected
@pytest.mark.parametrize("k", [2, 4, 0])
def test_even_head_kernel_rejected(k):
    with pytest.raises(ValueError, match="odd"):
        PixelHead(FeatureSpec(8, 1), 1, kernel_size=k)
    with pytest.raises(ValueError, match="odd"):
        create_model(Tiny(), heads={"s": {"out_channels": 1, "kernel_size": k}})


def test_image_head_valid_region_exports_with_dynamic_sizes():
    torch.manual_seed(0)
    m = create_model(Tiny(), decoder="fpn",
                     heads={"seg": {"out_channels": 2},
                            "cls": {"type": "classification", "out_channels": 3},
                            "cls_d": {"type": "classification", "out_channels": 3,
                                      "source": "decoder"}}).eval()
    h, w = torch.export.Dim("h", min=16, max=256), torch.export.Dim("w", min=16, max=256)
    ep = torch.export.export(m, (torch.randn(1, 3, 40, 48),),
                             dynamic_shapes={"x": {2: h, 3: w}})
    for hw in [(40, 48), (37, 51), (17, 64)]:
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            got, ref = ep.module()(x), m(x)
        for k in ref:
            torch.testing.assert_close(got[k], ref[k], atol=1e-5, rtol=1e-5)
