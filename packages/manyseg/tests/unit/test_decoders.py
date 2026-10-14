"""Common decoder interface, parametrised over every registered decoder."""

import math

import pytest
import torch
from manyseg import FeatureSpec, create_model
from manyseg.decoders import DECODERS

SPECS5 = (FeatureSpec(64, 2), FeatureSpec(64, 4), FeatureSpec(128, 8),
          FeatureSpec(256, 16), FeatureSpec(512, 32))
SPECS4 = SPECS5[1:]
INPUT_SPEC = FeatureSpec(3, 1, "input")
ALL = sorted(DECODERS)


def feats(specs, h, w, n=2, dec=None):
    out = [torch.randn(n, s.channels, math.ceil(h / s.stride), math.ceil(w / s.stride))
           for s in specs]
    if dec is not None and getattr(dec, "wants_input", False):
        out = [torch.randn(n, 3, h, w), *out]
    return out


def build(name, specs, **kw):
    cls = DECODERS[name]
    if getattr(cls, "wants_input", False):
        return cls((INPUT_SPEC, *specs), **kw)
    return cls(specs, **kw)


def test_expected_decoders_registered():
    assert {"unet", "fpn", "deeplabv3plus", "upernet", "segformer", "fastai_unet"} <= set(ALL)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("specs", [SPECS5, SPECS4], ids=["5lvl", "4lvl"])
def test_output_matches_out_spec(name, specs):
    dec = build(name, specs).eval()
    h, w = 64, 96
    out = dec(feats(specs, h, w, dec=dec))
    s = dec.out_spec
    assert out.shape == (2, s.channels, h // s.stride, w // s.stride)


@pytest.mark.parametrize("name", ALL)
def test_odd_feature_sizes(name):
    dec = build(name, SPECS5).eval()
    out = dec(feats(SPECS5, 65, 97, dec=dec))
    s = dec.out_spec
    assert out.shape[-2:] == (math.ceil(65 / s.stride), math.ceil(97 / s.stride)) or \
        out.shape[-2] >= math.ceil(65 / s.stride)


@pytest.mark.parametrize("name", ALL)
def test_gradients_finite_and_reach_all_params(name):
    dec = build(name, SPECS5)
    fs = feats(SPECS5, 64, 64, dec=dec)
    dec(fs).square().mean().backward()
    for n, p in dec.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("name", ALL)
def test_batch_size_one_training(name):
    """No BatchNorm on 1x1 maps (image pooling) that would fail at batch size 1."""
    dec = build(name, SPECS5).train()
    dec(feats(SPECS5, 64, 64, n=1, dec=dec)).mean().backward()


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("backbone", ["resnet18", "convnext_atto", "vit_tiny_patch16_224"])
def test_end_to_end(name, backbone):
    model = create_model(backbone=backbone, pretrained=False, decoder=name, num_classes=3).eval()
    for hw in [(64, 64), (65, 97)]:
        assert model(torch.randn(1, 3, *hw)).shape == (1, 3, *hw)


def test_deeplab_rates_scale_with_output_stride():
    d32 = DECODERS["deeplabv3plus"](SPECS5)
    d16 = DECODERS["deeplabv3plus"]((*SPECS5[:3], FeatureSpec(512, 16)))
    assert d32.atrous_rates == (6, 12, 18)
    assert d16.atrous_rates == (12, 24, 36)


def test_fastai_unet_wider_than_unet():
    n_fa = sum(p.numel() for p in build("fastai_unet", SPECS5).parameters())
    n_u = sum(p.numel() for p in build("unet", SPECS5).parameters())
    assert n_fa > 4 * n_u
