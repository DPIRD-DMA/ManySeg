"""Backbone wrapper contracts: feature specs, layout, channel adaptation, dynamic sizes."""

import math

import pytest
import timm
import torch
from manyseg import FeatureSpec
from manyseg.backbones import create_backbone

CNNS = ["resnet18", "efficientnet_b0", "convnext_atto", "mobilenetv3_small_100", "regnetx_002"]


@pytest.mark.parametrize("name", CNNS)
def test_feature_specs_match_tensors(name):
    bb = create_backbone(name, pretrained=False).eval()
    feats = bb(torch.randn(2, 3, 128, 96))
    assert len(feats) == len(bb.feature_specs)
    for f, spec in zip(feats, bb.feature_specs):
        assert isinstance(spec, FeatureSpec)
        assert f.shape[:2] == (2, spec.channels)
        assert f.shape[-2:] == (128 // spec.stride, 96 // spec.stride)
    strides = [s.stride for s in bb.feature_specs]
    assert strides == sorted(strides) and len(set(strides)) == len(strides)


@pytest.mark.parametrize("c", [1, 2, 4, 13])
def test_arbitrary_in_channels(c):
    bb = create_backbone("resnet18", pretrained=False, in_channels=c)
    feats = bb(torch.randn(1, c, 64, 64))
    assert feats[-1].shape[1] == bb.feature_specs[-1].channels
    assert bb.in_channels == c


def test_wrong_channel_count_raises():
    bb = create_backbone("resnet18", pretrained=False, in_channels=4)
    with pytest.raises(RuntimeError):
        bb(torch.randn(1, 3, 64, 64))


def test_input_divisor_is_max_stride():
    bb = create_backbone("resnet18", pretrained=False)
    assert bb.input_divisor == 32


def test_out_indices_selects_levels():
    bb = create_backbone("resnet18", pretrained=False, out_indices=(1, 2, 3, 4))
    assert [s.stride for s in bb.feature_specs] == [4, 8, 16, 32]


def test_exposes_normalisation_metadata():
    bb = create_backbone("resnet18", pretrained=False)
    assert len(bb.mean) == 3 and len(bb.std) == 3


def test_unknown_backbone_error_is_helpful():
    with pytest.raises(ValueError, match="resnet18"):
        create_backbone("resnet18x", pretrained=False)


def _stem_weight(bb):
    return bb.stem_conv().weight


@pytest.mark.parametrize("method", ["repeat", "mean", "rgb_plus_random", "rgb_plus_zero"])
def test_stem_weights_adapted_from_pretrained(method, monkeypatch):
    """The adapter is applied to the pretrained RGB stem (simulated without network)."""
    ref = timm.create_model("resnet18", pretrained=False, features_only=True)
    sd = ref.state_dict()

    import manyseg.backbones.timm_backbone as tb

    def fake_create(name, pretrained=False, **kw):
        m = timm.create_model(name, pretrained=False, **kw)
        if pretrained:
            m.load_state_dict(sd)
        return m

    monkeypatch.setattr(tb, "_timm_create", fake_create)
    bb = create_backbone("resnet18", pretrained=True, in_channels=5, channel_adapter=method)
    from manyseg.adapters import adapt_conv_weight

    expected = adapt_conv_weight(sd["conv1.weight"], 5, method)
    if method == "rgb_plus_random":
        torch.testing.assert_close(_stem_weight(bb)[:, :3], expected[:, :3])
    else:
        torch.testing.assert_close(_stem_weight(bb), expected)
    # everything else untouched
    other = {k: v for k, v in bb.model.state_dict().items() if k != "conv1.weight"}
    for k, v in other.items():
        torch.testing.assert_close(v, sd[k])


def test_projection_adapter_initially_matches_rgb_model():
    bb3 = create_backbone("resnet18", pretrained=False).eval()
    bb5 = create_backbone("resnet18", pretrained=False, in_channels=5, channel_adapter="projection")
    bb5.model.load_state_dict(bb3.model.state_dict())
    bb5.eval()
    x = torch.randn(1, 5, 64, 64)
    for a, b in zip(bb5(x), bb3(x[:, :3])):
        torch.testing.assert_close(a, b)


def test_projection_adds_few_parameters():
    n3 = sum(p.numel() for p in create_backbone("resnet18", pretrained=False).parameters())
    n5 = sum(
        p.numel()
        for p in create_backbone(
            "resnet18", pretrained=False, in_channels=5, channel_adapter="projection"
        ).parameters()
    )
    assert 0 < n5 - n3 <= 5 * 3 + 3


@pytest.mark.parametrize("hw", [(64, 64), (65, 97), (33, 200), (127, 31)])
def test_odd_sizes_produce_ceil_feature_maps(hw):
    bb = create_backbone("resnet18", pretrained=False).eval()
    feats = bb(torch.randn(1, 3, *hw))
    for f, s in zip(feats, bb.feature_specs):
        assert f.shape[-2:] == (math.ceil(hw[0] / s.stride), math.ceil(hw[1] / s.stride))


@pytest.mark.slow  # MaxViT on CPU
def test_capability_probe_reports_fixed_size_backbones():
    """Fixed-size backbones run the small probe sizes (padded up), but are not 'any size'."""
    from manyseg.backbones.capabilities import probe_backbone

    caps = probe_backbone("maxvit_tiny_tf_224")
    assert caps.size_support == "fixed"
    assert "224" in caps.error
    assert probe_backbone("resnet18").size_support == "any"
