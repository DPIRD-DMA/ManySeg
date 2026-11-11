"""Regression tests for the second core sweep: categorical ids range-checked after rounding,
the concat temporal aggregation zeroing all-masked samples, late fusion cropping a fixed-size
input's output, 'share' heads with their own decoder, tiling_spec max_size across inputs, the
metadata presence-mask shape, shared module backbones without in_channels, decoder output
strides, and the head option checks running the scope check first."""

from __future__ import annotations

import pytest
import torch
from manyseg import create_model
from manyseg.conditioning import MetadataEncoder
from manyseg.conditioning.fields import CategoricalField
from manyseg.features import FeatureSpec
from manyseg.heads.heads import create_head
from manyseg.temporal import TemporalEncoder
from torch import nn


class Tiny(nn.Module):
    """Two conv stages (strides 4, 8); optional fixed input size and in_channels."""

    input_divisor = 8

    def __init__(self, in_channels: int | None = 1, fixed_size=None) -> None:
        super().__init__()
        if in_channels is not None:
            self.in_channels = in_channels
        self.fixed_size = fixed_size
        self.feature_specs = (FeatureSpec(8, 4), FeatureSpec(8, 8))
        self.s4 = nn.Conv2d(in_channels or 1, 8, 4, stride=4)
        self.s8 = nn.Conv2d(8, 8, 2, stride=2)

    def forward(self, x):
        if self.fixed_size is not None:
            assert tuple(x.shape[-2:]) == tuple(self.fixed_size), x.shape
        a = self.s4(x)
        return [a, self.s8(a)]


# C1: an integer-valued id that rounds up to num is out of range (was an IndexError)
def test_categorical_range_checks_rounded_id():
    f = CategoricalField(3)
    v = torch.tensor([[2.9999995], [3.0000004], [-0.0000004], [2.0]])
    _, valid = f(v)
    assert valid.tolist() == [False, False, True, True]


# C2: 'concat' gives zero features for a sample whose dates are all masked
@pytest.mark.parametrize("train", [True, False])
def test_concat_aggregation_all_masked_is_zero(train):
    torch.manual_seed(0)

    class BB(nn.Module):
        input_divisor = 4
        feature_specs = (FeatureSpec(8, 4),)

        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 4, 4)

        def forward(self, x):
            return [self.c(x)]

    te = TemporalEncoder(BB(), "concat", {"dates": 2}).train(train)
    if not train:
        te.temporal_aggregation[0].fuse[1].running_mean.fill_(-1.0)
    mask = torch.tensor([[True, True], [False, False]])
    out = te(torch.randn(2, 2, 3, 8, 8), mask)[0]
    assert out[1].abs().max() == 0
    assert out[0].abs().max() > 0


# C3: late fusion crops a fixed-size same_grid input's padded output, like intermediate fusion
def test_late_fusion_crops_fixed_size_same_grid_input():
    torch.manual_seed(0)
    q = {"in_channels": 1, "backbone": Tiny(fixed_size=(64, 64))}
    m = create_model(Tiny(), inputs={"p": {"in_channels": 1}, "q": q},
                     fusion={"type": "late", "op": "add"}, num_classes=1, decoder="fpn").eval()
    cap = {}
    m.decoder.decoders[1].register_forward_hook(lambda mod, a, o: cap.__setitem__("q", o))
    m.decoder.fuse.register_forward_pre_hook(lambda mod, a: cap.__setitem__("in", a[0]))
    with torch.no_grad():
        m({"p": torch.zeros(1, 1, 32, 32), "q": torch.randn(1, 1, 32, 32)})
    q, fused_q = cap["q"], cap["in"][1]
    assert q.shape[-2:] == (16, 16) and fused_q.shape[-2:] == (8, 8)
    torch.testing.assert_close(fused_q, q[..., :8, :8])


# C3: with the primary fixed-size instead, the other input's output is zero-padded, not resized
def test_late_fusion_pads_when_primary_is_fixed_size():
    torch.manual_seed(0)
    m = create_model(Tiny(fixed_size=(64, 64)), inputs={"p": {"in_channels": 1},
                                                        "q": {"in_channels": 1,
                                                              "backbone": Tiny()}},
                     fusion={"type": "late", "op": "add"}, num_classes=1, decoder="fpn").eval()
    cap = {}
    m.decoder.decoders[1].register_forward_hook(lambda mod, a, o: cap.__setitem__("q", o))
    m.decoder.fuse.register_forward_pre_hook(lambda mod, a: cap.__setitem__("in", a[0]))
    with torch.no_grad():
        y = m({"p": torch.zeros(1, 1, 32, 32), "q": torch.randn(1, 1, 32, 32)})
    q, fused_q = cap["q"], cap["in"][1]
    assert y.shape[-2:] == (32, 32)
    assert fused_q.shape[-2:] == (16, 16)
    torch.testing.assert_close(fused_q[..., :8, :8], q)
    assert fused_q[..., 8:, :].abs().max() == 0 and fused_q[..., :, 8:].abs().max() == 0


def _per_input_heads(**b_extra):
    return dict(
        inputs={"pre": {"in_channels": 1, "backbone": Tiny()},
                "post": {"in_channels": 1, "backbone": "shared:pre"}},
        heads={"a": {"out_channels": 2, "input": "pre"},
               "b": {"out_channels": 2, "input": "post", "share": "a", **b_extra},
               "chg": {"out_channels": 1}})


# C4: a 'share' head cannot set a decoder of its own (it was dropped silently)
def test_share_head_with_different_decoder_raises():
    with pytest.raises(ValueError, match="cannot set a different 'decoder'"):
        create_model(decoder="unet", **_per_input_heads(decoder="fpn"))


def test_share_head_with_same_decoder_is_accepted():
    m = create_model(decoder="unet", **_per_input_heads(decoder="unet"))
    assert m.task_decoders["b"] is m.task_decoders["a"]
    m = create_model(decoder="unet", **_per_input_heads(decoder={"type": "unet"}))
    assert m.task_decoders["b"] is m.task_decoders["a"]


# C5: max_size is the most restrictive fixed size over the primary and same_grid inputs
def test_tiling_spec_max_size_includes_non_primary_inputs():
    q = {"in_channels": 1, "backbone": Tiny(fixed_size=(64, 48))}
    m = create_model(Tiny(), inputs={"p": {"in_channels": 1}, "q": q}, num_classes=1)
    spec = m.tiling_spec()
    assert spec["max_size"] == (64, 48)
    with pytest.raises(ValueError, match="fixed input size"):
        m({"p": torch.zeros(1, 1, 64, 64), "q": torch.zeros(1, 1, 64, 64)})
    m = create_model(Tiny(fixed_size=(40, 96)),
                     inputs={"p": {"in_channels": 1},
                             "q": {"in_channels": 1, "backbone": Tiny(fixed_size=(64, 48))}},
                     num_classes=1)
    assert m.tiling_spec()["max_size"] == (40, 48)


def test_tiling_spec_same_extent_fixed_size_left_out():
    # ManyPred's spec validation allows only {kind, scale} per input; the same_extent input's
    # limit (in its own pixels, ratio unknown until the data arrives) is enforced at forward
    m = create_model(Tiny(), inputs={"p": {"in_channels": 1},
                                     "q": {"in_channels": 1, "alignment": "same_extent",
                                           "backbone": Tiny(fixed_size=(64, 64))}},
                     num_classes=1)
    spec = m.tiling_spec()
    assert spec["max_size"] is None
    assert all(set(e) == {"kind", "scale"} for e in spec["inputs"].values())
    with pytest.raises(ValueError, match="fixed input size"):
        m({"p": torch.zeros(1, 1, 32, 32), "q": torch.zeros(1, 1, 96, 96)})


# C6: the field-presence mask must be (B, F), not (B, columns)
def test_metadata_mask_shape_is_checked():
    enc = MetadataEncoder({"loc": {"type": "coords"}, "t": {"type": "continuous"}}).eval()
    v = torch.tensor([[10.0, 20.0, 1.0]])
    assert enc(v, torch.tensor([[1.0, 0.0]])).shape == (1, 64)
    with pytest.raises(ValueError, match=r"\(B, F\) = \(1, 2\)"):
        enc(v, torch.tensor([[1.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="presence mask"):
        enc(v, torch.tensor([1.0, 1.0]))


# C7: 'shared:<input>' with a module backbone that has no in_channels attribute
def test_shared_module_backbone_without_in_channels():
    m = create_model(Tiny(in_channels=None),
                     inputs={"p": {"in_channels": 1}, "q": {"in_channels": 1,
                                                            "backbone": "shared:p"}},
                     num_classes=1)
    assert m({"p": torch.zeros(1, 1, 16, 16), "q": torch.zeros(1, 1, 16, 16)}).shape == \
        (1, 1, 16, 16)
    with pytest.raises(ValueError, match="channels"):
        create_model(Tiny(in_channels=None),
                     inputs={"p": {"in_channels": 1}, "q": {"in_channels": 2,
                                                            "backbone": "shared:p"}},
                     num_classes=1)


# C7 (same pattern): a decoder reading the input image with a module lacking in_channels
def test_input_reading_decoder_needs_module_in_channels():
    with pytest.raises(ValueError, match="no 'in_channels' attribute"):
        create_model(Tiny(in_channels=None), num_classes=1, decoder="fastai_unet")


# C8: the fixed-stride decoders output at the shallowest level they read, not always 4
@pytest.mark.parametrize("decoder", ["fpn", "segformer", "upernet", "deeplabv3plus"])
def test_fixed_stride_decoder_stride_follows_shallowest_level(decoder):
    class S8(Tiny):
        def __init__(self):
            super().__init__()
            self.feature_specs = (FeatureSpec(8, 8), FeatureSpec(8, 16))

    m = create_model(S8(), num_classes=1, decoder=decoder)
    assert m.decoder.out_spec.stride == 8


# C9: an image head given norm/dilation is told they are pixel-only, not "give hidden too"
@pytest.mark.parametrize("key", ["norm", "dilation"])
def test_image_head_norm_dilation_scope_first(key):
    spec = FeatureSpec(8, 4)
    cfg = {"type": "classification", "out_channels": 2, key: "batch" if key == "norm" else 2}
    with pytest.raises(ValueError, match="pixel heads only"):
        create_head(spec, spec, cfg)
    with pytest.raises(ValueError, match="pixel heads only"):  # adding hidden: same answer
        create_head(spec, spec, {**cfg, "hidden": 8})
    with pytest.raises(ValueError, match="pixel heads only"):  # activation on an image head
        create_head(spec, spec, {"type": "classification", "out_channels": 2,
                                 "activation": "sigmoid"})
    with pytest.raises(ValueError, match="give 'hidden' too"):  # pixel heads still ask
        create_head(spec, spec, {"out_channels": 2, key: cfg[key]})
