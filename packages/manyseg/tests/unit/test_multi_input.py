"""Multi-input models: early / intermediate / late fusion, alignment policies, sharing."""

import pytest
import torch
from manyseg import create_model
from manyseg.fusion import FUSION_OPS

TWO = {
    "optical": {"in_channels": 4, "backbone": "resnet18"},
    "sar": {"in_channels": 2, "backbone": "convnext_atto"},
}


def make(inputs=TWO, fusion=None, **kw):
    kw.setdefault("num_classes", 3)
    return create_model(inputs=inputs, fusion=fusion or {"type": "intermediate"},
                        pretrained=False, **kw)


def batch(h=64, w=96, n=2, sar_hw=None):
    return {"optical": torch.randn(n, 4, h, w), "sar": torch.randn(n, 2, *(sar_hw or (h, w)))}


def test_intermediate_default_concat():
    model = make().eval()
    assert model(batch()).shape == (2, 3, 64, 96)


@pytest.mark.parametrize("op", sorted(FUSION_OPS))
def test_every_fusion_op(op):
    levels = [16, 32] if op == "cross_attention" else None
    model = make(fusion={"type": "intermediate", "op": op, "strides": levels}).eval()
    assert model(batch(65, 97)).shape == (2, 3, 65, 97)


def test_fusion_strides_subset():
    model = make(fusion={"type": "intermediate", "op": "add", "strides": [16, 32]})
    assert sorted(model.backbone.fused_strides) == [16, 32]


def test_primary_only_levels_when_secondary_lacks_stride():
    """resnet18 has a stride-2 level, convnext_atto does not: stride 2 uses primary only."""
    model = make()
    assert 2 not in model.backbone.fused_strides
    assert {4, 8, 16, 32} <= set(model.backbone.fused_strides)


def test_early_fusion_single_backbone():
    model = make(fusion={"type": "early"}, inputs={
        "optical": {"in_channels": 4}, "sar": {"in_channels": 2}}, backbone="resnet18").eval()
    assert model.backbone.backbone.in_channels == 6
    assert model(batch()).shape == (2, 3, 64, 96)


def test_early_fusion_rejects_mismatched_sizes():
    model = make(fusion={"type": "early"}, inputs={
        "optical": {"in_channels": 4}, "sar": {"in_channels": 2}}, backbone="resnet18").eval()
    with pytest.raises(ValueError, match="same_grid"):
        model(batch(sar_hw=(32, 48)))


@pytest.mark.parametrize("op", ["concat", "add", "gated"])
def test_late_fusion(op):
    model = make(fusion={"type": "late", "op": op}).eval()
    assert model(batch(65, 97)).shape == (2, 3, 65, 97)
    assert len(model.decoder.decoders) == 2


def test_same_grid_default_rejects_mismatch():
    model = make().eval()
    with pytest.raises(ValueError, match="same_grid"):
        model(batch(sar_hw=(32, 48)))


def test_same_extent_lower_resolution_input():
    inputs = {"optical": {"in_channels": 4, "backbone": "resnet18"},
              "sar": {"in_channels": 2, "backbone": "resnet18", "alignment": "same_extent"}}
    model = make(inputs=inputs).eval()
    assert model(batch(64, 96, sar_hw=(32, 48))).shape == (2, 3, 64, 96)
    assert model(batch(65, 97, sar_hw=(33, 49))).shape == (2, 3, 65, 97)


def _centroid(grad):
    g = grad.abs().sum(dim=(0, 1))
    ys = torch.arange(g.shape[0], dtype=g.dtype)[:, None]
    xs = torch.arange(g.shape[1], dtype=g.dtype)[None, :]
    return float((g * ys).sum() / g.sum()), float((g * xs).sum() / g.sum())


def test_same_extent_is_spatially_aligned():
    """Output pixel (i, j) depends on the half-resolution input around (i/2, j/2)."""
    torch.manual_seed(0)
    inputs = {"optical": {"in_channels": 1, "backbone": "resnet18"},
              "sar": {"in_channels": 1, "backbone": "resnet18", "alignment": "same_extent"}}
    model = make(inputs=inputs, fusion={"type": "intermediate", "op": "add"},
                 decoder_kwargs={"upsample": "aligned"}).eval()
    errs = []
    for i, j in [(70, 70), (100, 60), (60, 110)]:
        x = {"optical": torch.randn(1, 1, 160, 160),
             "sar": torch.randn(1, 1, 80, 80, requires_grad=True)}
        model(x)[0, :, i, j].sum().backward()
        cy, cx = _centroid(x["sar"].grad)
        errs.append(max(abs(cy - i / 2), abs(cx - j / 2)))
    assert max(errs) < 3, errs


def test_shared_encoder_reuses_weights():
    inputs = {"t0": {"in_channels": 3, "backbone": "resnet18"},
              "t1": {"in_channels": 3, "backbone": "shared:t0"}}
    model = make(inputs=inputs, fusion={"type": "intermediate", "op": "add"})
    enc = model.backbone.encoders
    # Weights are shared (the same Parameter objects); BatchNorm is per input (E015).
    assert enc["t0"].stem_conv().weight is enc["t1"].stem_conv().weight
    def bn(e):
        return [m for m in e.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    assert bn(enc["t0"]) and all(a is not b for a, b in zip(bn(enc["t0"]), bn(enc["t1"])))
    n_shared = sum(p.numel() for p in model.parameters())
    single = create_model(backbone="resnet18", pretrained=False, num_classes=3)
    assert n_shared < 1.5 * sum(p.numel() for p in single.parameters())
    out = model({"t0": torch.randn(1, 3, 64, 64), "t1": torch.randn(1, 3, 64, 64)})
    assert out.shape == (1, 3, 64, 64)


def test_shared_encoder_different_channels_uses_projection():
    inputs = {"a": {"in_channels": 3, "backbone": "resnet18"},
              "b": {"in_channels": 5, "backbone": "shared:a", "channel_adapter": "projection"}}
    model = make(inputs=inputs).eval()
    assert model({"a": torch.randn(1, 3, 64, 64), "b": torch.randn(1, 5, 64, 64)}).shape == \
        (1, 3, 64, 64)


SHARED_A = {"in_channels": 3, "backbone": "resnet18"}


@pytest.mark.parametrize("top_level", ["repeat", "projection"])
def test_a_shared_input_with_other_channels_must_ask_for_the_projection(top_level):
    """Regression: a learned layer used to appear silently. Only the input's own setting counts."""
    inputs = {"a": SHARED_A, "b": {"in_channels": 5, "backbone": "shared:a"}}
    with pytest.raises(ValueError, match=r"'channel_adapter': 'projection'"):
        make(inputs=inputs, channel_adapter=top_level)


@pytest.mark.parametrize("adapter", ["repeat", "mean", "rgb_plus_zero", "rgb_plus_random"])
@pytest.mark.parametrize("channels", [3, 5])
def test_a_shared_input_refuses_other_adapters(adapter, channels):
    """Regression: these were silently ignored ("mean" still gave a projection)."""
    inputs = {"a": SHARED_A,
              "b": {"in_channels": channels, "backbone": "shared:a", "channel_adapter": adapter}}
    with pytest.raises(ValueError, match="only 'projection'"):
        make(inputs=inputs)


def test_a_shared_input_with_equal_channels_refuses_the_projection():
    inputs = {"a": SHARED_A,
              "b": {"in_channels": 3, "backbone": "shared:a", "channel_adapter": "projection"}}
    with pytest.raises(ValueError, match="nothing to adapt"):
        make(inputs=inputs)


def test_a_shared_input_with_equal_channels_needs_no_setting_whatever_the_top_level():
    inputs = {"a": SHARED_A, "b": {"in_channels": 3, "backbone": "shared:a"}}
    for top_level in ("repeat", "mean", "projection"):
        assert len(make(inputs=inputs, channel_adapter=top_level).backbone.adapters) == 0


def test_a_chain_of_shared_inputs_compares_with_the_encoders_own_channels():
    inputs = {"a": SHARED_A,
              "b": {"in_channels": 5, "backbone": "shared:a", "channel_adapter": "projection"},
              "c": {"in_channels": 3, "backbone": "shared:b"}}  # same as the encoder, not as b
    assert set(make(inputs=inputs).backbone.adapters) == {"b"}


def test_share_norm_with_other_channels_needs_the_projection_too():
    inputs = {"a": SHARED_A, "b": {"in_channels": 5, "backbone": "shared:a", "share_norm": True}}
    with pytest.raises(ValueError, match=r"'channel_adapter': 'projection'"):
        make(inputs=inputs)
    inputs["b"]["channel_adapter"] = "projection"
    make(inputs=inputs)


def test_metadata_channels_can_make_a_shared_input_differ():
    """'channels' metadata widens its target before create_model compares the two."""
    meta = {"kind": "vector", "num_features": 2, "conditioning": "channels"}
    inputs = {"a": SHARED_A, "b": {"in_channels": 3, "backbone": "shared:a"}, "meta": meta}
    with pytest.raises(ValueError, match=r"'channel_adapter': 'projection'"):
        make(inputs=inputs)
    inputs["b"]["channel_adapter"] = "projection"
    assert set(make(inputs=inputs).backbone.adapters) == {"b"}


def test_gradients_reach_both_encoders():
    model = make()
    model(batch()).mean().backward()
    for name in ("optical", "sar"):
        grads = [p.grad for p in model.backbone.encoders[name].parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name


def test_missing_input_is_clear():
    model = make().eval()
    with pytest.raises(KeyError, match="sar"):
        model({"optical": torch.randn(1, 4, 64, 64)})


def test_tensor_input_rejected_for_multi_input_model():
    with pytest.raises(TypeError, match="dict"):
        make().eval()(torch.randn(1, 4, 64, 64))


def test_multi_input_multi_head():
    model = make(num_classes=None, heads={
        "seg": {"type": "segmentation", "out_channels": 2},
        "area": {"type": "regression", "out_channels": 1, "scope": "image"}}).eval()
    out = model(batch())
    assert out["seg"].shape == (2, 2, 64, 96) and out["area"].shape == (2, 1)


def test_unknown_fusion_type():
    with pytest.raises(ValueError, match="intermediate"):
        make(fusion={"type": "nope"})


def test_top_level_backbone_is_default_for_inputs():
    model = make(inputs={"a": {"in_channels": 3}, "b": {"in_channels": 1}}, backbone="resnet18")
    enc = model.backbone.encoders
    assert enc["a"] is not enc["b"]
    assert enc["a"].name == enc["b"].name == "resnet18"


# --- construction bugs found by model.describe() (2026-10-15) ---------------------------------

def test_early_fusion_uses_agreeing_per_input_encoder_keys():
    """Regression: per-input encoder keys were silently dropped (resnet18 built instead)."""
    same = {"in_channels": 3, "backbone": "convnext_atto"}
    model = make(fusion={"type": "early"}, backbone="resnet18",
                 inputs={"a": dict(same), "b": {**same, "in_channels": 2}})
    assert model.backbone.backbone.name == "convnext_atto"


@pytest.mark.parametrize("key,va,vb", [("backbone", "convnext_atto", "resnet18"),
                                       ("channel_adapter", "repeat", "mean"),
                                       ("pretrained", True, False)])
def test_early_fusion_rejects_conflicting_per_input_encoder_keys(key, va, vb):
    with pytest.raises(ValueError, match="early fusion"):
        make(fusion={"type": "early"}, backbone="resnet18",
             inputs={"a": {"in_channels": 3, key: va}, "b": {"in_channels": 2, key: vb}})


@pytest.mark.parametrize("key,value", [("rgb_indices", [0, 1, 2]),
                                       ("alignment", "same_extent")])
def test_early_fusion_rejects_per_input_only_keys(key, value):
    with pytest.raises(ValueError, match="early fusion"):
        make(fusion={"type": "early"}, backbone="resnet18",
             inputs={"a": {"in_channels": 3, key: value}, "b": {"in_channels": 2}})


def test_late_fusion_image_head_reads_the_primary_deepest_level():
    """Regression: DenseModel._deepest searched every input's levels, so an image head built
    for the primary's 256-ch /16 level was fed the other input's 512-ch /32 level."""
    model = make(fusion={"type": "late"}, num_classes=None,
                 inputs={"a": {"in_channels": 4, "backbone": "resnet18",
                               "out_indices": (0, 1, 2, 3)},
                         "sar": {"in_channels": 2, "backbone": "resnet18"}},
                 heads={"seg": {"type": "segmentation", "out_channels": 2},
                        "cls": {"type": "classification", "out_channels": 4,
                                "scope": "image"}}).eval()
    out = model({"a": torch.randn(1, 4, 64, 64), "sar": torch.randn(1, 2, 64, 64)})
    assert out["cls"].shape == (1, 4)
    a, b = model.backbone.groups[0]
    assert a <= model._deepest < b


def test_late_fusion_same_extent_maps_valid_region_onto_valid_region():
    """Regression: the whole padded per-input decoder output was interpolated onto the
    primary's padded output, so 25 of 257 valid rows read the other input's padding."""
    from manyseg.features import FeatureSpec
    from manyseg.fusion import LateFusionDecoder
    from torch import nn

    class Identity(nn.Module):
        def __init__(self, specs):
            super().__init__()
            self.out_spec = FeatureSpec(1, 1)

        def forward(self, feats):
            return feats[0]

    H, W, h, w = 257, 301, 129, 151
    dec = LateFusionDecoder([FeatureSpec(1, 1), FeatureSpec(1, 1)], [(0, 1), (1, 2)], Identity,
                            op="add", alignment=("same_grid", "same_extent"))
    a = torch.zeros(1, 1, 288, 320)
    b = torch.zeros(1, 1, 160, 160)
    b[..., :h, :w] = 1.0
    out = dec([a, b], valid=[(H, W), (h, w)])[0, 0, :H, :W]
    assert float(out.min()) > 0.5  # every valid primary pixel reads b's valid region


def test_late_fusion_same_extent_end_to_end_alignment():
    """Through create_model: a half-resolution same_extent input is aligned by extent."""
    model = make(fusion={"type": "late", "op": "add"},
                 inputs={"optical": {"in_channels": 4, "backbone": "resnet18"},
                         "sar": {"in_channels": 2, "backbone": "resnet18",
                                 "alignment": "same_extent"}}).eval()
    out = model(batch(257, 301, n=1, sar_hw=(129, 151)))
    assert out.shape == (1, 3, 257, 301)



def test_shared_encoder_keeps_batchnorm_statistics_per_input():
    """Regression (E015): one BatchNorm shared by inputs with different statistics blends its
    running statistics, so evaluation used the blend for every input and the model collapsed
    (val 0.50-0.65 vs 0.89). Shared encoders now keep BN per input."""
    inputs = {"t0": {"in_channels": 3, "backbone": "resnet18"},
              "t1": {"in_channels": 3, "backbone": "shared:t0"}}
    model = make(inputs=inputs, fusion={"type": "intermediate", "op": "add"}).train()
    for _ in range(3):
        model({"t0": torch.randn(2, 3, 64, 64), "t1": 5 + 3 * torch.randn(2, 3, 64, 64)})
    bn0 = model.backbone.encoders["t0"].model.bn1
    bn1 = model.backbone.encoders["t1"].model.bn1
    assert bn0 is not bn1 and not torch.allclose(bn0.running_mean, bn1.running_mean)


def test_share_norm_true_keeps_one_module():
    inputs = {"t0": {"in_channels": 3, "backbone": "resnet18"},
              "t1": {"in_channels": 3, "backbone": "shared:t0", "share_norm": True}}
    model = make(inputs=inputs, fusion={"type": "intermediate", "op": "add"})
    assert model.backbone.encoders["t0"] is model.backbone.encoders["t1"]


def test_share_norm_requires_shared_backbone():
    with pytest.raises(ValueError, match="share_norm"):
        make(inputs={"a": {"in_channels": 3}, "b": {"in_channels": 3, "share_norm": True}})
