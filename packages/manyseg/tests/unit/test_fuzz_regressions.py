"""Regressions found by the random-configuration fuzzer (tests/fuzz): each was a crash deep in
forward / timm, a misleading error, or a valid configuration refused."""

import pytest
import torch
from manyseg import create_model


def build(**kw):
    kw.setdefault("backbone", "resnet18")
    return create_model(pretrained=False, **kw)


def test_out_of_range_out_indices_is_a_clear_error():
    with pytest.raises(ValueError, match="out_indices"):
        build(backbone="pvt_v2_b0", out_indices=[1, 2, 3, 4], num_classes=2)


def test_task_decoder_with_encoder_source_is_rejected():
    """A head reading the encoder directly cannot also have its own decoder (it used to build
    on the encoder's channels and then crash on the decoder's output)."""
    with pytest.raises(ValueError, match="source"):
        build(heads={"a": {"type": "segmentation", "out_channels": 2, "source": "encoder",
                           "decoder": "fpn"}})


def test_per_input_head_with_encoder_source_is_rejected():
    with pytest.raises(ValueError, match="source"):
        build(inputs={"t1": {"in_channels": 3}, "t2": {"in_channels": 3}},
              heads={"c": {"type": "segmentation", "out_channels": 2},
                     "s1": {"type": "segmentation", "out_channels": 2, "input": "t1",
                            "source": "encoder"}})


def test_image_head_from_the_decoder_alone_builds_the_decoder():
    m = build(heads={"c": {"type": "classification", "out_channels": 3, "scope": "image",
                           "source": "decoder"}}).eval()
    assert m.decoder is not None
    assert m(torch.randn(1, 3, 64, 64))["c"].shape == (1, 3)
    m = build(heads={"c": {"type": "classification", "out_channels": 3, "scope": "image",
                           "source": "decoder"},
                     "s": {"type": "segmentation", "out_channels": 2, "decoder": "fpn"}}).eval()
    out = m(torch.randn(1, 3, 64, 64))
    assert out["c"].shape == (1, 3) and out["s"].shape == (1, 2, 64, 64)


def test_early_fusion_with_a_shared_backbone_names_the_conflict():
    with pytest.raises(ValueError, match="early fusion"):
        build(inputs={"a": {"in_channels": 4}, "b": {"in_channels": 2, "backbone": "shared:a"}},
              fusion={"type": "early"}, num_classes=2)


@pytest.mark.parametrize("dec", [{"decoder": "fastai_unet"},
                                 {"decoder": "unet", "decoder_kwargs": {"input_cross": True}}])
def test_late_fusion_rejects_input_image_decoders_clearly(dec):
    with pytest.raises(ValueError, match="input image"):
        build(inputs={"a": {"in_channels": 1}, "b": {"in_channels": 3}}, fusion={"type": "late"},
              num_classes=2, **dec)


@pytest.mark.parametrize("temporal,t", [({"type": "concat", "dates": 3}, 3), ("diff", 2)])
def test_export_keeps_a_fixed_date_count_static(temporal, t):
    from manyseg.export import export_model

    m = build(num_classes=2, inputs={"s2": {"kind": "timeseries", "in_channels": 4,
                                            "temporal": temporal}}).eval()
    ep = export_model(m, {"s2": torch.randn(1, t, 4, 64, 64)})
    y = {"s2": torch.randn(1, t, 4, 96, 64)}
    torch.testing.assert_close(ep.module()(y), m(y), atol=1e-4, rtol=1e-4)


def test_export_same_extent_input_is_independent_of_example_sizes():
    """An example whose same_extent input happens to match the primary's size must not tie
    the two sizes together in the exported program."""
    from manyseg.export import export_model

    m = build(num_classes=2, inputs={"a": {"in_channels": 3},
                                     "b": {"in_channels": 2, "alignment": "same_extent"}}).eval()
    ep = export_model(m, {"a": torch.randn(1, 3, 64, 64), "b": torch.randn(1, 2, 64, 64)})
    y = {"a": torch.randn(1, 3, 96, 128), "b": torch.randn(1, 2, 48, 64)}
    torch.testing.assert_close(ep.module()(y), m(y), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kw", [{"heads": 8, "dim": 4}, {"heads": 0}])
def test_cross_attention_rejects_impossible_head_settings(kw):
    with pytest.raises(ValueError, match="heads"):
        build(num_classes=2, inputs={"a": {"in_channels": 3}, "b": {"in_channels": 2}},
              fusion={"type": "intermediate", "op": "cross_attention", "op_kwargs": kw})


@pytest.mark.parametrize("backbone", ["mobilenetv3_small_100", "convnext_atto"])
def test_input_cross_without_stride_padding(backbone):
    m = build(backbone=backbone, num_classes=2, pad_to_stride=False,
              decoder_kwargs={"input_cross": True}).eval()
    assert m(torch.randn(1, 3, 69, 75)).shape == (1, 2, 69, 75)


def test_same_grid_inputs_with_different_pad_multiples_fuse():
    """An input using only shallow levels pads to a smaller multiple (8) than the primary
    (32); its features must still line up with the primary's."""
    m = build(backbone="efficientnet_b0", num_classes=2, fusion={"type": "intermediate",
                                                                 "op": "add"},
              inputs={"a": {"in_channels": 4}, "b": {"in_channels": 1, "out_indices": [2]}}).eval()
    x = {"a": torch.randn(1, 4, 41, 45), "b": torch.randn(1, 1, 41, 45)}
    y = m(x)
    assert y.shape == (1, 2, 41, 45)
    # both are padded to the common multiple (32): same as padding them ourselves
    padded = {k: torch.nn.functional.pad(v, (0, 64 - 45, 0, 64 - 41)) for k, v in x.items()}
    torch.testing.assert_close(m(padded)[..., :41, :45], y)


def test_export_without_stride_padding_uses_a_safe_minimum_size():
    """Unpadded inputs through a stem that rounds down (ConvNeXt's 4x4 / stride 4 patchify)
    need two cells at the deepest stride: the dynamic range must start there."""
    from manyseg.export import dynamic_shapes, export_model

    m = build(backbone="convnext_atto", num_classes=2, pad_to_stride=False).eval()
    x = torch.randn(1, 3, 70, 75)
    assert dynamic_shapes(m, x)["x"][2].min == 64
    ep = export_model(m, x)
    y = torch.randn(1, 3, 97, 131)
    torch.testing.assert_close(ep.module()(y), m(y), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kw,x", [
    # timm's MobileNetV3 feature model runs every block even when only stride 4 is returned
    ({"backbone": "mobilenetv3_small_100", "out_indices": [1], "pad_to_stride": False,
      "heads": {"c": {"type": "classification", "out_channels": 1, "scope": "image"}}},
     torch.randn(1, 3, 70, 75)),
])
def test_export_minimum_size_covers_the_whole_network(kw, x):
    from manyseg.export import export_model

    m = build(**kw).eval()
    ep = export_model(m, x)
    y = ({k: torch.randn(*v.shape[:-2], 97, 131) for k, v in x.items()} if isinstance(x, dict)
         else torch.randn(1, 3, 97, 131))
    out, ref = ep.module()(y), m(y)
    ref = ref if isinstance(ref, dict) else {"y": ref}
    out = out if isinstance(out, dict) else {"y": out}
    for k in ref:
        torch.testing.assert_close(out[k], ref[k], atol=1e-4, rtol=1e-4)


def test_cross_attention_needs_a_second_input():
    from manyseg.fusion.ops import CrossAttentionFusion

    with pytest.raises(ValueError, match="cross_attention"):
        CrossAttentionFusion([64], 64)


@pytest.mark.slow  # MaxViT on CPU
def test_fixed_size_backbone_in_a_multi_input_model_pads_like_a_single_input():
    """A fixed-size encoder (MaxViT 224) pads smaller inputs to its size, as single-input
    models do, instead of refusing everything but exactly 224 x 224."""
    m = build(backbone="maxvit_tiny_tf_224", num_classes=2,
              inputs={"a": {"in_channels": 3}, "b": {"in_channels": 3, "alignment": "same_extent"},
                      "c": {"in_channels": 1, "backbone": "resnet18"}}).eval()
    x = {"a": torch.randn(1, 3, 150, 170), "b": torch.randn(1, 3, 90, 100),
         "c": torch.randn(1, 1, 150, 170)}
    assert m(x).shape == (1, 2, 150, 170)
    with pytest.raises(ValueError, match="fixed input size"):
        m({**x, "a": torch.randn(1, 3, 300, 300), "c": torch.randn(1, 1, 300, 300)})


def test_fastai_unet_export_without_stride_padding_at_other_parities():
    """Unpadded, PVT's level sizes depend on the input's parity; the decoder's resize to the
    skip size must not be a shape branch frozen at the example's parity."""
    from manyseg.export import export_model

    m = build(backbone="pvt_v2_b0", num_classes=2, decoder="fastai_unet",
              pad_to_stride=False).eval()
    ep = export_model(m, torch.randn(1, 3, 70, 75))
    for hw in ((97, 131), (128, 96)):
        y = torch.randn(1, 3, *hw)
        torch.testing.assert_close(ep.module()(y), m(y), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("bad", [[1.5, 2], [True, 2], ["1", 2], [1, 5]])
def test_out_indices_must_be_integers(bad):
    with pytest.raises(ValueError, match="out_indices"):
        build(out_indices=bad, num_classes=2)


def test_negative_out_indices_still_work():
    m = build(out_indices=[-2, -1], num_classes=2).eval()
    assert [s.stride for s in m.feature_specs] == [16, 32]


@pytest.mark.parametrize("kw", [{"heads": 1.5}, {"heads": True}, {"heads": "4"}, {"dim": 0},
                                {"dim": -8}, {"dim": 2.5}])
def test_cross_attention_settings_must_be_positive_integers(kw):
    with pytest.raises(ValueError, match="cross_attention"):
        build(num_classes=2, inputs={"a": {"in_channels": 3}, "b": {"in_channels": 2}},
              fusion={"type": "intermediate", "op": "cross_attention", "op_kwargs": kw})


@pytest.mark.parametrize("ftype", ["intermediate", "late"])
def test_fusion_op_with_a_single_raster_input_is_rejected(ftype):
    """With one raster input there is nothing to fuse: an explicit op would be silently
    ignored (intermediate) or meaningless (late)."""
    with pytest.raises(ValueError, match="two raster inputs"):
        build(num_classes=2, inputs={"a": {"in_channels": 3}},
              fusion={"type": ftype, "op": "cross_attention"})
    with pytest.raises(ValueError, match="two raster inputs"):
        build(num_classes=2, inputs={"a": {"in_channels": 3},
                                     "m": {"kind": "vector", "num_features": 2}},
              fusion={"type": ftype, "op": "add"})


def assert_every_parameter_used(m, x):
    """DDP's default (find_unused_parameters=False) fails on parameters without a gradient."""
    m.train()
    y = m(x)
    sum(v.float().square().mean() for v in (y.values() if isinstance(y, dict) else [y])
        ).backward()
    unused = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, unused


TWO = {"a": {"in_channels": 3}, "b": {"in_channels": 2}}


def two(hw=(64, 64)):
    return {"a": torch.randn(2, 3, *hw), "b": torch.randn(2, 2, *hw)}


@pytest.mark.parametrize("decoder", ["fpn", "deeplabv3plus", "upernet", "segformer", "unet"])
def test_fusion_only_at_levels_something_reads(decoder):
    """ResNet has a stride-2 level that FPN / DeepLab / UPerNet / SegFormer never read: no
    fuser there (it would be a parameter without gradient)."""
    m = build(num_classes=2, decoder=decoder, inputs=TWO,
              fusion={"type": "intermediate", "op": "gated"})
    assert_every_parameter_used(m, two())


def test_unread_levels_with_encoder_heads_only():
    m = build(heads={"c": {"type": "classification", "out_channels": 3, "scope": "image"},
                     "e": {"type": "segmentation", "out_channels": 2, "source": "encoder"}},
              inputs=TWO, fusion={"type": "intermediate", "op": "cross_attention"})
    assert m.decoder is None
    assert_every_parameter_used(m, two())


@pytest.mark.parametrize("temporal", ["attention", {"type": "concat", "dates": 2}])
def test_time_series_aggregation_only_at_read_levels(temporal):
    m = build(num_classes=2, decoder="deeplabv3plus",
              inputs={"s": {"kind": "timeseries", "in_channels": 4, "temporal": temporal}})
    assert_every_parameter_used(m, {"s": torch.randn(2, 2, 4, 64, 64)})


@pytest.mark.parametrize("decoder", ["deeplabv3plus", None])
def test_vit_neck_only_at_read_levels(decoder):
    kw = ({"num_classes": 2, "decoder": decoder} if decoder else
          {"heads": {"c": {"type": "classification", "out_channels": 3, "scope": "image"}}})
    m = build(backbone="vit_tiny_patch16_224", **kw)
    assert_every_parameter_used(m, torch.randn(2, 3, 64, 64))
    feats = m.eval().features(torch.randn(1, 3, 64, 64))
    assert feats[-1] is not None


def test_late_fusion_and_per_input_heads_use_every_parameter():
    m = build(decoder="deeplabv3plus", inputs=TWO, fusion={"type": "late", "op": "add"},
              num_classes=2)
    assert_every_parameter_used(m, two())
    m = build(decoder="fpn", inputs={"t1": {"in_channels": 3}, "t2": {"in_channels": 3}},
              fusion={"type": "intermediate", "op": "diff"},
              heads={"c": {"type": "segmentation", "out_channels": 2},
                     "s1": {"type": "segmentation", "out_channels": 4, "input": "t1"},
                     "s2": {"type": "segmentation", "out_channels": 4, "input": "t2",
                            "share": "s1"}})
    assert_every_parameter_used(m, {"t1": torch.randn(2, 3, 64, 64),
                                    "t2": torch.randn(2, 3, 64, 64)})


@pytest.mark.parametrize("name", ["efficientnet_b0", "mobilenetv3_small_100"])
def test_blocks_after_the_last_requested_level_are_dropped(name):
    """timm's EfficientNet / MobileNetV3 feature models run every block whatever out_indices
    asks for; the blocks past the deepest requested level are removed (same features)."""
    import timm
    from manyseg.backbones import create_backbone

    torch.manual_seed(0)
    bb = create_backbone(name, pretrained=False, out_indices=[0, 2]).eval()
    torch.manual_seed(0)
    ref = timm.create_model(name, pretrained=False, features_only=True,
                            out_indices=(0, 2)).eval()
    x = torch.randn(1, 3, 64, 64)
    for a, b in zip(bb(x), ref(x)):
        torch.testing.assert_close(a, b)
    m = build(backbone=name, out_indices=[0, 2], num_classes=2)
    assert_every_parameter_used(m, torch.randn(2, 3, 64, 64))


def test_export_minimum_size_when_inputs_are_padded_less_than_the_network_stride():
    """A multi-input model pads to the multiple its returned levels need (4 here) while PVT's
    stage-1 attention still reduces by 32 in all."""
    from manyseg.export import export_model

    m = build(backbone="pvt_v2_b0", num_classes=2,
              inputs={"r0": {"in_channels": 2, "out_indices": [0]}}).eval()
    ep = export_model(m, {"r0": torch.randn(1, 2, 70, 75)})
    y = {"r0": torch.randn(1, 2, 97, 131)}
    torch.testing.assert_close(ep.module()(y), m(y), atol=1e-4, rtol=1e-4)


def test_an_input_nothing_reads_is_rejected():
    """Late fusion combines inputs in the decoder: with only encoder-reading heads the second
    input's encoder would be trained for nothing."""
    with pytest.raises(ValueError, match="never be read"):
        build(inputs=TWO, fusion={"type": "late"},
              heads={"c": {"type": "classification", "out_channels": 3, "scope": "image"}})
    with pytest.raises(ValueError, match="never be read"):  # its only level is unread
        build(inputs={"a": {"in_channels": 3}, "b": {"in_channels": 2, "out_indices": [0]}},
              decoder="deeplabv3plus", num_classes=2)


def test_an_unread_input_sharing_a_read_encoder_is_allowed():
    """Siamese inputs: the second date's encoder is the first's, so leaving it unread wastes
    compute but leaves no parameter without a gradient."""
    def siamese(share_norm):
        return build(inputs={"t1": {"in_channels": 3},
                             "t2": {"in_channels": 3, "backbone": "shared:t1",
                                    "share_norm": share_norm}},
                     fusion={"type": "intermediate", "op": "diff"},
                     heads={"lc1": {"type": "segmentation", "out_channels": 4, "input": "t1"}})

    assert_every_parameter_used(siamese(True), {"t1": torch.randn(2, 3, 64, 64),
                                                "t2": torch.randn(2, 3, 64, 64)})
    with pytest.raises(ValueError, match="never be read"):  # its own norms: no gradient
        siamese(False)


def test_an_unread_input_with_its_own_channel_adapter_is_rejected():
    """Sharing a read input's encoder is not enough when the input has its own adapter (a
    different band count): the adapter would get no gradient."""
    with pytest.raises(ValueError, match="never be read"):
        build(inputs={"a": {"in_channels": 3}, "b": {"in_channels": 4, "backbone": "shared:a",
                                                  "share_norm": True,
                                                  "channel_adapter": "projection"}},
              fusion={"type": "late"},
              heads={"c": {"type": "classification", "out_channels": 3, "scope": "image"}})


def test_shared_per_input_decoders_mark_both_inputs_read():
    """One decoder shared by two per-input heads reads both inputs (Codex review)."""
    m = build(inputs={"t1": {"in_channels": 3}, "t2": {"in_channels": 3}},
              fusion={"type": "intermediate"},
              heads={"s1": {"type": "segmentation", "out_channels": 4, "input": "t1"},
                     "s2": {"type": "segmentation", "out_channels": 4, "input": "t2",
                            "share": "s1"}})
    assert_every_parameter_used(m, {"t1": torch.randn(2, 3, 64, 64),
                                    "t2": torch.randn(2, 3, 64, 64)})


def test_same_grid_padding_uses_the_least_common_multiple():
    from manyseg.features import FeatureSpec

    class Enc(torch.nn.Module):  # custom encoder with a stride that is not a power of two
        def __init__(self, stride):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, stride, stride)
            self.feature_specs = (FeatureSpec(8, stride, "c"),)
            self.input_divisor = stride
            self.in_channels = 3

        def forward(self, x):
            return [self.conv(x)]

    from manyseg.fusion.encoders import MultiEncoder

    enc = MultiEncoder({"a": Enc(6), "b": Enc(10)}, mode="intermediate", op="add")
    assert enc.grid_divisor == 30


def test_select_heads_prunes_levels_only_the_dropped_heads_read():
    m = build(decoder="unet", inputs=TWO, fusion={"type": "intermediate", "op": "gated"},
              heads={"seg": {"type": "segmentation", "out_channels": 2},
                     "cls": {"type": "classification", "out_channels": 3, "scope": "image"}})
    x = two()
    full = m.eval()(x)["cls"]
    m.select_heads(["cls"])
    torch.testing.assert_close(m.eval()(x)["cls"], full)  # outputs unchanged
    assert_every_parameter_used(m, two())



def test_deeper_levels_of_an_input_that_nothing_reads_are_rejected():
    """The primary stops at stride 8, so the second input's stride 16 / 32 levels are never
    fused: the blocks computing them would get no gradient. Limiting out_indices fixes it."""
    with pytest.raises(ValueError, match="out_indices"):
        build(backbone="mobilenetv3_small_100", num_classes=2, decoder="unet",
              inputs={"a": {"in_channels": 3, "out_indices": [1, 2]}, "b": {"in_channels": 1}})
    m = build(backbone="mobilenetv3_small_100", num_classes=2, decoder="unet",
              inputs={"a": {"in_channels": 3, "out_indices": [1, 2]},
                      "b": {"in_channels": 1, "out_indices": [0, 1, 2]}})
    assert_every_parameter_used(m, {"a": torch.randn(2, 3, 64, 64),
                                    "b": torch.randn(2, 1, 64, 64)})


def test_film_reaches_pixel_heads_that_read_the_encoder():
    """With only an encoder-reading pixel head there is no decoder to modulate: FiLM goes on
    the deepest encoder feature, as for image heads (the metadata encoder was unused)."""
    meta = {"kind": "vector", "conditioning": "film",
            "fields": {"d": {"type": "continuous", "mean": 0.0, "std": 1.0}}}
    m = build(inputs={"img": {"in_channels": 3}, "meta": meta},
              heads={"e": {"type": "regression", "out_channels": 2, "source": "encoder"}})
    assert_every_parameter_used(m, {"img": torch.randn(2, 3, 64, 64),
                                    "meta": torch.randn(2, 1)})


def test_select_heads_may_leave_inputs_unread():
    """At deployment the kept heads may ignore some inputs (here: late fusion, keeping only
    an image head drops the decoder that combined them); that is not an error."""
    m = build(decoder="upernet", inputs=TWO, fusion={"type": "late", "op": "add"},
              heads={"cls": {"type": "classification", "out_channels": 3, "scope": "image"},
                     "seg": {"type": "segmentation", "out_channels": 2}}).eval()
    x = two()
    full = m(x)["cls"]
    torch.testing.assert_close(m.select_heads(["cls"])(x)["cls"], full)


def test_early_fusion_with_a_prunable_backbone():
    """Early fusion stacks inputs into one ViT: its unread neck levels are pruned too."""
    m = build(backbone="vit_tiny_patch16_224", decoder="deeplabv3plus", num_classes=2,
              inputs=TWO, fusion={"type": "early"})
    assert_every_parameter_used(m, two())


@pytest.mark.parametrize("head,right", [
    ({"type": "segmentation", "out_channels": 2, "scope": "image"}, "classification"),
    ({"type": "classification", "out_channels": 2, "scope": "pixel"}, "segmentation")])
def test_head_type_must_match_its_scope(head, right):
    with pytest.raises(ValueError, match=right):
        build(heads={"h": head})


def test_decoder_as_a_dict():
    a = build(decoder={"type": "unet", "out_stride": 2}, num_classes=2).eval()
    torch.manual_seed(0)
    b = build(decoder="unet", decoder_kwargs={"out_stride": 2}, num_classes=2).eval()
    assert a.decoder.out_spec == b.decoder.out_spec
    with pytest.raises(ValueError, match="type"):
        build(decoder={"out_stride": 2}, num_classes=2)
    with pytest.raises(ValueError, match="out_stride"):
        build(decoder={"type": "unet", "out_stride": 2}, decoder_kwargs={"out_stride": 1},
              num_classes=2)


def test_resize_nearest_rule():
    """Exact floor(i * in / out); equal to F.interpolate(mode="nearest") whenever the output
    is a multiple of the input, which is always the case with stride padding."""
    from manyseg.utils.spatial import resize_nearest

    torch.manual_seed(0)
    for h, w in [(1, 1), (3, 5), (41, 33), (49, 66), (97, 131)]:
        x = torch.randn(2, 3, h, w)
        for size in [(2 * h - 1, 2 * w), (2 * h + 1, 2 * w - 1), (h // 2 + 1, 3 * w + 1)]:
            rows = torch.arange(size[0]) * h // size[0]
            cols = torch.arange(size[1]) * w // size[1]
            assert torch.equal(resize_nearest(x, size), x[..., rows[:, None], cols[None, :]])
        for size in [(2 * h, 2 * w), (h, w), (4 * h, 3 * w)]:
            assert torch.equal(resize_nearest(x, size),
                               torch.nn.functional.interpolate(x, size=size, mode="nearest"))


@pytest.mark.slow
@pytest.mark.cuda
def test_nearest_upsampling_compiles_correctly_on_cuda():
    """torch.compile(dynamic=True) on CUDA (torch 2.14) miscomputes F.interpolate(mode=
    "nearest") when the output height is 2h - 1 and the width 2w; the decoders avoid it
    (found by the GPU fuzzer: unet / fastai_unet with pad_to_stride=False)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from torch import _dynamo

    torch.backends.cudnn.allow_tf32 = False  # compare like with like
    for decoder in ["unet", "fastai_unet", "fpn"]:
        torch.manual_seed(0)
        m = build(backbone="efficientnet_b0", in_channels=1, num_classes=5, decoder=decoder,
                  pad_to_stride=False).eval().cuda()
        _dynamo.reset()
        c = torch.compile(m, dynamic=True)
        with torch.no_grad():
            for hw in [(70, 75), (97, 131), (81, 66)]:
                x = torch.randn(2, 1, *hw, device="cuda")
                a, b = c(x), m(x)  # the bug gave relative errors of 0.07 to 0.99
                assert float((a - b).norm() / b.norm()) < 1e-3, (decoder, hw)
