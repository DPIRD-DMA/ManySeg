"""Decoder ``out_stride``: U-Net-family decoders can stop before full resolution.

Stages finer than ``out_stride`` are not built, so a reduced decoder is a strict prefix of the
default one: same module names, shapes and computation for everything it keeps.
"""

import math

import pytest
import torch
from manyseg import FeatureSpec, create_model
from manyseg.decoders import DECODERS
from torch.utils.flop_counter import FlopCounterMode

SPECS5 = (FeatureSpec(64, 2), FeatureSpec(64, 4), FeatureSpec(128, 8),
          FeatureSpec(256, 16), FeatureSpec(512, 32))
SPECS4 = SPECS5[1:]  # hierarchical-transformer-like: shallowest stride 4
INPUT_SPEC = FeatureSpec(3, 1, "input")
UNET_FAMILY = ["unet", "unetplusplus", "fastai_unet"]
STRIDE4 = ["fpn", "deeplabv3plus", "upernet", "segformer"]


def build(name, specs, **kw):
    cls = DECODERS[name]
    if getattr(cls, "wants_input", False):
        return cls((INPUT_SPEC, *specs), **kw)
    return cls(specs, **kw)


def feats(specs, h, w, n=1, with_input=False):
    out = [torch.randn(n, s.channels, math.ceil(h / s.stride), math.ceil(w / s.stride))
           for s in specs]
    return [torch.randn(n, 3, h, w), *out] if with_input else out


def run(dec, specs, h=64, w=96, n=1):
    return dec(feats(specs, h, w, n, getattr(dec, "wants_input", False)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def module_flops(dec, specs, h=64, w=64):
    dec.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        run(dec, specs, h, w)
    return {k.split(".", 1)[1] if "." in k else "": sum(v.values())
            for k, v in fc.get_flop_counts().items() if k != "Global"}


@pytest.mark.parametrize("name", UNET_FAMILY)
@pytest.mark.parametrize("specs", [SPECS5, SPECS4], ids=["5lvl", "4lvl"])
@pytest.mark.parametrize("out_stride", [1, 2, 4])
def test_shapes(name, specs, out_stride):
    dec = build(name, specs, out_stride=out_stride).eval()
    assert dec.out_spec.stride == out_stride
    out = run(dec, specs, 64, 96, n=2)
    assert out.shape == (2, dec.out_spec.channels, 64 // out_stride, 96 // out_stride)


@pytest.mark.parametrize("name", UNET_FAMILY)
def test_default_is_out_stride_1(name):
    torch.manual_seed(0)
    a = build(name, SPECS5).eval()
    torch.manual_seed(0)
    b = build(name, SPECS5, out_stride=1).eval()
    assert a.out_spec == b.out_spec
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("name", UNET_FAMILY)
@pytest.mark.parametrize("specs", [SPECS5, SPECS4], ids=["5lvl", "4lvl"])
@pytest.mark.parametrize("out_stride", [2, 4])
def test_reduced_decoder_is_a_prefix_of_the_default(name, specs, out_stride):
    """Every kept parameter exists in the default decoder with the same name and shape, and
    the parameter / FLOP savings are exactly the skipped modules."""
    full = build(name, specs)
    red = build(name, specs, out_stride=out_stride)
    fs, rs = full.state_dict(), red.state_dict()
    assert set(rs) < set(fs)
    assert all(rs[k].shape == fs[k].shape for k in rs)
    skipped = {k: v for k, v in full.named_parameters() if k not in rs}
    assert n_params(full) - n_params(red) == sum(p.numel() for p in skipped.values()) > 0

    ff, rf = module_flops(full, specs), module_flops(red, specs)
    assert rf[""] < ff[""]
    # kept modules run on identical shapes, so their FLOPs are unchanged
    for k, v in rf.items():
        if k:
            assert ff[k] == v, k


def test_unet_out_stride_drops_the_finest_blocks():
    full, red = build("unet", SPECS5), build("unet", SPECS5, out_stride=4)
    assert len(full.blocks) == 5 and len(red.blocks) == 3
    ff, rf = module_flops(full, SPECS5), module_flops(red, SPECS5)
    assert ff[""] - rf[""] == ff["blocks.3"] + ff["blocks.4"]


def test_unetplusplus_out_stride_drops_level0_and_tail():
    red = build("unetplusplus", SPECS5, out_stride=4)
    assert not any(k.startswith("0_") for k in red.nodes)
    assert len(red.nodes) == 6 and len(red.tail) == 0
    red2 = build("unetplusplus", SPECS5, out_stride=2)
    assert len(red2.nodes) == 10 and len(red2.tail) == 0


def test_fastai_unet_drop_definitions():
    """out_stride 1: everything; 2: no final shuffle, no input cross / residual block;
    4: additionally no last (stride-2) up-block."""
    full = build("fastai_unet", SPECS5)
    s2 = build("fastai_unet", SPECS5, out_stride=2)
    s4 = build("fastai_unet", SPECS5, out_stride=4)
    assert full.final_shuffle is not None and full.last_cross and len(full.blocks) == 4
    assert s2.final_shuffle is None and not s2.last_cross and not hasattr(s2, "res")
    assert len(s2.blocks) == 4
    assert s4.final_shuffle is None and not s4.last_cross and len(s4.blocks) == 3
    # last kept block keeps its default width (the reduced decoder is a prefix)
    assert s4.out_spec.channels == full.blocks[2].out_channels
    assert s2.out_spec.channels == full.blocks[3].out_channels


def test_fastai_unet_stride4_backbone_out_stride_2_keeps_the_shuffle():
    """With a stride-4 shallowest skip, the final pixel shuffle is what reaches stride 2."""
    s2 = build("fastai_unet", SPECS4, out_stride=2)
    assert s2.final_shuffle is not None and not s2.last_cross
    s4 = build("fastai_unet", SPECS4, out_stride=4)
    assert s4.final_shuffle is None and len(s4.blocks) == 3


def test_fastai_unet_last_cross_needs_full_resolution():
    with pytest.raises(ValueError, match="last_cross"):
        build("fastai_unet", SPECS5, out_stride=2, last_cross=True)
    dec = build("fastai_unet", SPECS5, last_cross=False)  # still allowed at stride 1
    assert dec.out_spec.stride == 1


@pytest.mark.parametrize("name", UNET_FAMILY)
@pytest.mark.parametrize("bad", [0, 3, 6, 64])
def test_invalid_out_stride(name, bad):
    with pytest.raises(ValueError, match="out_stride"):
        build(name, SPECS5, out_stride=bad)


@pytest.mark.parametrize("name", STRIDE4)
def test_stride4_decoders_accept_their_own_stride(name):
    dec = build(name, SPECS5, out_stride=4)
    assert dec.out_spec.stride == 4


@pytest.mark.parametrize("name", STRIDE4)
@pytest.mark.parametrize("bad", [1, 2, 8])
def test_stride4_decoders_reject_other_strides(name, bad):
    with pytest.raises(ValueError, match=rf"{name}.*stride 4.*out_stride={bad}"):
        build(name, SPECS5, out_stride=bad)


@pytest.mark.parametrize("name", UNET_FAMILY)
@pytest.mark.parametrize("out_stride", [2, 4])
def test_gradients_reach_every_parameter(name, out_stride):
    dec = build(name, SPECS5, out_stride=out_stride)
    run(dec, SPECS5, 64, 64, n=2).square().mean().backward()
    for k, p in dec.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), k


@pytest.mark.parametrize("name", UNET_FAMILY)
@pytest.mark.parametrize("out_stride", [1, 2, 4])
@pytest.mark.parametrize("hw", [(64, 64), (65, 97), (33, 200)])
def test_model_output_stays_full_resolution(name, out_stride, hw):
    """The head upsamples by the decoder stride and the model crops: outputs are (H, W)."""
    model = create_model(backbone="resnet18", pretrained=False, decoder=name, num_classes=3,
                         decoder_kwargs={"out_stride": out_stride}).eval()
    assert model.decoder.out_spec.stride == out_stride
    assert model(torch.randn(1, 3, *hw)).shape == (1, 3, *hw)


@pytest.mark.parametrize("name", UNET_FAMILY)
def test_model_with_transformer_like_backbone(name):
    """Backbones whose shallowest feature is stride 4 (no stride-2 skip)."""
    model = create_model(backbone="convnext_atto", pretrained=False, decoder=name, num_classes=2,
                         decoder_kwargs={"out_stride": 2}).eval()
    assert model(torch.randn(1, 3, 65, 97)).shape == (1, 2, 65, 97)


@pytest.mark.parametrize("name,out_stride", [("unet", 4), ("fastai_unet", 2),
                                             ("unetplusplus", 2)])
def test_film_identity_at_init_with_reduced_out_stride(name, out_stride):
    vec = {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}}
    kw = dict(backbone="resnet18", pretrained=False, num_classes=2, decoder=name,
              decoder_kwargs={"out_stride": out_stride})
    cond = create_model(inputs={"img": {"channels": 3}, "meta": vec}, **kw).eval()
    plain = create_model(**kw)
    plain.load_state_dict({k: v for k, v in cond.state_dict().items()
                           if not k.startswith("conditioner.")})
    x = torch.randn(2, 3, 33, 47)
    out = cond({"img": x, "meta": torch.tensor([[10.0], [200.0]])})
    torch.testing.assert_close(out, plain.eval()(x), rtol=0, atol=0)
    if name == "unet" and out_stride == 4:  # the unread stride-2 level gets no FiLM
        assert "0" not in cond.conditioner.film["decoder"].features


def test_task_decoders_with_different_out_strides():
    heads = {"seg": {"type": "segmentation", "channels": 3},
             "edges": {"type": "segmentation", "channels": 1,
                       "decoder": {"type": "unet", "out_stride": 2}},
             "depth": {"type": "regression", "channels": 1,
                       "decoder": {"type": "fastai_unet", "out_stride": 4}}}
    model = create_model(backbone="resnet18", pretrained=False, heads=heads,
                         decoder_kwargs={"out_stride": 4}).eval()
    assert model.decoder.out_spec.stride == 4
    assert model.task_decoders["edges"].out_spec.stride == 2
    assert model.task_decoders["depth"].out_spec.stride == 4
    out = model(torch.randn(2, 3, 65, 97))
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "seg": (2, 3, 65, 97), "edges": (2, 1, 65, 97), "depth": (2, 1, 65, 97)}


def test_multi_input_outputs_follow_the_primary_grid():
    inputs = {"optical": {"channels": 4}, "sar": {"channels": 2, "alignment": "same_extent"}}
    model = create_model(inputs=inputs, backbone="resnet18", pretrained=False, num_classes=2,
                         decoder_kwargs={"out_stride": 2}).eval()
    out = model({"optical": torch.randn(1, 4, 65, 97), "sar": torch.randn(1, 2, 33, 49)})
    assert out.shape == (1, 2, 65, 97)
