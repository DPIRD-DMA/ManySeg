import pytest
import torch
from manyseg import FeatureSpec
from manyseg.decoders import DECODERS, UNetDecoder

SPECS5 = (
    FeatureSpec(64, 2), FeatureSpec(64, 4), FeatureSpec(128, 8),
    FeatureSpec(256, 16), FeatureSpec(512, 32),
)
SPECS4 = SPECS5[1:]


def feats(specs, h, w, n=2):
    import math
    return [torch.randn(n, s.channels, math.ceil(h / s.stride), math.ceil(w / s.stride))
            for s in specs]


def test_registered():
    assert DECODERS["unet"] is UNetDecoder


@pytest.mark.parametrize("specs", [SPECS5, SPECS4], ids=["5lvl", "4lvl"])
def test_default_output_is_full_resolution(specs):
    dec = UNetDecoder(specs)
    out = dec(feats(specs, 64, 96))
    assert dec.out_spec.stride == 1
    assert out.shape == (2, dec.out_spec.channels, 64, 96)


def test_default_channels_match_smp():
    dec = UNetDecoder(SPECS5)
    assert dec.channels == (256, 128, 64, 32, 16)


@pytest.mark.parametrize("out_stride", [1, 2, 4])
def test_out_stride(out_stride):
    dec = UNetDecoder(SPECS4, out_stride=out_stride)
    out = dec(feats(SPECS4, 64, 64))
    assert dec.out_spec.stride == out_stride
    assert out.shape[-2:] == (64 // out_stride, 64 // out_stride)


def test_channels_length_validated():
    with pytest.raises(ValueError, match="channels"):
        UNetDecoder(SPECS5, channels=(64, 32))


def test_odd_skip_sizes_aligned():
    """Upsampled features are resized to each skip's exact (ceil-rounded) size."""
    dec = UNetDecoder(SPECS5).eval()
    out = dec(feats(SPECS5, 65, 97))
    assert out.shape[-2:] == (66, 98)  # 2 x ceil(h/2)


def test_all_parameters_get_gradients():
    dec = UNetDecoder(SPECS5)
    dec(feats(SPECS5, 64, 64)).square().mean().backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_gradcheck_double():
    specs = (FeatureSpec(3, 2), FeatureSpec(4, 4))
    dec = UNetDecoder(specs, channels=(4, 3), norm=None).double()
    xs = [torch.randn(1, s.channels, 8 // s.stride, 8 // s.stride, dtype=torch.double,
                      requires_grad=True) for s in specs]
    assert torch.autograd.gradcheck(lambda *a: dec(list(a)), xs)


@pytest.mark.parametrize("hw, size", [((16, 16), (32, 32)), ((8, 13), (16, 26)),
                                      ((8, 13), (15, 25)), ((5, 7), (9, 14)), ((1, 1), (1, 2))])
def test_fast_2x_upsample_equals_integer_gather(hw, size):
    """E025: the scale_factor=2 path gives the gather's exact values (forward) and gradients
    that agree up to summation order, for 2n and 2n - 1 outputs."""
    from manyseg.utils.spatial import is_2x, resize_nearest, upsample2x_nearest

    assert is_2x(hw[0], size[0]) and is_2x(hw[1], size[1])
    x = torch.randn(2, 3, *hw, requires_grad=True)
    g = torch.randn(2, 3, *size)
    a, b = upsample2x_nearest(x, size), resize_nearest(x, size)
    assert torch.equal(a, b)
    ga, = torch.autograd.grad(a, x, g)
    gb, = torch.autograd.grad(b, x, g)
    torch.testing.assert_close(ga, gb)


def test_fast_upsample_keeps_channels_last():
    from manyseg.utils.spatial import upsample2x_nearest

    x = torch.randn(1, 8, 5, 6).to(memory_format=torch.channels_last)
    y = upsample2x_nearest(x, (10, 11))
    assert y.shape == (1, 8, 10, 11) and y.is_contiguous(memory_format=torch.channels_last)


def test_padded_models_mark_exact_2x_blocks():
    from manyseg import create_model

    padded = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    assert all(b.exact_2x for b in padded.decoder.blocks)
    unpadded = create_model(backbone="resnet18", pretrained=False, num_classes=2,
                            pad_to_stride=False)
    assert not any(b.exact_2x for b in unpadded.decoder.blocks)
    task = create_model(backbone="resnet18", pretrained=False,
                        heads={"a": {"type": "segmentation", "out_channels": 2,
                                     "decoder": "unet"}})
    assert all(b.exact_2x for b in task.task_decoders["a"].blocks)
    x = torch.randn(1, 3, 67, 93)
    padded.eval(), unpadded.load_state_dict(padded.state_dict()), unpadded.eval()
    with torch.no_grad():
        assert padded(x).shape == unpadded(x).shape == (1, 2, 67, 93)
