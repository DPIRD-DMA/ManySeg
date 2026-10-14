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
