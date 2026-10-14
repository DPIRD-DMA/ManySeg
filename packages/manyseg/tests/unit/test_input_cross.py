"""U-Net `input_cross`: fastai-style full-resolution stage (concat the input image + residual
block) as an option on the SMP-equivalent U-Net (E006 follow-up)."""

import pytest
import torch
from manyseg import create_model


def model(**dkw):
    return create_model(backbone="resnet18", pretrained=False, num_classes=2,
                        decoder_kwargs=dkw)


def test_default_unchanged():
    assert not getattr(model().decoder, "wants_input", False)


@pytest.mark.parametrize("hw", [(64, 64), (65, 97)])
def test_shapes_and_gradients(hw):
    m = model(input_cross=True)
    assert m.decoder.wants_input
    assert m.decoder.out_spec.stride == 1
    assert m.decoder.out_spec.channels == 16 + 3  # last block width + input channels
    out = m(torch.randn(2, 3, *hw))
    assert out.shape == (2, 2, *hw)
    out.mean().backward()
    assert all(p.grad is not None for p in m.decoder.cross.parameters())


def test_param_cost_is_small():
    base = sum(p.numel() for p in model().parameters())
    cross = sum(p.numel() for p in model(input_cross=True).parameters())
    assert 0 < cross - base < 10_000  # two 3x3 convs at 19 channels


def test_input_reaches_output_directly():
    """The raw input feeds the output stage, so a one-pixel input change moves that pixel."""
    torch.manual_seed(0)
    m = model(input_cross=True).eval()
    x = torch.randn(1, 3, 64, 64)
    x2 = x.clone()
    x2[0, :, 30, 30] += 5.0
    d = (m(x2) - m(x)).abs().sum(1)[0]
    assert d[30, 30] > 0


def test_requires_full_resolution():
    with pytest.raises(ValueError, match="out_stride"):
        model(input_cross=True, out_stride=2)


def test_in_channels_respected():
    m = create_model(backbone="resnet18", pretrained=False, num_classes=2, in_channels=5,
                     decoder_kwargs={"input_cross": True})
    assert m.decoder.out_spec.channels == 16 + 5
    assert m(torch.randn(1, 5, 64, 64)).shape == (1, 2, 64, 64)


def test_multi_input_rejected_clearly():
    with pytest.raises(ValueError, match="input image"):
        create_model(inputs={"a": {"channels": 3}, "b": {"channels": 2}}, backbone="resnet18",
                     pretrained=False, num_classes=2, decoder_kwargs={"input_cross": True})


@pytest.mark.slow
def test_export_input_cross():
    from manyseg.export import export_model

    m = model(input_cross=True).eval()
    ep = export_model(m, torch.randn(1, 3, 64, 96))
    x = torch.randn(1, 3, 97, 131)
    torch.testing.assert_close(ep.module()(x), m(x), atol=1e-4, rtol=1e-4)
