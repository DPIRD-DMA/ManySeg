"""E025: decoders replace the integer-index nearest gather (slow scatter-add backward) with
``F.interpolate(scale_factor=2)`` or a crop where the sizes allow it. Both paths must give the
same model outputs and gradients, padded or not, eager or compiled."""

import pytest
import torch
from manyseg import create_model

DECODERS = ["unet", "unetplusplus", "fpn", "fastai_unet"]


def force_gather(monkeypatch, model):
    """Every nearest resize back on the gather: no static flags, no eager size checks."""
    import manyseg.decoders.fastai_unet as fa
    import manyseg.decoders.fpn as fpn
    import manyseg.decoders.unet as un
    import manyseg.utils.spatial as sp

    for mod in (un, fpn, fa):
        monkeypatch.setattr(mod, "is_2x", lambda *a: False)
    monkeypatch.setattr(sp, "_crops", lambda *a: False)
    for m in model.modules():
        for flag in ("exact_2x", "exact", "_exact", "_exact_2x"):
            if isinstance(getattr(m, flag, None), bool):
                setattr(m, flag, False)
    monkeypatch.setattr(un.UNetBlock, "forward", _gather_unet_block)


def _gather_unet_block(self, x, skip, size=None):
    """UNetBlock.forward without its "no skip: exactly 2x" shortcut."""
    import manyseg.decoders.unet as un

    if size is None:
        size = skip.shape[-2:] if skip is not None else (2 * x.shape[-2], 2 * x.shape[-1])
    x = un.upsample2x(x, size, self.upsample, False)
    if skip is not None:
        x = torch.cat([x, skip], dim=1)
    return self.conv2(self.conv1(x))


def flags_only(monkeypatch):
    """What torch.compile / export run: the static flags, no eager size checks."""
    import manyseg.decoders.fastai_unet as fa
    import manyseg.decoders.fpn as fpn
    import manyseg.decoders.unet as un
    import manyseg.utils.spatial as sp

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    for mod in (un, fpn, fa):
        monkeypatch.setattr(mod, "is_2x", lambda *a: False)
    monkeypatch.setattr(sp, "_crops", lambda *a: False)


CASES = [("resnet18", None), ("convnextv2_femto", None),  # stride-2 level or not
         ("resnet18", [1, 3, 4]), ("resnet18", [2, 4])]  # sparse levels: 4x steps


@pytest.mark.parametrize("decoder", DECODERS)
@pytest.mark.parametrize("backbone, out_indices", CASES,
                         ids=["resnet18", "convnextv2", "strides-4-16-32", "strides-8-32"])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("traced", [False, True], ids=["eager", "flags-only"])
def test_fast_nearest_paths_match_the_gather(decoder, backbone, out_indices, pad, traced,
                                             monkeypatch):
    torch.manual_seed(0)
    model = create_model(backbone=backbone, pretrained=False, num_classes=3, decoder=decoder,
                         pad_to_stride=pad, out_indices=out_indices).double().eval()
    x = torch.randn(2, 3, 67, 93, dtype=torch.float64, requires_grad=True)

    def run():
        y = model(x)
        g, = torch.autograd.grad(y.square().sum(), x)
        return y.detach(), g

    with monkeypatch.context() as mp:
        if traced:
            flags_only(mp)
        fast = run()
    with monkeypatch.context() as mp:
        force_gather(mp, model)
        slow = run()
    torch.testing.assert_close(fast[0], slow[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(fast[1], slow[1], rtol=1e-10, atol=1e-10)


def _flags(decoder):
    name = type(decoder).__name__
    if name == "UNetDecoder":
        return [b.exact_2x for b in decoder.blocks]
    if name == "UNetPlusPlusDecoder":
        return [n.exact_2x for n in decoder.nodes.values()]
    if name == "FPNDecoder":
        return list(decoder._exact_2x)
    return [decoder._exact, *(b.exact for b in decoder.blocks)]


@pytest.mark.parametrize("decoder", DECODERS)
def test_padded_models_flag_their_decoders(decoder):
    m = create_model(backbone="resnet18", pretrained=False, num_classes=2, decoder=decoder)
    assert all(_flags(m.decoder))
    u = create_model(backbone="resnet18", pretrained=False, num_classes=2, decoder=decoder,
                     pad_to_stride=False)
    assert not any(_flags(u.decoder))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fast_upsample_keeps_bf16_under_autocast(device):
    """Autocast runs upsample_nearest2d in float32; the fast path casts back (lossless), so a
    bf16 decoder does not carry float32 maps (E025: ~10% more training memory otherwise)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from manyseg.utils.spatial import resize_nearest, upsample2x_nearest

    x = torch.randn(2, 4, 8, 9, device=device, dtype=torch.bfloat16)
    with torch.autocast(device, dtype=torch.bfloat16):
        y = upsample2x_nearest(x, (16, 17))
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, resize_nearest(x, (16, 17)))
