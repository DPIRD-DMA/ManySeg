"""End-to-end contracts of models built with ``create_model``."""

import io

import pytest
import torch
import torch.nn.functional as F

from manyseg import create_model

SIZES = [(64, 64), (65, 97), (33, 200), (127, 31), (32, 32)]


def small(**kw):
    kw.setdefault("backbone", "resnet18")
    kw.setdefault("pretrained", False)
    kw.setdefault("num_classes", 3)
    return create_model(**kw)


@pytest.mark.parametrize("hw", SIZES)
def test_output_matches_input_size(hw):
    model = small().eval()
    assert model(torch.randn(2, 3, *hw)).shape == (2, 3, *hw)


def test_same_instance_many_sizes():
    model = small().eval()
    for hw in SIZES:
        assert model(torch.randn(1, 3, *hw)).shape[-2:] == hw


@pytest.mark.parametrize("c", [1, 4, 13])
def test_in_channels(c):
    assert small(in_channels=c)(torch.randn(1, c, 64, 64)).shape == (1, 3, 64, 64)


@pytest.mark.parametrize(
    "backbone", ["efficientnet_b0", "convnext_atto", "mobilenetv3_small_100", "regnetx_002"]
)
def test_other_cnn_backbones(backbone):
    model = small(backbone=backbone).eval()
    assert model(torch.randn(1, 3, 65, 97)).shape == (1, 3, 65, 97)


def test_padding_semantics_are_bottom_right_zero_pad():
    """Output for an odd-sized input equals the cropped output of its explicit zero padding."""
    model = small().eval()
    x = torch.randn(1, 3, 65, 97)
    padded = F.pad(x, (0, 128 - 97, 0, 96 - 65))
    torch.testing.assert_close(model(x), model(padded)[..., :65, :97])


def test_pad_to_stride_false_still_aligned_shape():
    model = small(pad_to_stride=False).eval()
    assert model(torch.randn(1, 3, 65, 97)).shape[-2:] == (65, 97)


def _mean_offset(model, size=192, step=9):
    """Mean (dy, dx) between output pixels and the centroid of their input-gradient magnitude."""
    model.eval()
    lo, hi = size // 2 - 24, size // 2 + 24
    offs = []
    for i in range(lo, hi, step):
        for j in range(lo + 3, hi, step):
            x = torch.randn(1, 3, size, size, requires_grad=True)
            model(x)[0, :, i, j].sum().backward()
            g = x.grad.abs().sum(dim=(0, 1))
            ys = torch.arange(size, dtype=g.dtype)[:, None]
            xs = torch.arange(size, dtype=g.dtype)[None, :]
            offs.append(((g * ys).sum() / g.sum() - i, (g * xs).sum() / g.sum() - j))
    return torch.tensor(offs).mean(0)


def test_spatial_alignment_aligned_upsampling_is_centred():
    """With grid-consistent upsampling, output pixels are centred on their input pixel."""
    torch.manual_seed(1)
    off = _mean_offset(small(decoder_kwargs={"upsample": "aligned"}))
    assert off.abs().max() < 1.5, off


def test_spatial_alignment_nearest_bias_is_bounded():
    """Regression guard for the known up-left bias of nearest upsampling (E001, ~6 px at init;
    SMP's Unet measures ~10 px). Fails if a change makes it worse."""
    torch.manual_seed(1)
    off = _mean_offset(small())
    assert (off < 0).all() and off.abs().max() < 8, off


def test_gradients_reach_every_parameter():
    model = small(in_channels=4)
    model(torch.randn(2, 4, 64, 64)).logsumexp(1).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_state_dict_roundtrip():
    a = small(in_channels=4).eval()
    b = small(in_channels=4).eval()
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    b.load_state_dict(torch.load(buf, weights_only=True))
    x = torch.randn(1, 4, 64, 64)
    torch.testing.assert_close(a(x), b(x))


def test_multiple_heads_return_dict():
    model = small(
        num_classes=None,
        heads={
            "seg": {"type": "segmentation", "channels": 5},
            "depth": {"type": "regression", "channels": 1},
            "area": {"type": "regression", "channels": 1, "scope": "image"},
            "scene": {"type": "classification", "channels": 7, "scope": "image"},
        },
    ).eval()
    out = model(torch.randn(2, 3, 65, 97))
    assert set(out) == {"seg", "depth", "area", "scene"}
    assert out["seg"].shape == (2, 5, 65, 97)
    assert out["depth"].shape == (2, 1, 65, 97)
    assert out["area"].shape == (2, 1)
    assert out["scene"].shape == (2, 7)


def test_num_classes_and_heads_exclusive():
    with pytest.raises(ValueError):
        small(num_classes=3, heads={"a": {"type": "segmentation", "channels": 2}})


def test_unknown_decoder_lists_options():
    with pytest.raises(ValueError, match="unet"):
        small(decoder="nope")


def test_autocast_bf16_cpu():
    model = small().eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randn(1, 3, 64, 64))
    assert torch.isfinite(out.float()).all()


def test_device_execution(device):
    model = small().to(device).eval()
    out = model(torch.randn(1, 3, 65, 97, device=device))
    assert out.device.type == device.type
    assert out.shape == (1, 3, 65, 97)


def test_device_results_match_cpu(device):
    if device.type == "cpu":
        pytest.skip("reference")
    model = small().eval()
    x = torch.randn(1, 3, 64, 64)
    ref = model(x)
    out = model.to(device)(x.to(device)).cpu()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_fp16_autocast_on_accelerator(device):
    if device.type == "cpu":
        pytest.skip("fp16 autocast is an accelerator path")
    model = small().to(device).eval()
    with torch.autocast(device.type, dtype=torch.float16):
        out = model(torch.randn(2, 3, 64, 96, device=device))
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_channels_last():
    model = small().eval()
    x = torch.randn(1, 3, 64, 64)
    ref = model(x)
    model = model.to(memory_format=torch.channels_last)
    out = model(x.contiguous(memory_format=torch.channels_last))
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
