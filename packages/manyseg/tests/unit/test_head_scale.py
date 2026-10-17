"""Per-head output ``scale``: pixel heads may predict on a coarser grid than the input.

Contract: a head with ``scale = 1/f`` (f in {1, 2, 4, 8}) returns ``ceil(H / f) x ceil(W / f)``
for a primary input of ``H x W``; output pixel ``(i, j)`` covers input pixels
``[i*f, (i+1)*f) x [j*f, (j+1)*f)`` (the last row/column may be partial).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from manyseg import FeatureSpec, create_model
from manyseg.decoders import DECODERS
from manyseg.heads import PixelHead
from manyseg.models import DenseModel
from manyseg.utils import downsample_mean
from torch import nn

SCALES = [1, 0.5, 0.25, 0.125]
SIZES = [(64, 64), (65, 97), (33, 200)]


def ceil_size(hw, s):
    f = round(1 / s)
    return math.ceil(hw[0] / f), math.ceil(hw[1] / f)


def scaled_heads(channels=2, scales=SCALES):
    return {f"s{round(1 / s)}": {"type": "segmentation", "out_channels": channels, "scale": s}
            for s in scales}


@pytest.mark.parametrize("decoder,out_stride", [(d, None) for d in sorted(DECODERS)]
                         + [(d, s) for d in ("unet", "unetplusplus", "fastai_unet")
                            for s in (2, 4)])
def test_shapes_every_decoder_out_stride_and_scale(decoder, out_stride):
    dk = {} if out_stride is None else {"out_stride": out_stride}
    model = create_model(backbone="resnet18", pretrained=False, decoder=decoder,
                         decoder_kwargs=dk, heads=scaled_heads()).eval()
    for hw in SIZES:
        out = model(torch.randn(1, 3, *hw))
        for s in SCALES:
            assert out[f"s{round(1 / s)}"].shape == (1, 2, *ceil_size(hw, s)), (hw, s)


@pytest.mark.parametrize("s", SCALES)
def test_odd_sizes_round_up(s):
    """ceil: every input pixel belongs to exactly one output pixel."""
    model = create_model(backbone="resnet18", pretrained=False, heads=scaled_heads(scales=[s]))
    out = model.eval()(torch.randn(1, 3, 65, 97))
    f = round(1 / s)
    assert out[f"s{f}"].shape[-2:] == (-(-65 // f), -(-97 // f))


def test_default_scale_is_unchanged():
    torch.manual_seed(0)
    a = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    torch.manual_seed(0)
    b = create_model(backbone="resnet18", pretrained=False,
                     heads={"y": {"type": "segmentation", "out_channels": 3, "scale": 1}}).eval()
    x = torch.randn(1, 3, 65, 97)
    torch.testing.assert_close(a(x), b(x)["y"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", [0, 0.3, 2, 1 / 16, -0.5, "half"])
def test_invalid_scale_rejected(bad):
    with pytest.raises(ValueError, match="scale"):
        create_model(backbone="resnet18", pretrained=False,
                     heads={"y": {"type": "segmentation", "out_channels": 1, "scale": bad}})


def test_image_heads_reject_scale():
    with pytest.raises(ValueError, match="scale"):
        create_model(backbone="resnet18", pretrained=False,
                     heads={"y": {"type": "classification", "out_channels": 3, "scope": "image",
                                  "scale": 0.5}})


def _count_interpolations(monkeypatch):
    import manyseg.heads.heads as hmod

    calls = []
    real = hmod.F.interpolate

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(hmod.F, "interpolate", spy)
    return calls


def test_head_skips_upsampling_when_stride_matches(monkeypatch):
    """Decoder stride == 1/scale: the head is just its conv (this is the saving)."""
    head = PixelHead(FeatureSpec(16, 4), 2, scale=0.25)
    x = torch.randn(1, 16, 8, 12)
    calls = _count_interpolations(monkeypatch)
    y = head(x)
    assert calls == [] and y.shape == (1, 2, 8, 12)
    torch.testing.assert_close(y, head.conv(x))


def test_head_upsamples_by_stride_times_scale():
    head = PixelHead(FeatureSpec(16, 4), 2, scale=0.5)
    assert head(torch.randn(1, 16, 8, 12)).shape == (1, 2, 16, 24)


def test_head_downsamples_when_decoder_is_finer():
    """A stride-1 decoder feeding a 1/4 head: area-average, then the conv on the coarse grid."""
    head = PixelHead(FeatureSpec(16, 1), 2, scale=0.25)
    x = torch.randn(1, 16, 32, 48)
    y = head(x)
    assert y.shape == (1, 2, 8, 12)
    torch.testing.assert_close(y, head.conv(F.avg_pool2d(x, 4)))


def test_pad_to_stride_false_uses_ceil_size():
    model = create_model(backbone="resnet18", pretrained=False, pad_to_stride=False,
                         heads=scaled_heads()).eval()
    out = model(torch.randn(1, 3, 65, 97))
    for s in SCALES:
        assert out[f"s{round(1 / s)}"].shape[-2:] == ceil_size((65, 97), s)


# --- known answer: output pixel (i, j) is the area [i*f, (i+1)*f) x [j*f, (j+1)*f) -------------

class PoolBackbone(nn.Module):
    """Features = the input area-averaged to strides 2, 4, 8 (exact block means)."""

    def __init__(self, c=1):
        super().__init__()
        self.feature_specs = tuple(FeatureSpec(c, s) for s in (2, 4, 8))
        self.input_divisor = 8

    def forward(self, x):
        return [F.avg_pool2d(x, s) for s in (2, 4, 8)]


class Pick(nn.Module):
    def __init__(self, specs, stride):
        super().__init__()
        self.i = [s.stride for s in specs].index(stride)
        self.out_spec = specs[self.i]

    def forward(self, feats):
        return feats[self.i]


def identity_head(spec, scale):
    head = PixelHead(spec, 1, kernel_size=1, scale=scale)
    with torch.no_grad():
        head.conv.weight.fill_(1.0)
        head.conv.bias.zero_()
    head.kind, head.scope = "segmentation", "pixel"
    return head


@pytest.mark.parametrize("dec_stride,scale", [(2, 0.5), (4, 0.25), (8, 0.125), (2, 0.25),
                                              (2, 0.125), (4, 0.125)])
def test_known_answer_block_means(dec_stride, scale):
    bb = PoolBackbone()
    dec = Pick(bb.feature_specs, dec_stride)
    model = DenseModel(bb, dec, {"y": identity_head(dec.out_spec, scale)}, single_output="y")
    x = torch.randn(1, 1, 48, 80)
    torch.testing.assert_close(model(x), downsample_mean(x, scale))
    # a single bright pixel lands in output pixel (y // f, x // f)
    f = round(1 / scale)
    x = torch.zeros(1, 1, 48, 80)
    x[0, 0, 21, 37] = 1.0
    out = model(x)[0, 0]
    assert (out > 0).nonzero().tolist() == [[21 // f, 37 // f]]


def test_known_answer_odd_size_partial_blocks_use_the_zero_padding():
    """Odd sizes: the last (partial) output pixel averages the in-image pixels and the zero
    padding of pad_to_stride, i.e. it equals the block mean of the explicitly padded input."""
    bb = PoolBackbone()
    dec = Pick(bb.feature_specs, 4)
    model = DenseModel(bb, dec, {"y": identity_head(dec.out_spec, 0.25)}, single_output="y")
    x = torch.randn(1, 1, 45, 77)
    out = model(x)
    assert out.shape[-2:] == (12, 20)
    torch.testing.assert_close(out, F.avg_pool2d(F.pad(x, (0, 3, 0, 3)), 4))


# --- alignment (gradient centroid), cf. test_model._mean_offset -------------------------------

def _centroid_offset(model, f, size=128, n=36):
    """Mean (dy, dx) between the documented centre of output pixel (i, j) in input pixels,
    ``(i + 0.5) * f - 0.5``, and the centroid of its input-gradient magnitude.

    One batched backward pass: sample ``b`` probes its own output pixel (eval mode, so samples
    are independent). Pixels cover the central input region at every phase modulo 8."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    c = torch.randint(size // 2 - 20, size // 2 + 20, (n, 2), generator=g) // f
    x = torch.randn(n, 3, size, size, generator=g, requires_grad=True)
    out = model(x)
    out[torch.arange(n), :, c[:, 0], c[:, 1]].sum().backward()
    gm = x.grad.abs().sum(1)                       # (n, H, W)
    ys = torch.arange(size, dtype=gm.dtype)
    cy = (gm.sum(2) * ys).sum(1) / gm.sum((1, 2))
    cx = (gm.sum(1) * ys).sum(1) / gm.sum((1, 2))
    centre = (c.float() + 0.5) * f - 0.5
    return torch.stack([cy, cx], 1).sub(centre).mean(0).detach()


def _scaled(decoder="unet", scale=1, **dk):
    torch.manual_seed(1)
    model = create_model(backbone="resnet18", pretrained=False, decoder=decoder,
                         decoder_kwargs=dk,
                         heads={"y": {"type": "segmentation", "out_channels": 3, "scale": scale}})
    model.single_output = "y"
    return model


@pytest.mark.parametrize("out_stride,scale", [(1, 1), (2, 1), (4, 1), (2, 0.5), (4, 0.25),
                                              (4, 0.5), (1, 0.25)])
def test_alignment_aligned_upsampling(out_stride, scale):
    """Grid-consistent decoder upsampling: output pixels stay centred on the input area they
    cover at every out_stride and head scale.

    Bound: the stride-1 bound of test_model (1.5 px) plus (out_stride - 1) / 2. A stride-d
    feature from "same"-padded strided convs is centred on input pixel ``k*d`` rather than on
    its block centre ``(k + 0.5) d - 0.5`` (experiments/E001_upsample_alignment); the head's
    bilinear upsampling (and no resampling at all when stride == 1/scale) keeps that offset.
    """
    off = _centroid_offset(_scaled(scale=scale, upsample="aligned", out_stride=out_stride),
                           round(1 / scale))
    assert off.abs().max() < 1.5 + (out_stride - 1) / 2, off


@pytest.mark.parametrize("decoder", ["unet", "fastai_unet"])
@pytest.mark.parametrize("out_stride,scale", [(2, 1), (4, 1), (4, 0.25)])
def test_alignment_default_upsampling_bias_stays_bounded(decoder, out_stride, scale):
    """Regression guard: reduced strides keep the known up-left bias of nearest / pixel-shuffle
    upsampling (about 6 px at init at stride 1, test_model) within the same bound."""
    off = _centroid_offset(_scaled(decoder, scale, out_stride=out_stride), round(1 / scale))
    assert (off < 0).all() and off.abs().max() < 8, off


# --- multi-input, task decoders, conditioning --------------------------------------------------

def test_multi_input_scaled_heads_follow_the_primary_grid():
    inputs = {"optical": {"in_channels": 4}, "sar": {"in_channels": 2, "alignment": "same_extent"}}
    model = create_model(inputs=inputs, backbone="resnet18", pretrained=False,
                         heads=scaled_heads()).eval()
    out = model({"optical": torch.randn(1, 4, 65, 97), "sar": torch.randn(1, 2, 33, 49)})
    for s in SCALES:
        assert out[f"s{round(1 / s)}"].shape[-2:] == ceil_size((65, 97), s)


def test_task_decoders_with_scales_and_out_strides():
    heads = {"seg": {"type": "segmentation", "out_channels": 3, "scale": 0.25},
             "edges": {"type": "segmentation", "out_channels": 1,
                       "decoder": {"type": "unet", "out_stride": 2}},
             "density": {"type": "regression", "out_channels": 1, "scale": 0.125,
                         "decoder": {"type": "fastai_unet", "out_stride": 4}},
             "area": {"type": "regression", "out_channels": 1, "scope": "image"}}
    model = create_model(backbone="resnet18", pretrained=False, heads=heads,
                         decoder_kwargs={"out_stride": 4}).eval()
    out = model(torch.randn(2, 3, 65, 97))
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "seg": (2, 3, 17, 25), "edges": (2, 1, 65, 97), "density": (2, 1, 9, 13),
        "area": (2, 1)}


def test_film_identity_at_init_with_scaled_head():
    vec = {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}}
    heads = {"y": {"type": "segmentation", "out_channels": 2, "scale": 0.25}}
    kw = dict(backbone="resnet18", pretrained=False, heads=heads,
              decoder_kwargs={"out_stride": 4})
    cond = create_model(inputs={"img": {"in_channels": 3}, "meta": vec}, **kw).eval()
    plain = create_model(**kw)
    plain.load_state_dict({k: v for k, v in cond.state_dict().items()
                           if not k.startswith("conditioner.")})
    x = torch.randn(2, 3, 33, 47)
    out = cond({"img": x, "meta": torch.tensor([[10.0], [200.0]])})
    torch.testing.assert_close(out["y"], plain.eval()(x)["y"], rtol=0, atol=0)
    assert out["y"].shape[-2:] == (9, 12)


def test_scaled_head_gradients_and_training_step():
    model = create_model(backbone="resnet18", pretrained=False, heads=scaled_heads(1))
    out = model(torch.randn(2, 3, 64, 64))
    sum(v.square().mean() for v in out.values()).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_image_head_init_std():
    """``init_std``: truncated-normal weights and zero bias (timm's ViT/ConvNeXt/PVT classifier
    init); default keeps PyTorch's Linear init."""
    torch.manual_seed(0)
    heads = {"cls": {"type": "classification", "out_channels": 37, "scope": "image",
                     "init_std": 0.02}}
    fc = create_model(backbone="resnet18", pretrained=False, heads=heads).heads["cls"].fc
    assert torch.all(fc.bias == 0) and abs(float(fc.weight.std()) - 0.02) < 0.002
    plain = create_model(backbone="resnet18", pretrained=False, heads={
        "cls": {"type": "classification", "out_channels": 37, "scope": "image"}}).heads["cls"].fc
    assert float(plain.bias.abs().sum()) > 0
