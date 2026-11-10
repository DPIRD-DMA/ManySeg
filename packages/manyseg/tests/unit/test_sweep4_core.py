"""Sweep 4 core fixes: loss shape checks, float32 metadata encoding in reduced-precision
models, FastaiUNet's one level per stride; pixel-head output shapes."""

import pytest
import torch
from manyseg import create_model
from manyseg.decoders import DECODERS
from manyseg.features import FeatureSpec
from manyseg.losses import dice_bce, heatmap_focal, masked_regression, soft_dice

# --- pixel-head output shapes ----------------------------------------------------------------

@pytest.mark.parametrize("scale", [1, 0.5, 0.25, 0.125])
@pytest.mark.parametrize("hw", [(64, 64), (65, 97), (101, 70), (33, 127)])
def test_pixel_head_output_shapes_unchanged(scale, hw):
    m = create_model("resnet18", pretrained=False, heads={"p": {
        "type": "regression", "out_channels": 1, "scale": scale}}).eval()
    f = round(1 / scale)
    with torch.no_grad():
        y = m(torch.randn(1, 3, *hw))["p"]
    assert tuple(y.shape[-2:]) == (-(-hw[0] // f), -(-hw[1] // f))


# --- C2: losses reject a target shaped unlike the prediction ---------------------------------

@pytest.mark.parametrize("fn", [soft_dice, dice_bce, masked_regression, heatmap_focal])
def test_losses_reject_mismatched_target(fn):
    p = torch.randn(2, 1, 8, 8)
    t = (torch.rand(2, 8, 8) > 0.5).float()
    with pytest.raises(ValueError, match="does not match"):
        fn(p, t)
    assert torch.isfinite(fn(p, t[:, None]))  # the matching shape works


@pytest.mark.parametrize("fn", [soft_dice, dice_bce, masked_regression])
def test_losses_keep_mask_broadcasting(fn):
    p = torch.randn(2, 3, 8, 8)
    t = (torch.rand(2, 3, 8, 8) > 0.5).float()
    m = torch.rand(2, 8, 8) > 0.3
    a = fn(p, t, mask=m)
    b = fn(p, t, mask=m[:, None].expand(-1, 3, -1, -1))
    torch.testing.assert_close(a, b)


def test_heatmap_focal_validates_pos():
    p, t = torch.randn(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    with pytest.raises(ValueError, match="pos shape"):
        heatmap_focal(p, t, pos=t[:, 0] > 0.9)
    with pytest.raises(ValueError, match="bool"):
        heatmap_focal(p, t, pos=(t > 0.9).float())
    assert torch.isfinite(heatmap_focal(p, t, pos=t > 0.9))


# --- C3: metadata is validated and encoded in float32 whatever the model dtype ---------------

def _meta_model(**fields):
    return create_model("resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"in_channels": 3},
        "meta": {"kind": "vector", "fields": fields}}).eval()


def test_bfloat16_model_takes_float32_metadata_and_keeps_ids():
    torch.manual_seed(0)
    m = _meta_model(tile={"type": "categorical", "num": 1000}, loc={"type": "coords"},
                    doy={"type": "cyclic", "period": 365})
    enc = m.conditioner.encoders["meta"]
    v = torch.tensor([[257.0, 52.13, 4.37, 200.0], [256.0, 52.13, 4.37, 200.0]])
    ref = enc.encode_fields(v)
    m = m.to(torch.bfloat16)
    got = enc.encode_fields(v)
    assert got.dtype == torch.float32
    # ids 257 and 256 stay distinct; every encoding matches the float32 model's (the bf16
    # embedding table is the only rounding)
    assert not torch.equal(got[0, :8], got[1, :8])
    torch.testing.assert_close(got[:, 8:], ref[:, 8:])
    torch.testing.assert_close(got[:, :8], ref[:, :8], atol=1e-2, rtol=1e-2)
    with torch.no_grad():
        y = m({"img": torch.randn(2, 3, 64, 64, dtype=torch.bfloat16), "meta": v})
    assert y.dtype == torch.bfloat16
    with torch.no_grad():  # any float dtype is accepted
        m({"img": torch.randn(2, 3, 64, 64, dtype=torch.bfloat16), "meta": v.double()})


@pytest.mark.parametrize("mode", ["film", {"mode": "head", "pixel": True}, "channels"])
def test_bfloat16_model_each_conditioning_mode(mode):
    m = create_model("resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"in_channels": 3},
        "meta": {"kind": "vector", "conditioning": mode,
                 "fields": {"tile": {"type": "categorical", "num": 300}}}}).eval()
    m = m.to(torch.bfloat16)
    with torch.no_grad():
        y = m({"img": torch.randn(1, 3, 64, 64, dtype=torch.bfloat16),
               "meta": torch.tensor([[257.0]])})
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


# --- C4: FastaiUNet reads one level per stride, the last ---------------------------------------

def test_fastai_unet_dedupes_repeated_strides():
    F = FeatureSpec
    specs = [F(3, 1, "input"), F(64, 2), F(64, 4), F(128, 8), F(256, 16), F(512, 16)]
    d = DECODERS["fastai_unet"](specs).eval()
    assert d.deep == 5 and d.skips == [3, 2, 1]
    assert d.read_levels == [0, 1, 2, 4]
    feats = [torch.randn(1, 3, 64, 64)] + [
        torch.randn(1, s.channels, 64 // s.stride, 64 // s.stride) for s in specs[1:]]
    assert d(feats).shape[-2:] == (64, 64)


def test_fastai_unet_dilated_encoder_runs():
    m = create_model("resnet18", pretrained=False, decoder="fastai_unet", num_classes=2,
                     backbone_kwargs={"output_stride": 16}).eval()
    assert [m.feature_specs[i - 1].stride for i in m.decoder.skips] == [8, 4, 2]
    with torch.no_grad():
        assert m(torch.randn(1, 3, 70, 90)).shape[-2:] == (70, 90)


def test_fastai_unet_parameters_unchanged_for_plain_encoder():
    """resnet34 without output_stride: one level per stride already, fastai's 19.9M decoder."""
    m = create_model("resnet34", pretrained=False, decoder="fastai_unet", num_classes=2)
    assert sum(p.numel() for p in m.decoder.parameters()) == 19936296
