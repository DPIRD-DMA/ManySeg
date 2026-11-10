"""Regressions: NaN under masks, (B, H, W) masks, group norm widths, per-field metadata widths."""

import pytest
import torch
from manyseg.conditioning.encoder import MetadataEncoder
from manyseg.layers import make_norm
from manyseg.losses import dice_bce, masked_regression, soft_dice
from manyseg.utils.resolution import downsample_mean

NAN = float("nan")


# --- masked NaN targets -------------------------------------------------------------------

def test_masked_regression_ignores_nan_under_mask():
    for kind in ("l1", "mse"):
        pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = masked_regression(pred, torch.tensor([[0.0, NAN]]),
                                 torch.tensor([[True, False]]), kind=kind)
        assert loss.item() == pytest.approx(1.0)
        loss.backward()
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[0, 1] == 0


def test_masked_regression_empty_mask_is_zero():
    pred = torch.ones(1, 1, 2, 2, requires_grad=True)
    loss = masked_regression(pred, torch.full((1, 1, 2, 2), NAN),
                             torch.zeros(1, 1, 2, 2, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("fn", [soft_dice, lambda x, t, mask: dice_bce(x, t, mask=mask)])
def test_dice_losses_ignore_nan_under_mask(fn):
    logits = torch.randn(2, 1, 4, 4)
    target = (torch.rand(2, 1, 4, 4) > 0.5).float()
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    mask[:, :, :2] = False
    clean = fn(logits, target, mask=mask)
    t_nan, x_nan = target.clone(), logits.clone()
    t_nan[~mask] = NAN
    x_nan[~mask] = NAN
    x_nan.requires_grad_(True)
    loss = fn(x_nan, t_nan, mask=mask)
    assert torch.isfinite(loss) and loss.item() == pytest.approx(clean.item())
    loss.backward()
    assert torch.isfinite(x_nan.grad).all()
    assert (x_nan.grad[~mask] == 0).all()
    # empty mask: zero
    empty = torch.zeros_like(mask)
    assert fn(x_nan.detach(), t_nan, mask=empty).item() == pytest.approx(0.0)


def test_downsample_mean_ignores_nan_under_valid():
    x = torch.tensor([[1.0, 2.0], [3.0, NAN]])
    valid = torch.tensor([[True, True], [True, False]])
    assert downsample_mean(x, 0.5, valid).item() == pytest.approx(2.0)


# --- (B, H, W) masks ----------------------------------------------------------------------

def test_masked_regression_bhw_mask_selects_samples():
    pred = torch.zeros(2, 2, 4, 4)
    target = torch.zeros(2, 2, 4, 4)
    target[1] = 10.0
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[1] = True
    assert masked_regression(pred, target, mask).item() == pytest.approx(10.0)
    assert masked_regression(pred, target, mask[:, None]).item() == pytest.approx(10.0)
    # B != C
    pred3, target3 = torch.zeros(3, 2, 4, 4), torch.zeros(3, 2, 4, 4)
    target3[2] = 4.0
    mask3 = torch.zeros(3, 4, 4, dtype=torch.bool)
    mask3[2] = True
    assert masked_regression(pred3, target3, mask3).item() == pytest.approx(4.0)


def test_soft_dice_bhw_mask_matches_b1hw():
    logits = torch.randn(2, 1, 4, 4)
    target = (torch.rand(2, 1, 4, 4) > 0.5).float()
    mask = torch.rand(2, 4, 4) > 0.3
    for fn in (soft_dice, lambda x, t, mask: dice_bce(x, t, mask=mask)):
        assert fn(logits, target, mask=mask).item() == pytest.approx(
            fn(logits, target, mask=mask[:, None]).item())


@pytest.mark.parametrize("shape", [(4, 4), (2, 4), (1, 4, 4), (2, 3, 4, 4), (2, 1, 1, 4, 4)])
def test_bad_mask_shapes_raise(shape):
    pred, target = torch.zeros(2, 2, 4, 4), torch.zeros(2, 2, 4, 4)
    mask = torch.ones(shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask must be"):
        masked_regression(pred, target, mask)
    with pytest.raises(ValueError, match="mask must be"):
        soft_dice(pred, target, mask)
    with pytest.raises(ValueError, match="mask must be"):
        dice_bce(pred, target, mask=mask)


# --- group norm ---------------------------------------------------------------------------

def test_group_norm_groups():
    from manyseg.layers import group_norm_groups

    assert [group_norm_groups(c) for c in (8, 16, 24, 32, 64, 256)] == [8, 16, 24, 32, 32, 32]
    assert group_norm_groups(48) == 24
    assert group_norm_groups(37) == 1
    assert make_norm("group", 48).num_groups == 24


def test_group_norm_decoder_width_48_builds():
    from manyseg import create_model
    m = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2,
                     decoder_kwargs={"channels": [48] * 5, "norm": "group"})
    out = m(torch.randn(1, 3, 64, 64))
    out = out if isinstance(out, torch.Tensor) else next(iter(out.values()))
    assert out.shape[-2:] == (64, 64)


# --- pack_metadata ------------------------------------------------------------------------

def _enc():
    return MetadataEncoder({"location": {"type": "coords"},
                            "temperature": {"type": "continuous"}})


def test_pack_checks_each_field_width():
    with pytest.raises(ValueError, match="'location'"):
        _enc().pack({"location": torch.tensor([10.0]), "temperature": torch.tensor([[20.0, 30.0]])})
    with pytest.raises(ValueError, match="'temperature'"):
        _enc().pack({"location": torch.tensor([[1.0, 2.0]]),
                     "temperature": torch.tensor([[20.0, 30.0]])})


def test_pack_checks_batch_size():
    with pytest.raises(ValueError, match="batch size"):
        _enc().pack({"location": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                     "temperature": torch.tensor([20.0])})


def test_pack_valid_shapes():
    enc = _enc()
    loc = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    a = enc.pack({"location": loc, "temperature": torch.tensor([20.0, NAN])})
    b = enc.pack({"location": loc, "temperature": torch.tensor([[20.0], [NAN]])})
    assert a.shape == (2, 3)
    assert torch.equal(a.nan_to_num(-1), b.nan_to_num(-1))
