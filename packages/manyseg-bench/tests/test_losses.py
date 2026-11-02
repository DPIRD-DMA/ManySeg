import pytest
import torch
from manyseg_bench import losses


def test_dice_bce_perfect_is_low_and_wrong_is_high():
    t = (torch.rand(2, 1, 16, 16) > 0.5).float()
    good = losses.dice_bce(t * 20 - 10, t)
    bad = losses.dice_bce(-(t * 20 - 10), t)
    assert good < 0.01 and bad > 1.0


def test_dice_bce_pos_weight_increases_loss_on_missed_positives():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 3, :] = 1
    logits = torch.zeros(1, 1, 8, 8) - 2
    assert losses.dice_bce(logits, t, pos_weight=10.0) > losses.dice_bce(logits, t)


def test_masked_regression_ignores_outside_mask():
    p, t = torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4)
    m = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    m[..., 0, 0] = True
    assert losses.masked_regression(p, t, m, kind="l1") == pytest.approx(1.0)
    assert losses.masked_regression(p, t, torch.zeros_like(m)) == 0


def test_heatmap_focal_prefers_correct_peaks():
    t = torch.zeros(1, 1, 9, 9)
    t[..., 4, 4] = 1
    good = torch.full_like(t, -6.0)
    good[..., 4, 4] = 6
    assert losses.heatmap_focal(good, t) < losses.heatmap_focal(-good, t)


def test_losses_differentiable():
    p = torch.randn(1, 1, 8, 8, requires_grad=True)
    t = (torch.rand(1, 1, 8, 8) > 0.5).float()
    (losses.dice_bce(p, t) + losses.heatmap_focal(p, t * 0.5)
     + losses.masked_regression(p, t, t > 0)).backward()
    assert torch.isfinite(p.grad).all()


def test_heatmap_focal_explicit_positives():
    """Centres between pixels never reach 1; passing ``pos`` keeps the positive term."""
    t = torch.zeros(1, 8, 8)
    t[0, 3:5, 3:5] = 0.94
    pos = t > 0.9
    good = torch.where(pos, 4.0, -4.0)
    assert losses.heatmap_focal(good, t, pos=pos) < losses.heatmap_focal(-good, t, pos=pos)
    assert not (t >= 0.999).any()  # the default would find no positives here


def test_reexports_the_library_losses():
    import manyseg.losses

    for name in manyseg.losses.__all__:
        assert getattr(losses, name) is getattr(manyseg.losses, name)
