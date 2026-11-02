"""manyseg.losses: the functions moved from manyseg-bench (tested there through the re-export)
plus multiclass_dice and multi_head_loss."""

import manyseg_bench.losses as bench_losses
import pytest
import torch
import torch.nn.functional as F
from manyseg import losses


def onehot_logits(target, classes, scale=20.0):
    return F.one_hot(target, classes).permute(0, 3, 1, 2).float() * scale - scale / 2


def test_bench_reexports_the_same_functions():
    for name in losses.__all__:
        assert getattr(bench_losses, name) is getattr(losses, name)


def test_multiclass_dice_perfect_wrong_and_gradient():
    torch.manual_seed(0)
    t = torch.randint(0, 4, (2, 16, 16))
    assert losses.multiclass_dice(onehot_logits(t, 4), t) < 1e-3
    wrong = onehot_logits((t + 1) % 4, 4)
    assert losses.multiclass_dice(wrong, t) > 0.99
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    losses.multiclass_dice(x, t[:, None]).backward()  # (B, 1, H, W) targets too
    assert x.grad.abs().sum() > 0


def test_multiclass_dice_ignore_index_and_absent_classes():
    t = torch.zeros(1, 8, 8, dtype=torch.long)
    t[:, :, 4:] = 1
    logits = onehot_logits(t, 3)  # class 2 absent from both target and prediction
    labelled = t.clone()
    labelled[:, :2] = 255
    logits[:, :, :2] = onehot_logits(torch.full((1, 2, 8), 1), 3)  # wrong where ignored
    assert losses.multiclass_dice(logits, labelled, ignore_index=255) < 1e-3
    assert losses.multiclass_dice(logits, t) > 0.05  # counted when not ignored


def test_multiclass_dice_shape_errors():
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        losses.multiclass_dice(torch.randn(4, 8, 8), torch.zeros(4, 8, 8, dtype=torch.long))
    with pytest.raises(ValueError, match="does not match"):
        losses.multiclass_dice(torch.randn(1, 3, 8, 8), torch.zeros(1, 4, 4, dtype=torch.long))


def test_multi_head_loss_weights_parts_and_ignored_heads():
    out = {"a": torch.ones(2, 1, 4, 4, requires_grad=True), "b": torch.zeros(2, 1, 4, 4),
           "aux": torch.zeros(1)}
    tgt = {"a": torch.zeros(2, 1, 4, 4), "b": torch.ones(2, 1, 4, 4)}
    total, parts = losses.multi_head_loss(
        out, tgt, {"a": losses.masked_regression, "b": losses.masked_regression},
        weights={"b": 0.5})
    assert total.item() == pytest.approx(1.0 + 0.5)
    assert parts["a"].item() == pytest.approx(1.0) and parts["b"].item() == pytest.approx(1.0)
    assert not parts["a"].requires_grad
    total.backward()
    assert out["a"].grad is not None


def test_multi_head_loss_closures_and_errors():
    t = torch.zeros(1, 1, 9, 9)
    t[0, 0, 4, 4] = 1.0
    total, _ = losses.multi_head_loss(
        {"corner": torch.full((1, 1, 9, 9), -5.0)}, {"corner": t},
        {"corner": lambda o, y: losses.heatmap_focal(o, y, pos=losses.heatmap_peaks(y))})
    assert torch.isfinite(total)
    with pytest.raises(KeyError, match="missing from outputs"):
        losses.multi_head_loss({}, {"a": t}, {"a": losses.masked_regression})
    with pytest.raises(ValueError, match="without a loss"):
        losses.multi_head_loss({"a": t}, {"a": t}, {"a": losses.masked_regression},
                               weights={"b": 1.0})
    with pytest.raises(ValueError, match="empty"):
        losses.multi_head_loss({"a": t}, {"a": t}, {})


def test_density_loss_is_reexported():
    from manyseg.points import density_loss

    assert losses.density_loss is density_loss
