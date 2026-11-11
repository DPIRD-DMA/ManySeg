"""Regression tests for the pre-release sweep: quantize_pt2e's dynamic batch at fixed H/W,
re-saving a loaded .pt2 program, mixed-dtype save/load, strict ``extra`` metadata,
downsample_mean's mask shapes, and the losses whose docstrings were corrected."""

import warnings

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg.losses import distance_weighted_bce, multiclass_dice, soft_dice
from manyseg.utils.resolution import downsample_mean

KW = {"normalize": [{"type": "none"}], "activations": {"logits": "softmax"}}


def _model():
    torch.manual_seed(0)
    return create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()


# --- B7: per-tensor dtypes round-trip ------------------------------------------------------

def _half_with_float_bn(model):
    model.half()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.float()
    return model


def test_mixed_dtypes_load_as_saved(tmp_path):
    model = _half_with_float_bn(_model())
    for m in model.modules():  # values float16 cannot hold, so a float16 round trip shows
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.running_var.fill_(1 + 1e-4)
    want = {k: v.clone() for k, v in model.state_dict().items()}
    path = manyseg.save(model, tmp_path / "m.safetensors", **KW)
    loaded = manyseg.load(path)
    assert loaded.saved_metadata["dtype"] == "float16"
    have = loaded.state_dict()
    assert {k: v.dtype for k, v in have.items()} == {k: v.dtype for k, v in want.items()}
    for k, v in want.items():
        assert torch.equal(have[k], v), k
    assert any(v.dtype == torch.float32 for v in have.values())


def test_tied_parameters_stay_tied_after_dtype_cast(tmp_path):
    from manyseg.io import _match_dtypes

    lin = torch.nn.Linear(2, 2)
    model = torch.nn.Sequential(lin, lin).half()
    _match_dtypes(model, {"0.weight": torch.zeros(2, 2), "0.bias": torch.zeros(2),
                          "1.weight": torch.zeros(2, 2), "1.bias": torch.zeros(2)})
    assert model[0].weight is model[1].weight and model[0].weight.dtype == torch.float32


# --- X12: extra must round-trip through JSON -------------------------------------------------

@pytest.mark.parametrize("extra", [{1: "a"}, {"t": (1, 2)}, {"n": float("nan")},
                                   {"nested": {"k": [(1,)]}}, {"o": object()}])
def test_extra_that_json_would_change_is_refused(tmp_path, extra):
    with pytest.raises(ValueError, match="extra"):
        manyseg.save(_model(), tmp_path / "m.safetensors", extra=extra, **KW)


def test_plain_extra_round_trips(tmp_path):
    extra = {"epochs": 10, "lr": 1e-3, "tags": ["a", "b"], "nested": {"ok": True, "none": None}}
    path = manyseg.save(_model(), tmp_path / "m.safetensors", extra=extra, **KW)
    assert manyseg.read_metadata(path)["extra"] == extra


# --- B6: downsample_mean's valid mask ------------------------------------------------------

def test_downsample_mean_bhw_mask_is_per_pixel_not_per_channel():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    valid = torch.zeros(2, 4, 4, dtype=torch.bool)
    valid[:, :2, :2] = True
    out = downsample_mean(x, 0.5, valid)
    assert out.shape == (2, 3, 2, 2)
    torch.testing.assert_close(out, downsample_mean(x, 0.5, valid[:, None]))
    torch.testing.assert_close(out[:, :, 0, 0], x[:, :, :2, :2].mean((-1, -2)))
    assert (out[:, :, 1:, :] == 0).all() and (out[:, :, :, 1:] == 0).all()


def test_downsample_mean_accepts_full_shape_mask_and_rejects_others():
    x = torch.rand(2, 3, 4, 4)
    full = torch.rand(2, 3, 4, 4) > 0.5
    downsample_mean(x, 0.5, full)
    for bad in (torch.ones(3, 4, 4, dtype=torch.bool), torch.ones(4, 4, dtype=torch.bool),
                torch.ones(2, 2, 4, 4, dtype=torch.bool)):
        with pytest.raises(ValueError, match="valid must be"):
            downsample_mean(x, 0.5, bad)


# --- C6 / X7: what the losses' docstrings now say ------------------------------------------

def test_far_term_gradient_vanishes_for_confident_false_positive():
    d = torch.full((1, 1, 1), 32.0)
    y = torch.zeros(1, 1, 1)

    def grad(far, logit):
        x = torch.full((1, 1, 1), logit, requires_grad=True)
        distance_weighted_bce(x, y, d, far_weight=far).backward()
        return x.grad.item()

    assert grad(1.0, 10.0) == pytest.approx(grad(0.0, 10.0), rel=1e-3)  # far term ~0
    assert grad(1.0, 0.0) == pytest.approx(grad(0.0, 0.0) + 0.25)       # p(1-p) at p=0.5


def test_soft_dice_pools_channels_into_one_score_per_sample():
    logits = torch.tensor([[[[9.0]], [[-9.0]]]])  # channel 0 right, channel 1 misses its pixel
    target = torch.tensor([[[[1.0]], [[1.0]]]])
    pooled = soft_dice(logits, target, eps=0.0)
    per_channel = (soft_dice(logits[:, :1], target[:, :1], eps=0.0)
                   + soft_dice(logits[:, 1:], target[:, 1:], eps=0.0)) / 2
    assert pooled.item() == pytest.approx(1 / 3, abs=1e-3)
    assert per_channel.item() == pytest.approx(0.5, abs=1e-3)
    assert multiclass_dice(torch.randn(1, 2, 4, 4), torch.zeros(1, 4, 4).long()).ndim == 0


# --- B5 / X8: quantised programs (need torchao) -------------------------------------------

@pytest.fixture(scope="module")
def fixed_hw_dynamic_batch():
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = quantize_pt2e(model, torch.randn(2, 3, 64, 64),
                          [torch.randn(3, 3, 64, 64), torch.randn(1, 3, 64, 64)],
                          dynamic=False, dynamic_batch=True, reduce_range=True)
    return model, q


def test_dynamic_batch_at_fixed_hw(fixed_hw_dynamic_batch):
    _, q = fixed_hw_dynamic_batch
    for b in (1, 2, 5):
        assert q.module()(torch.randn(b, 3, 64, 64)).shape == (b, 3, 64, 64)


def test_fixed_batch_calibration_mismatch_is_a_clear_error():
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    with pytest.raises(ValueError, match="dynamic_batch=True"):
        quantize_pt2e(_model(), torch.randn(2, 3, 64, 64), [torch.randn(3, 3, 64, 64)],
                      dynamic=False, reduce_range=True)


def test_dynamic_batch_fixed_hw_saves_and_loads(fixed_hw_dynamic_batch, tmp_path):
    model, q = fixed_hw_dynamic_batch
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.pt2", source=model, **KW))
    assert loaded.batch_size is None and loaded.fixed_size == (64, 64)
    assert loaded(torch.randn(3, 3, 64, 64)).shape == (3, 3, 64, 64)
    with pytest.raises(ValueError, match="fixed input shape"):
        loaded(torch.randn(3, 3, 64, 96))


@pytest.mark.parametrize("unwrap", [True, False])
def test_loaded_program_saves_again(fixed_hw_dynamic_batch, tmp_path, unwrap):
    model, q = fixed_hw_dynamic_batch
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.pt2", source=model, **KW))
    again = manyseg.save(loaded.program if unwrap else loaded, tmp_path / "q2.pt2",
                         source=model, **KW)
    reloaded = manyseg.load(again)
    assert reloaded.input_shapes == loaded.input_shapes
    x = torch.randn(2, 3, 64, 64)
    torch.testing.assert_close(reloaded(x), loaded(x))


def test_loaded_dynamic_hw_program_saves_again(tmp_path):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), [torch.randn(1, 3, 64, 96)],
                          reduce_range=True)
    loaded = manyseg.load(manyseg.save(q, tmp_path / "q.pt2", source=model, **KW))
    reloaded = manyseg.load(manyseg.save(loaded, tmp_path / "q2.pt2", source=model, **KW))
    assert reloaded.input_shapes == {"x": [1, 3, None, None]}


def test_scene_step_after_unscoped_percentile_is_refused():
    """An unscoped percentile step is per tile (as in ManyPred), so a scene step can't follow."""
    from manyseg._normalize_spec import check_steps

    with pytest.raises(ValueError, match="per-tile step"):
        check_steps([{"type": "percentile"}, {"type": "zscore", "scope": "scene"}])
    check_steps([{"type": "zscore", "scope": "scene"}, {"type": "percentile"}])
