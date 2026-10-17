"""Time-series inputs (``"kind": "timeseries"``): one 2D encoder applied to every date, then a
temporal aggregation per feature level (masked mean / max / lightweight temporal attention)."""

import pytest
import torch
from manyseg import create_model

AGGS = ["mean", "max", "attention"]


def build(temporal="mean", heads=None, extra=None, **kw):
    inputs = {"s2": {"kind": "timeseries", "in_channels": 4, "temporal": temporal}, **(extra or {})}
    if heads is None:
        kw.setdefault("num_classes", 3)
    return create_model(backbone="resnet18", pretrained=False, inputs=inputs, heads=heads,
                        **kw).eval()


def perturb(model, scale=0.5):
    """Give the zero-initialised attention queries non-zero values."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "temporal" in n and n.endswith("query"):
                p.normal_(0, scale)
    return model


def series(b=2, t=5, c=4, h=64, w=64, seed=0):
    return torch.randn(b, t, c, h, w, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("agg", AGGS)
def test_any_number_of_dates_and_odd_sizes(agg):
    model = build(agg)
    for t, (h, w) in ((1, (64, 64)), (3, (33, 47)), (7, (64, 96))):
        assert model({"s2": series(t=t, h=h, w=w)}).shape == (2, 3, h, w)


@pytest.mark.parametrize("agg", AGGS)
def test_repeated_frame_equals_single_frame(agg):
    model = perturb(build(agg))
    x = series(t=1)
    torch.testing.assert_close(model({"s2": x.repeat(1, 4, 1, 1, 1)}), model({"s2": x}),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("agg", AGGS)
def test_masked_dates_have_no_effect(agg):
    model = perturb(build(agg))
    x = series(t=5)
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 0, 0, 1]], dtype=torch.bool)
    y = x.clone()
    y[~mask] = torch.randn_like(y[~mask]) * 10  # garbage in the masked dates
    a = model({"s2": x, "s2_mask": mask})
    b = model({"s2": y, "s2_mask": mask})
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("agg", AGGS)
def test_fully_masked_sample_is_finite(agg):
    model = perturb(build(agg))
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    assert torch.isfinite(model({"s2": series(t=3), "s2_mask": mask})).all()


@pytest.mark.parametrize("agg", ["mean", "max"])
def test_order_invariant_aggregations(agg):
    model = build(agg)
    x = series(t=4)
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(model({"s2": x}), model({"s2": x[:, perm]}), rtol=1e-5,
                               atol=1e-5)


def test_attention_follows_dates_not_order():
    model = perturb(build("attention"))
    x = series(t=4)
    dates = torch.tensor([[10.0, 80.0, 150.0, 220.0]] * 2)
    perm = torch.tensor([2, 0, 3, 1])
    a = model({"s2": x, "s2_dates": dates})
    b = model({"s2": x[:, perm], "s2_dates": dates[:, perm]})
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    c = model({"s2": x, "s2_dates": dates + torch.tensor([0.0, 100.0, -50.0, 30.0])})
    assert not torch.allclose(a, c, atol=1e-4), "dates have no effect"


def test_attention_is_the_masked_mean_at_init():
    torch.manual_seed(0)
    att = build("attention")
    torch.manual_seed(0)
    mean = build("mean")
    sd = {k: v for k, v in att.state_dict().items() if "temporal" not in k}
    mean.load_state_dict(sd)
    x, mask = series(t=4), torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.bool)
    dates = torch.tensor([[1.0, 50.0, 90.0, 200.0]] * 2)
    torch.testing.assert_close(att({"s2": x, "s2_mask": mask, "s2_dates": dates}),
                               mean({"s2": x, "s2_mask": mask}), rtol=1e-5, atol=1e-5)


def test_attention_learns():
    model = build("attention").train()
    out = model({"s2": series(t=3), "s2_dates": torch.tensor([[1.0, 2.0, 3.0]] * 2)})
    out.square().mean().backward()
    q = [p for n, p in model.named_parameters() if n.endswith("query")]
    assert q and all(p.grad is not None and p.grad.abs().sum() > 0 for p in q)


def test_temporal_options_dict():
    model = build({"type": "attention", "heads": 2, "key_dim": 8})
    assert model({"s2": series(t=2)}).shape == (2, 3, 64, 64)


def test_composes_with_rasters_metadata_and_mixed_heads():
    heads = {"crop": {"type": "segmentation", "out_channels": 5},
             "yield_": {"type": "regression", "out_channels": 1, "scope": "image"}}
    extra = {"dem": {"in_channels": 1, "alignment": "same_extent"},
             "s1": {"kind": "timeseries", "in_channels": 2, "temporal": "max",
                    "backbone": "resnet18"},
             "meta": {"kind": "vector", "fields": {"latlon": {"type": "coords"}}}}
    model = build("attention", heads=heads, extra=extra)
    x = {"s2": series(t=5), "s2_dates": torch.rand(2, 5) * 365,
         "s1": series(t=3, c=2), "s1_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]).bool(),
         "dem": torch.randn(2, 1, 32, 32), "meta": torch.tensor([[-31.9, 115.8], [45.0, 2.0]])}
    out = model(x)
    assert out["crop"].shape == (2, 5, 64, 64) and out["yield_"].shape == (2, 1)
    model.select_heads(["yield_"])
    assert model.decoder is None and model(x)["yield_"].shape == (2, 1)


def test_describe_mentions_the_temporal_aggregation():
    text = build("attention").describe()
    assert "attention" in text and "s2" in text


def test_errors():
    with pytest.raises(ValueError, match="early"):
        build(fusion={"type": "early"})
    with pytest.raises(ValueError, match="temporal"):
        build("median")
    with pytest.raises(ValueError, match="temporal"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={"a": {"in_channels": 3, "temporal": "mean"}})
    with pytest.raises(ValueError, match="shared"):
        build(extra={"b": {"kind": "timeseries", "in_channels": 4, "backbone": "shared:s2"}})
    model = build()
    with pytest.raises(ValueError, match=r"\(B, T, C, H, W\)"):
        model({"s2": torch.randn(2, 4, 64, 64)})


@pytest.mark.parametrize("agg", AGGS)
def test_masked_dates_have_no_effect_in_training_mode(agg):
    """BatchNorm batch statistics must not see masked dates (Codex review)."""
    model = perturb(build(agg)).train()
    x = series(t=5)
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 0, 0, 1]], dtype=torch.bool)
    y = x.clone()
    y[~mask] = torch.randn_like(y[~mask]) * 10
    torch.manual_seed(0)
    a = model({"s2": x, "s2_mask": mask})
    torch.manual_seed(0)
    b = model({"s2": y, "s2_mask": mask})
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("agg", AGGS)
@pytest.mark.parametrize("train", [False, True])
def test_nan_in_masked_dates_does_not_leak(agg, train):
    model = perturb(build(agg)).train(train)
    x = series(t=4)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)
    x[~mask] = float("nan")
    assert torch.isfinite(model({"s2": x, "s2_mask": mask})).all()


def test_dates_on_another_device_and_bad_shapes():
    model = build("attention")
    x = series(t=3)
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        model({"s2": x, "s2_dates": torch.ones(2, 4)})
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        model({"s2": x, "s2_mask": torch.ones(2, 3, 1).bool()})
    if torch.cuda.is_available():
        m = build("attention").cuda()
        assert m({"s2": x.cuda(), "s2_dates": torch.ones(2, 3)}).is_cuda


def test_date_dim_must_be_even():
    with pytest.raises(ValueError, match="date_dim"):
        build({"type": "attention", "date_dim": 31})


def test_vector_input_cannot_take_a_series_extra_name():
    with pytest.raises(ValueError, match="reserved"):
        build(extra={"s2_dates": {"kind": "vector", "fields": {"d": {"type": "continuous"}}}})


@pytest.mark.parametrize("agg", AGGS)
def test_entirely_masked_training_batch(agg):
    model = perturb(build(agg)).train()
    out = model({"s2": series(t=3), "s2_mask": torch.zeros(2, 3, dtype=torch.bool)})
    assert out.shape == (2, 3, 64, 64) and torch.isfinite(out).all()


def test_series_extra_name_reserved_without_vectors():
    with pytest.raises(ValueError, match="reserved"):
        build(extra={"s2_dates": {"kind": "timeseries", "in_channels": 4}})
