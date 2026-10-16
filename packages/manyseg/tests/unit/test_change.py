"""Change detection: difference fusion ops, order-keeping temporal aggregation, and per-input
heads (a head reading one input's own features, e.g. land cover at each date for semantic
change detection), optionally sharing their decoder and head with another per-input head."""

import pytest
import torch
from manyseg import create_model

PAIR = {"t1": {"channels": 3}, "t2": {"channels": 3, "backbone": "shared:t1", "share_norm": True}}


def pair(seed=0, b=2, hw=(64, 64)):
    g = torch.Generator().manual_seed(seed)
    return {"t1": torch.randn(b, 3, *hw, generator=g), "t2": torch.randn(b, 3, *hw, generator=g)}


@pytest.mark.parametrize("op", ["diff", "concat_diff"])
def test_difference_fusion_ops(op):
    model = create_model(backbone="resnet18", pretrained=False, inputs=PAIR, num_classes=2,
                         fusion={"type": "intermediate", "op": op}).eval()
    x = pair()
    out = model(x)
    assert out.shape == (2, 2, 64, 64)
    swapped = model({"t1": x["t2"], "t2": x["t1"]})
    if op == "diff":  # |f1 - f2| is symmetric with a shared (Siamese) encoder
        torch.testing.assert_close(out, swapped, rtol=1e-5, atol=1e-5)
    else:
        assert not torch.allclose(out, swapped)


def test_diff_fusion_needs_two_inputs():
    inputs = {**PAIR, "t3": {"channels": 3, "backbone": "shared:t1", "share_norm": True}}
    with pytest.raises(ValueError, match="two inputs"):
        create_model(backbone="resnet18", pretrained=False, inputs=inputs, num_classes=2,
                     fusion={"type": "intermediate", "op": "diff"})


def series(t=2, seed=0):
    return torch.randn(2, t, 4, 64, 64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("temporal", [{"type": "concat", "dates": 3}, "diff"])
def test_order_keeping_temporal_aggregation(temporal):
    t = 3 if isinstance(temporal, dict) else 2
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "s2": {"kind": "timeseries", "channels": 4, "temporal": temporal}}).eval()
    x = series(t)
    out = model({"s2": x})
    assert out.shape == (2, 2, 64, 64)
    rev = model({"s2": x.flip(1)})
    if temporal == "diff":  # |f_last - f_first|
        torch.testing.assert_close(out, rev, rtol=1e-5, atol=1e-5)
    else:
        assert not torch.allclose(out, rev), "concat must depend on the date order"
    with pytest.raises(ValueError, match="dates"):
        model({"s2": series(t + 1)})


def scd_model(share=True, **kw):
    heads = {"change": {"type": "segmentation", "channels": 2},
             "lc1": {"type": "segmentation", "channels": 7, "input": "t1"},
             "lc2": {"type": "segmentation", "channels": 7, "input": "t2",
                     **({"share": "lc1"} if share else {})}}
    return create_model(backbone="resnet18", pretrained=False, inputs=PAIR, heads=heads,
                        fusion={"type": "intermediate", "op": "concat_diff"}, **kw).eval()


def test_per_input_heads_read_only_their_input():
    model = scd_model()
    a, b = pair(0), pair(1)
    out = model(a)
    assert out["lc1"].shape == (2, 7, 64, 64) and out["change"].shape == (2, 2, 64, 64)
    other = model({"t1": a["t1"], "t2": b["t2"]})  # new t2, same t1
    torch.testing.assert_close(other["lc1"], out["lc1"], rtol=0, atol=0)
    assert not torch.allclose(other["lc2"], out["lc2"])
    assert not torch.allclose(other["change"], out["change"])


def test_shared_per_input_heads_are_one_module():
    model = scd_model()
    assert model.task_decoders["lc2"] is model.task_decoders["lc1"]
    assert model.heads["lc2"] is model.heads["lc1"]
    x = pair()
    out = model(x)
    swapped = model({"t1": x["t2"], "t2": x["t1"]})
    torch.testing.assert_close(out["lc2"], swapped["lc1"], rtol=1e-5, atol=1e-5)
    n_shared = sum(p.numel() for p in model.parameters())
    n_sep = sum(p.numel() for p in scd_model(share=False).parameters())
    assert n_shared < n_sep


def test_per_input_head_decoder_type_and_select_heads():
    heads = {"lc1": {"type": "segmentation", "channels": 3, "input": "t1", "decoder": "fpn"},
             "change": {"type": "segmentation", "channels": 2}}
    model = create_model(backbone="resnet18", pretrained=False, inputs=PAIR, heads=heads).eval()
    assert type(model.task_decoders["lc1"]).__name__.lower().startswith("fpn")
    model.select_heads(["lc1"])
    assert model.decoder is None and model(pair())["lc1"].shape == (2, 3, 64, 64)


def test_per_input_heads_describe():
    text = scd_model().describe()
    assert "lc1" in text and "t1" in text


def test_per_input_head_errors():
    with pytest.raises(ValueError, match="input"):
        create_model(backbone="resnet18", pretrained=False, num_classes=None,
                     heads={"lc": {"channels": 3, "input": "t1"}})
    with pytest.raises(ValueError, match="nope"):
        create_model(backbone="resnet18", pretrained=False, inputs=PAIR,
                     heads={"lc": {"channels": 3, "input": "nope"}})
    with pytest.raises(ValueError, match="share"):
        create_model(backbone="resnet18", pretrained=False, inputs=PAIR, heads={
            "a": {"channels": 3, "input": "t1"}, "b": {"channels": 4, "input": "t2",
                                                      "share": "a"}})
    with pytest.raises(ValueError, match="late"):
        create_model(backbone="resnet18", pretrained=False, inputs=PAIR,
                     fusion={"type": "late"}, heads={"a": {"channels": 3, "input": "t1"}})


def test_per_input_heads_with_metadata_film():
    inputs = {**PAIR, "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic",
                                                                     "period": 365.25}}}}
    heads = {"change": {"type": "segmentation", "channels": 2},
             "lc1": {"type": "segmentation", "channels": 7, "input": "t1"},
             "lc2": {"type": "segmentation", "channels": 7, "input": "t2", "share": "lc1"}}
    model = create_model(backbone="resnet18", pretrained=False, inputs=inputs, heads=heads,
                         fusion={"type": "intermediate", "op": "concat_diff"}).eval()
    assert {"decoder", "task_lc1", "task_lc2"} <= set(model.conditioner.film)
    out = model({**pair(), "meta": torch.tensor([[10.0], [200.0]])})
    assert out["lc2"].shape == (2, 7, 64, 64)
