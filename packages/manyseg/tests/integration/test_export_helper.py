"""manyseg.export: one call to a dynamic-shape ExportedProgram for any ManySeg model."""

import pytest
import torch
from manyseg import create_model
from manyseg.export import export_model

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_atto", "caformer_s18"])
def test_export_model_dynamic_hw(backbone):
    """convnext/caformer hit an unprovable export guard without deferred runtime asserts."""
    model = create_model(backbone=backbone, pretrained=False, num_classes=2).eval()
    ep = export_model(model, torch.randn(1, 3, 64, 96))
    for hw in [(64, 96), (97, 131), (200, 65)]:
        x = torch.randn(1, 3, *hw)
        torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)


def test_export_model_dynamic_batch():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    ep = export_model(model, torch.randn(2, 3, 64, 64), dynamic_batch=True)
    for n in (1, 3):
        x = torch.randn(n, 3, 64, 96)
        torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)


def test_export_model_dict_inputs():
    model = create_model(inputs={"a": {"in_channels": 4}, "b": {"in_channels": 2}},
                         backbone="resnet18", pretrained=False, num_classes=2).eval()
    ex = {"a": torch.randn(1, 4, 64, 96), "b": torch.randn(1, 2, 64, 96)}
    ep = export_model(model, ex)
    x = {"a": torch.randn(1, 4, 97, 131), "b": torch.randn(1, 2, 97, 131)}
    torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)


def test_export_rejects_too_small_example():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    with pytest.raises(ValueError, match="at least"):
        export_model(model, torch.randn(1, 3, 32, 32))


@pytest.mark.parametrize("decoders", [None, "separate"])
def test_export_instance_preset(decoders):
    """Instance heads export; post-processing stays outside the graph (documented)."""
    from manyseg.instances import instance_heads

    model = create_model(backbone="resnet18", pretrained=False,
                         heads=instance_heads(decoders=decoders)).eval()
    ep = export_model(model, torch.randn(1, 3, 64, 96))
    x = torch.randn(1, 3, 97, 131)
    ref = model(x)
    for k, v in ep.module()(x).items():
        torch.testing.assert_close(v, ref[k], atol=1e-4, rtol=1e-4)


def test_export_point_and_density_presets():
    from manyseg.points import density_head, point_heads

    heads = {**point_heads(offsets=True), **density_head()}
    model = create_model(backbone="resnet18", pretrained=False, heads=heads).eval()
    ep = export_model(model, torch.randn(1, 3, 64, 96))
    x = torch.randn(1, 3, 97, 131)
    ref = model(x)
    for k, v in ep.module()(x).items():
        torch.testing.assert_close(v, ref[k], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("config", [
    {"decoder": "segformer"},
    {"decoder": "unet", "decoder_kwargs": {"channels": [64, 32, 16, 8, 8]}},
    {"decoder": "unet", "decoder_kwargs": {"channels": [128, 64, 32, 16, 8]}},
    {"decoder": "unet"},
], ids=["segformer", "unet-quarter", "unet-half", "unet-full"])
def test_pvt_decoders_export_dynamic_hw(config):
    """PVTv2-b0 is a small encoder people deploy, so these stay covered: dynamic H/W export."""
    model = create_model(backbone="pvt_v2_b0.in1k", **config, pretrained=False, in_channels=4,
                         num_classes=2).eval()
    ep = export_model(model, torch.randn(1, 4, 64, 96))
    for hw in [(64, 96), (97, 131)]:
        x = torch.randn(1, 4, *hw)
        torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("agg", ["mean", "attention"])
def test_timeseries_export_dynamic_dates_and_size(agg):
    """Time series export with dynamic H/W *and* number of dates, alongside a plain raster."""
    inputs = {"s2": {"kind": "timeseries", "in_channels": 4, "temporal": agg},
              "dem": {"in_channels": 1}}
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2,
                         inputs=inputs).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("query"):
                p.normal_(0, 0.5)

    def ex(t, h, w):
        return {"s2": torch.randn(1, t, 4, h, w), "s2_mask": torch.rand(1, t) > 0.2,
                "s2_dates": torch.rand(1, t) * 365, "dem": torch.randn(1, 1, h, w)}

    ep = export_model(model, ex(4, 64, 96))
    for t, h, w in ((4, 64, 96), (7, 97, 131), (2, 70, 65), (1, 64, 64)):
        x = ex(t, h, w)
        torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)


def test_semantic_change_model_exports():
    inputs = {"t1": {"in_channels": 3},
              "t2": {"in_channels": 3, "backbone": "shared:t1", "share_norm": True}}
    heads = {"change": {"type": "segmentation", "out_channels": 2},
             "lc1": {"type": "segmentation", "out_channels": 7, "input": "t1"},
             "lc2": {"type": "segmentation", "out_channels": 7, "input": "t2", "share": "lc1"}}
    model = create_model(backbone="resnet18", pretrained=False, inputs=inputs, heads=heads,
                         fusion={"type": "intermediate", "op": "concat_diff"}).eval()
    ex = {"t1": torch.randn(1, 3, 64, 96), "t2": torch.randn(1, 3, 64, 96)}
    ep = export_model(model, ex)
    x = {"t1": torch.randn(1, 3, 97, 131), "t2": torch.randn(1, 3, 97, 131)}
    got, ref = ep.module()(x), model(x)
    for k in heads:
        torch.testing.assert_close(got[k], ref[k], atol=1e-4, rtol=1e-4)
