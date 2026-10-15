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
    model = create_model(inputs={"a": {"channels": 4}, "b": {"channels": 2}},
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


@pytest.mark.parametrize("name", ["fast", "compact", "balanced", "accurate"])
def test_presets_export_dynamic_hw(name):
    """The E018 presets (PVTv2-b0 encoder) are meant for deployment: dynamic H/W export."""
    from manyseg.presets import preset

    model = create_model(**preset(name), pretrained=False, in_channels=4, num_classes=2).eval()
    ep = export_model(model, torch.randn(1, 4, 64, 96))
    for hw in [(64, 96), (97, 131)]:
        x = torch.randn(1, 4, *hw)
        torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)
