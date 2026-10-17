"""Heads may own a task-specific decoder instead of sharing the model's decoder."""

import pytest
import torch
from manyseg import create_model


def build(**heads):
    return create_model(backbone="resnet18", pretrained=False, heads=heads)


def test_per_head_decoder_shapes():
    model = build(
        seg={"type": "segmentation", "out_channels": 3},
        depth={"type": "regression", "out_channels": 1, "decoder": "fpn"},
        edges={"type": "segmentation", "out_channels": 1,
               "decoder": {"type": "unet", "upsample": "aligned"}},
    ).eval()
    out = model(torch.randn(2, 3, 65, 97))
    assert out["seg"].shape == (2, 3, 65, 97)
    assert out["depth"].shape == (2, 1, 65, 97)
    assert out["edges"].shape == (2, 1, 65, 97)
    assert set(model.task_decoders) == {"depth", "edges"}


def test_shared_decoder_skipped_when_unused():
    model = build(a={"type": "segmentation", "out_channels": 2, "decoder": "fpn"},
                  b={"type": "regression", "out_channels": 1, "decoder": "segformer"})
    assert model.decoder is None
    assert model(torch.randn(1, 3, 64, 64))["a"].shape == (1, 2, 64, 64)


def test_task_decoder_receives_gradient_only_from_its_head():
    model = build(seg={"type": "segmentation", "out_channels": 2},
                  depth={"type": "regression", "out_channels": 1, "decoder": "fpn"})
    model(torch.randn(2, 3, 64, 64))["seg"].mean().backward()
    assert all(p.grad is None for p in model.task_decoders["depth"].parameters())
    assert all(p.grad is not None for p in model.decoder.parameters())


def test_image_head_with_decoder_rejected():
    with pytest.raises(ValueError, match="pixel"):
        build(area={"type": "regression", "out_channels": 1, "scope": "image", "decoder": "fpn"})


@pytest.mark.slow
def test_task_decoders_export():
    from manyseg.export import export_model

    model = build(seg={"type": "segmentation", "out_channels": 2},
                  depth={"type": "regression", "out_channels": 1, "decoder": "fpn"}).eval()
    ep = export_model(model, torch.randn(1, 3, 64, 96))
    x = torch.randn(1, 3, 97, 131)
    for k, v in ep.module()(x).items():
        torch.testing.assert_close(v, model(x)[k], atol=1e-4, rtol=1e-4)
