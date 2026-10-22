"""E027 performance changes: bilinear resizes keep the activation dtype under autocast."""

from __future__ import annotations

import pytest
import torch
from manyseg import create_model
from torch import nn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FUSION = {"a": {"in_channels": 4, "backbone": "resnet18"},
          "b": {"in_channels": 2, "backbone": "resnet18", "alignment": "same_extent"}}


def _bf16_everywhere(m, x, where):
    seen = []
    for mod in where.modules():
        if isinstance(mod, nn.Conv2d | nn.ReLU):
            mod.register_forward_hook(lambda _m, _i, o: seen.append(o.dtype) and None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    return seen and all(d == torch.bfloat16 for d in seen), set(seen)


@cuda
@pytest.mark.parametrize("ftype", ["late", "intermediate"])
def test_concat_fusion_of_a_coarser_input_stays_bf16(ftype):
    m = create_model(pretrained=False, num_classes=2, decoder="unet", inputs=FUSION,
                     fusion={"type": ftype, "op": "concat"}).cuda()
    x = {"a": torch.randn(2, 4, 64, 64, device="cuda"),
         "b": torch.randn(2, 2, 32, 32, device="cuda")}
    ok, seen = _bf16_everywhere(m, x, m)
    assert ok, seen


@cuda
@pytest.mark.parametrize("upsample", ["bilinear", "aligned"])
def test_unet_bilinear_upsampling_stays_bf16(upsample):
    m = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2,
                     decoder="unet", decoder_kwargs={"upsample": upsample}).cuda()
    ok, seen = _bf16_everywhere(m, torch.randn(2, 3, 64, 64, device="cuda"), m.decoder)
    assert ok, seen


@pytest.mark.parametrize("upsample", ["bilinear", "aligned"])
def test_unet_bilinear_upsampling_values_unchanged_in_float64(upsample):
    """The cast is a no-op outside autocast: float64 in, float64 out, same values as the
    plain interpolation."""
    import torch.nn.functional as F
    from manyseg.decoders.unet import upsample2x

    x = torch.randn(1, 3, 5, 7, dtype=torch.float64)
    y = upsample2x(x, (10, 13), upsample)
    assert y.dtype == torch.float64
    if upsample == "bilinear":
        torch.testing.assert_close(
            y, F.interpolate(x, size=(10, 13), mode="bilinear", align_corners=False))
