"""Cross-library parity of ManySeg reference decoders (exact parameter counts and FLOPs).

These guard the claim that ManySeg's ``unet`` reproduces SMP's Unet and ``fastai_unet``
reproduces fastai's DynamicUnet in size and cost, so accuracy differences come from
implementation details rather than capacity.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode

from manyseg import create_model
from manyseg_bench.providers import ModelSpec, get_provider


def n(m):
    return sum(p.numel() for p in m.parameters())


def flops(m, x):
    with torch.no_grad(), FlopCounterMode(display=False) as f:
        m.eval()(x)
    return f.get_total_flops()


def test_unet_matches_smp_unet():
    ms = create_model(backbone="resnet34", pretrained=False, num_classes=3)
    smp = get_provider("smp").build(ModelSpec("unet", "resnet34", weights=None,
                                              num_classes=3)).model
    x = torch.randn(1, 3, 128, 128)
    assert n(ms) == n(smp)
    assert flops(ms, x) == flops(smp, x)


def test_fastai_unet_matches_fastai_dynamic_unet():
    ms = create_model(backbone="resnet34", pretrained=False, decoder="fastai_unet",
                      heads={"logits": {"type": "segmentation", "channels": 3,
                                        "kernel_size": 1}})
    fa = get_provider("fastai").build(ModelSpec("unet", "resnet34", weights=None,
                                                num_classes=3), img_size=(128, 128)).model
    x = torch.randn(1, 3, 128, 128)
    assert n(ms) == n(fa)
    assert flops(ms, x) == flops(fa, x)
