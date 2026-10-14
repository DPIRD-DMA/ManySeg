"""Cross-library parity of ManySeg reference decoders (exact parameter counts and FLOPs).

These guard the claim that ManySeg's ``unet`` reproduces SMP's Unet and ``fastai_unet``
reproduces fastai's DynamicUnet in size and cost, so accuracy differences come from
implementation details rather than capacity.
"""

import pytest
import torch
from manyseg import create_model
from manyseg_bench.providers import ModelSpec, get_provider
from torch.utils.flop_counter import FlopCounterMode


def n(m):
    return sum(p.numel() for p in m.parameters())


def flops(m, x):
    with torch.no_grad(), FlopCounterMode(display=False) as f:
        m.eval()(x)
    return f.get_total_flops()


# out_stride=1 is the default; passing it explicitly must not change anything.
@pytest.mark.parametrize("dkw", [{}, {"out_stride": 1}], ids=["default", "out_stride1"])
def test_unet_matches_smp_unet(dkw):
    ms = create_model(backbone="resnet34", pretrained=False, num_classes=3, decoder_kwargs=dkw)
    smp = get_provider("smp").build(ModelSpec("unet", "resnet34", weights=None,
                                              num_classes=3)).model
    x = torch.randn(1, 3, 128, 128)
    assert n(ms) == n(smp)
    assert flops(ms, x) == flops(smp, x)


@pytest.mark.parametrize("dkw", [{}, {"out_stride": 1}], ids=["default", "out_stride1"])
def test_fastai_unet_matches_fastai_dynamic_unet(dkw):
    ms = create_model(backbone="resnet34", pretrained=False, decoder="fastai_unet",
                      decoder_kwargs=dkw,
                      heads={"logits": {"type": "segmentation", "channels": 3,
                                        "kernel_size": 1}})
    fa = get_provider("fastai").build(ModelSpec("unet", "resnet34", weights=None,
                                                num_classes=3), img_size=(128, 128)).model
    x = torch.randn(1, 3, 128, 128)
    assert n(ms) == n(fa)
    assert flops(ms, x) == flops(fa, x)


@pytest.mark.parametrize("decoder", ["unet", "fastai_unet"])
def test_reduced_out_stride_is_cheaper_than_the_reference(decoder):
    """Reduced strides only remove decoder stages: fewer parameters and FLOPs than parity."""
    x = torch.randn(1, 3, 128, 128)
    ref = create_model(backbone="resnet34", pretrained=False, num_classes=3, decoder=decoder)
    costs = [(n(ref), flops(ref, x))]
    for s in (2, 4):
        m = create_model(backbone="resnet34", pretrained=False, num_classes=3, decoder=decoder,
                         decoder_kwargs={"out_stride": s})
        costs.append((n(m), flops(m, x)))
    assert costs == sorted(costs, reverse=True) and len(set(costs)) == 3
