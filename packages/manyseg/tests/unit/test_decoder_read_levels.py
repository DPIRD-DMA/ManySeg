"""Decoders that declare ``read_levels`` skip the gradient probe at build time, so what they declare
has to be what they read: a wrong answer would prune a level the decoder needs."""

import pytest
from manyseg import create_model
from manyseg.conditioning.inject import probe_levels

CONFIGS = [
    ("unet", {}),
    ("unet", {"out_stride": 2}),
    ("unet", {"out_stride": 4}),
    ("unet", {"channels": [64, 32, 16, 8, 8]}),
    ("unet", {"input_cross": True}),
    ("unetplusplus", {}),
    ("unetplusplus", {"out_stride": 2}),
    ("fastai_unet", {}),
    ("fastai_unet", {"out_stride": 2}),
    ("fastai_unet", {"out_stride": 4}),
    ("deeplabv3plus", {}),
]


@pytest.mark.parametrize("backbone", ["resnet18", "pvt_v2_b0.in1k", "vit_tiny_patch16_224"])
@pytest.mark.parametrize(("decoder", "kwargs"), CONFIGS,
                         ids=[f"{d}-{sorted(k.items())}" for d, k in CONFIGS])
def test_declared_read_levels_match_the_probe(backbone, decoder, kwargs):
    try:
        model = create_model(backbone, pretrained=False, num_classes=2, decoder=decoder,
                             decoder_kwargs=kwargs)
    except ValueError as e:  # a combination the encoder's strides don't allow
        pytest.skip(str(e))
    dec = model.decoder
    assert sorted(dec.read_levels) == probe_levels(dec, tuple(model.feature_specs), 3)
