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


def test_a_subclass_that_overrides_forward_is_probed_not_trusted():
    """Its reads may differ from the parent's, whose read_levels describe the parent's forward."""
    from manyseg.conditioning.inject import used_levels
    from manyseg.decoders import UNetDecoder

    class LastLevelOnly(UNetDecoder):
        def forward(self, features):
            return super().forward([f.detach() * 0 for f in features[:-1]] + [features[-1]])

    model = create_model("resnet18", pretrained=False, num_classes=2)
    specs = tuple(model.feature_specs)
    dec = LastLevelOnly(specs)
    assert used_levels(dec, specs) == [len(specs) - 1]
    assert dec.read_levels != [len(specs) - 1]  # the inherited answer, which would be wrong


def test_a_subclass_that_only_adds_options_keeps_the_declaration():
    from manyseg.conditioning.inject import declared_levels
    from manyseg.decoders import UNetDecoder

    class Renamed(UNetDecoder):
        pass

    model = create_model("resnet18", pretrained=False, num_classes=2)
    assert declared_levels(Renamed(tuple(model.feature_specs))) is not None
