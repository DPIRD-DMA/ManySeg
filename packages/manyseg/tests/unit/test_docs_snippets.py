"""Code shown in docs/api_design.md that is not covered elsewhere runs as written."""

import torch


def test_lower_level_composition():
    from manyseg.backbones import create_backbone
    from manyseg.decoders import UNetDecoder
    from manyseg.heads import PixelHead
    from manyseg.models import DenseModel

    bb = create_backbone("resnet34", pretrained=False, in_channels=4)
    dec = UNetDecoder(bb.feature_specs)
    model = DenseModel(bb, dec, {"mask": PixelHead(dec.out_spec, 3)}, single_output="mask")
    assert model.eval()(torch.randn(1, 4, 97, 131)).shape == (1, 3, 97, 131)
