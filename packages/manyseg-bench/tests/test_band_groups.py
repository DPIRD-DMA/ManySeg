"""Native-resolution Sentinel-2 band groups (E015): the provider splits a stacked 13-band 10 m
tensor into 10/20/60 m inputs, block-averages the coarse groups, and feeds a multi-input model
with same_extent alignment."""

import torch
from manyseg_bench.providers import ModelSpec, get_provider

S2_GROUPS = {
    "s2_10m": {"bands": [1, 2, 3, 7], "factor": 1},
    "s2_20m": {"bands": [4, 5, 6, 8, 11, 12], "factor": 2},
    "s2_60m": {"bands": [0, 9, 10], "factor": 6},
}


def _build(**extra):
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=2, in_channels=13,
                     options={"band_groups": S2_GROUPS, **extra})
    return get_provider("manyseg").build(spec, img_size=(256, 256))


def test_band_groups_split_downsample_and_shapes():
    b = _build()
    m = b.model.eval()
    seen = {}
    inner = m.model
    hook = inner.backbone.register_forward_pre_hook(
        lambda mod, args: seen.update({k: tuple(v.shape) for k, v in args[0].items()}))
    out = m(torch.randn(2, 13, 256, 256))
    hook.remove()
    assert out.shape == (2, 2, 256, 256)
    assert seen == {"s2_10m": (2, 4, 256, 256), "s2_20m": (2, 6, 128, 128),
                    "s2_60m": (2, 3, 43, 43)}  # ceil(256 / 6)
    assert {n: a for n, a in inner.backbone.alignment.items()} == {
        "s2_10m": "same_grid", "s2_20m": "same_extent", "s2_60m": "same_extent"}
    assert len({id(e) for e in inner.backbone.encoders.values()}) == 3  # independent encoders
    assert b.encoder_params and b.decoder_params


def test_band_groups_block_average_is_exact():
    from manyseg_bench.providers.manyseg_provider import BandGroupModel

    x = torch.arange(13 * 12 * 12, dtype=torch.float32).reshape(1, 13, 12, 12)
    split = BandGroupModel.split(x, S2_GROUPS)
    assert torch.equal(split["s2_10m"], x[:, [1, 2, 3, 7]])
    assert torch.allclose(split["s2_20m"][0, 0, 0, 0], x[0, 4, :2, :2].mean())
    assert torch.allclose(split["s2_60m"][0, 1, 1, 1], x[0, 9, 6:12, 6:12].mean())


def test_band_groups_shared_encoder():
    b = _build(band_group_inputs={"s2_20m": {"backbone": "shared:s2_10m"},
                                  "s2_60m": {"backbone": "shared:s2_10m"}})
    encs = list(b.model.model.backbone.encoders.values())
    # Shared weights (the same Parameter objects), BatchNorm per input since 0a4025d.
    assert all(e.stem_conv().weight is encs[0].stem_conv().weight for e in encs)
    assert len({id(e) for e in encs}) == 3
