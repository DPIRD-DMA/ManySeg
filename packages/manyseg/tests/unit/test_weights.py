"""Remote-sensing pretrained weights (manyseg.weights): registry entries and loading into timm
architectures through create_model(pretrained="<tag>"). Download tests are network-marked."""

import pytest
import torch
from manyseg import create_model
from manyseg.weights import WEIGHTS, get_weights


def test_registry_entries_are_complete():
    assert len(WEIGHTS) >= 8
    for key, w in WEIGHTS.items():
        assert key == f"{w.model}.{w.tag}"
        assert len(w.bands) == w.in_channels == len(w.mean) == len(w.std), key
        assert w.url.startswith("https://") and w.license and w.source, key
        assert w.citation and "20" in w.citation, f"{key}: CC BY / ODC-BY need attribution"
        assert "-" in w.url.rsplit("/", 1)[-1], f"{key}: torch.hub needs a hash suffix"


def test_lookup_and_errors():
    w = get_weights("resnet50", "ssl4eo_s2_moco")
    assert w.in_channels == 13 and w.bands[0] == "B01"
    with pytest.raises(ValueError, match="ssl4eo_s2_moco"):
        get_weights("resnet50", "nope")
    with pytest.raises(ValueError, match="resnet50"):
        get_weights("convnext_tiny", "ssl4eo_s2_moco")


def fake_hub(monkeypatch, sd):
    import manyseg.weights as mw

    monkeypatch.setattr(mw, "_download", lambda url: {k: v.clone() for k, v in sd.items()})


def test_loading_from_a_state_dict(monkeypatch):
    import timm

    src = timm.create_model("resnet18", in_chans=13, num_classes=10)
    sd = src.state_dict()
    fake_hub(monkeypatch, sd)
    m = create_model(backbone="resnet18", pretrained="ssl4eo_s2_moco", in_channels=13,
                     num_classes=2)
    stem = m.backbone.stem_conv()
    torch.testing.assert_close(stem.weight, sd["conv1.weight"])
    assert m.backbone.weights == "resnet18.ssl4eo_s2_moco"
    assert "ssl4eo_s2_moco" in m.describe()


def test_loading_rejects_band_count_mismatch_and_missing_keys(monkeypatch):
    import timm

    fake_hub(monkeypatch, timm.create_model("resnet18", in_chans=13).state_dict())
    with pytest.raises(ValueError, match="13 bands"):
        create_model(backbone="resnet18", pretrained="ssl4eo_s2_moco", in_channels=4,
                     num_classes=2)
    sd = timm.create_model("resnet18", in_chans=13).state_dict()
    del sd["layer4.1.conv2.weight"]
    fake_hub(monkeypatch, sd)
    with pytest.raises(RuntimeError, match="layer4.1.conv2.weight"):
        create_model(backbone="resnet18", pretrained="ssl4eo_s2_moco", in_channels=13,
                     num_classes=2)


def test_pretrained_tag_on_multi_input_encoder(monkeypatch):
    import timm

    fake_hub(monkeypatch, timm.create_model("resnet18", in_chans=13).state_dict())
    m = create_model(backbone="resnet18", num_classes=2, inputs={
        "s2": {"in_channels": 13, "pretrained": "ssl4eo_s2_moco"},
        "dem": {"in_channels": 1, "pretrained": False}})
    assert m.backbone.encoders["s2"].weights == "resnet18.ssl4eo_s2_moco"


@pytest.mark.network
@pytest.mark.parametrize("key", ["resnet50.ssl4eo_s2_moco", "resnet50.satlas_s2_si_ms",
                                 "vit_small_patch16_224.ssl4eo_s2_dino"])
def test_real_weights_load(key):
    w = WEIGHTS[key]
    m = create_model(backbone=w.model, pretrained=w.tag, in_channels=w.in_channels,
                     num_classes=2).eval()
    assert m(torch.randn(1, w.in_channels, 64, 64)).shape == (1, 2, 64, 64)


def test_vit_weights_resample_pos_embed_for_another_img_size(monkeypatch):
    import timm

    src = timm.create_model("vit_small_patch16_224", in_chans=13)
    fake_hub(monkeypatch, src.state_dict())
    m = create_model(backbone="vit_small_patch16_224", pretrained="ssl4eo_s2_moco",
                     in_channels=13, num_classes=2, backbone_kwargs={"img_size": 320})
    assert m.eval()(torch.randn(1, 13, 320, 320)).shape == (1, 2, 320, 320)
