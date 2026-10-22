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


def test_attribution_is_saved_restored_and_described(monkeypatch, tmp_path):
    import manyseg
    import timm
    from manyseg.weights import attribution

    fake_hub(monkeypatch, timm.create_model("resnet18", in_chans=13).state_dict())
    m = create_model(backbone="resnet18", num_classes=2, inputs={
        "s2": {"in_channels": 13, "pretrained": "ssl4eo_s2_moco"},
        "dem": {"in_channels": 1, "pretrained": False}})
    (rec,) = attribution(m)  # the randomly initialised dem encoder needs none
    assert rec["module"] == "backbone.encoders.s2" and rec["license"] == "CC BY 4.0"
    assert rec["citation"].startswith("Wang") and rec["source"].startswith("https://")
    path = manyseg.save(m, tmp_path / "m.safetensors", normalize=[],
                        activations={"logits": "softmax"})
    assert manyseg.read_metadata(path)["pretrained_weights"] == [rec]
    # load rebuilds without downloading: the record, not the backbones, says where it started
    loaded = manyseg.load(path)
    assert loaded.backbone.encoders["s2"].weights is None and attribution(loaded) == [rec]
    assert rec["provider"] == "manyseg"
    text = loaded.describe()
    assert "resnet18 (weights resnet18.ssl4eo_s2_moco)" in text
    assert "stem: 13 ch, unchanged" in text and "cite: Wang" in text
    again = manyseg.save(loaded, tmp_path / "again.safetensors", normalize=[],
                         activations={"logits": "softmax"})
    assert manyseg.read_metadata(again)["pretrained_weights"] == [rec]
    # a backbone swapped in after loading reports its own weights, not the restored record
    loaded.backbone.encoders["s2"] = create_model(
        backbone="resnet18", pretrained=False, in_channels=13, num_classes=2).backbone
    assert attribution(loaded) == []


def test_attribution_of_weight_sharing_copies(monkeypatch):
    import timm
    from manyseg.weights import attribution

    fake_hub(monkeypatch, timm.create_model("resnet18", in_chans=13).state_dict())
    m = create_model(backbone="resnet18", num_classes=2, inputs={
        "a": {"in_channels": 13, "pretrained": "ssl4eo_s2_moco"},
        "b": {"in_channels": 13, "backbone": "shared:a"}})
    # separate modules with shared weights: one entry each, so describe can label each encoder
    assert [(e["module"], e["weights"]) for e in attribution(m)] == [
        ("backbone.encoders.a", "resnet18.ssl4eo_s2_moco"),
        ("backbone.encoders.b", "resnet18.ssl4eo_s2_moco")]


def test_attribution_of_timm_and_random_weights():
    from manyseg.weights import attribution

    m = create_model(backbone="resnet34", pretrained=False, num_classes=2)
    assert attribution(m) == [] and "pretrained weights" not in m.describe()
    m.backbone.pretrained = True  # as if built with timm's ImageNet weights (no download)
    (rec,) = attribution(m)
    assert rec["weights"] == "resnet34.a1_in1k" and rec["provider"] == "timm"
    assert rec["license"] == "apache-2.0"
    assert rec["source"] == "https://huggingface.co/timm/resnet34.a1_in1k"


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
