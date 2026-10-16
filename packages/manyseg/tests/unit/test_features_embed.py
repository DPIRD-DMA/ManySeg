"""Feature / embedding output: model.features(x) (encoder pyramid on the input's valid grid),
model.embed(x, ...) (per-pixel or pooled vectors) and model.embedder(...) (an exportable
module)."""

import pytest
import torch
from manyseg import create_model


def model(**kw):
    kw.setdefault("num_classes", 3)
    return create_model(backbone="resnet18", pretrained=False, **kw).eval()


def test_features_pyramid_cropped_to_the_valid_region():
    m = model()
    x = torch.randn(1, 3, 97, 131)
    feats = m.features(x)
    assert [f.shape[1] for f in feats] == [s.channels for s in m.feature_specs]
    for f, s in zip(feats, m.feature_specs):
        assert f.shape[-2:] == (-(-97 // s.stride), -(-131 // s.stride))
    padded = m.backbone(torch.nn.functional.pad(x, (0, 160 - 131, 0, 128 - 97)))
    torch.testing.assert_close(feats[-1], padded[-1][..., :4, :5])


def test_embed_pooled_and_per_pixel():
    m = model()
    x = torch.randn(2, 3, 64, 96)
    deep = m.features(x)[-1]
    torch.testing.assert_close(m.embed(x, pool="mean"), deep.mean((-2, -1)))
    torch.testing.assert_close(m.embed(x, pool="max"), deep.amax((-2, -1)))
    assert m.embed(x).shape == deep.shape
    assert m.embed(x, level=0).shape == m.features(x)[0].shape
    dec = m.embed(x, source="decoder")
    assert dec.shape == (2, m.decoder.out_spec.channels, 64, 96)
    unit = m.embed(x, pool="mean", normalize=True)
    torch.testing.assert_close(unit.norm(dim=1), torch.ones(2))


def test_embed_with_multi_input_and_metadata():
    inputs = {"a": {"channels": 3}, "b": {"channels": 2},
              "meta": {"kind": "vector", "fields": {"d": {"type": "continuous"}}}}
    m = create_model(backbone="resnet18", pretrained=False, inputs=inputs, num_classes=2).eval()
    x = {"a": torch.randn(1, 3, 64, 64), "b": torch.randn(1, 2, 64, 64),
         "meta": torch.tensor([[0.5]])}
    assert m.embed(x, pool="mean").shape == (1, m.feature_specs[-1].channels)
    assert m.embed(x, source="decoder").shape[-2:] == (64, 64)


def test_embed_errors():
    m = model()
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="source"):
        m.embed(x, source="head")
    with pytest.raises(ValueError, match="pool"):
        m.embed(x, pool="median")
    image_only = create_model(backbone="resnet18", pretrained=False, heads={
        "c": {"type": "classification", "channels": 2, "scope": "image"}})
    with pytest.raises(ValueError, match="decoder"):
        image_only.eval().embed(x, source="decoder")


def test_embedder_module_matches_and_exports():
    from manyseg.export import export_model

    m = model()
    emb = m.embedder(pool="mean")
    x = torch.randn(1, 3, 64, 96)
    torch.testing.assert_close(emb(x), m.embed(x, pool="mean"))
    ep = export_model(emb, x)
    y = torch.randn(1, 3, 97, 131)
    torch.testing.assert_close(ep.module()(y), m.embed(y, pool="mean"), atol=1e-4, rtol=1e-4)


def test_late_fusion_features_cropped_per_input_and_embed_uses_the_primary():
    m = create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     fusion={"type": "late"},
                     inputs={"a": {"channels": 3},
                             "b": {"channels": 2, "alignment": "same_extent"}}).eval()
    x = {"a": torch.randn(1, 3, 97, 131), "b": torch.randn(1, 2, 33, 33)}
    feats = m.features(x)
    (a0, a1), (b0, b1) = m.backbone.groups
    for f, s in zip(feats[a0:a1], m.feature_specs[a0:a1]):
        assert f.shape[-2:] == (-(-97 // s.stride), -(-131 // s.stride))
    for f, s in zip(feats[b0:b1], m.feature_specs[b0:b1]):
        assert f.shape[-2:] == (-(-33 // s.stride), -(-33 // s.stride))
    torch.testing.assert_close(m.embed(x, pool="mean"), feats[a1 - 1].mean((-2, -1)))
