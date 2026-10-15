"""DenseModel.select_heads: keep only the heads needed at deployment (e.g. train with an
auxiliary pixel head, deploy the classifier alone; E020)."""

import pytest
import torch
from manyseg import create_model

HEADS = {"seg": {"type": "segmentation", "channels": 4},
         "edges": {"type": "segmentation", "channels": 1, "decoder": "fpn"},
         "labels": {"type": "classification", "channels": 4, "scope": "image"}}


def model(**kw):
    return create_model(backbone="resnet18", pretrained=False, heads=HEADS, **kw).eval()


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_dropping_pixel_heads_drops_their_decoders_and_keeps_outputs():
    m = model()
    x = torch.randn(1, 3, 64, 96)
    ref = m(x)["labels"]
    before = nparams(m)
    out = m.select_heads(["labels"])
    assert out is m and set(m.heads) == {"labels"}
    assert m.decoder is None and len(m.task_decoders) == 0
    assert nparams(m) < before
    y = m(x)
    assert set(y) == {"labels"}
    torch.testing.assert_close(y["labels"], ref, rtol=0, atol=0)


def test_keeping_a_pixel_head_keeps_its_decoder():
    m = model().select_heads(["seg"])
    assert m.decoder is not None and len(m.task_decoders) == 0
    assert m(torch.randn(1, 3, 64, 64))["seg"].shape == (1, 4, 64, 64)
    m = model().select_heads(["edges", "labels"])
    assert m.decoder is None and set(m.task_decoders) == {"edges"}


def test_select_heads_with_metadata_conditioning():
    inputs = {"img": {"channels": 3},
              "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic",
                                                            "period": 365.25}}}}
    m = create_model(backbone="resnet18", pretrained=False, heads=HEADS, inputs=inputs).eval()
    with torch.no_grad():
        for p in m.conditioner.film.parameters():
            p.normal_(0, 0.3)
    x = {"img": torch.randn(2, 3, 64, 64), "meta": torch.tensor([[10.0], [200.0]])}
    ref = m(x)["labels"]
    m.select_heads(["labels"])
    assert set(m.conditioner.film) == {"image"}
    torch.testing.assert_close(m(x)["labels"], ref, rtol=0, atol=0)


@pytest.mark.slow
def test_select_heads_exports():
    from manyseg.export import export_model

    m = model().select_heads(["labels"])
    ep = export_model(m, torch.randn(1, 3, 64, 96))
    x = torch.randn(1, 3, 97, 131)
    torch.testing.assert_close(ep.module()(x)["labels"], m(x)["labels"], atol=1e-5, rtol=1e-5)


def test_pruned_checkpoint_round_trip():
    m = model().select_heads(["labels"])
    sd = m.state_dict()
    with pytest.raises(RuntimeError, match="Missing"):
        model().load_state_dict(sd)  # the full model expects the removed modules
    fresh = model().select_heads(["labels"])
    fresh.load_state_dict(sd)
    x = torch.randn(1, 3, 64, 64)
    torch.testing.assert_close(fresh(x)["labels"], m(x)["labels"], rtol=0, atol=0)


def test_errors():
    with pytest.raises(ValueError, match="unknown"):
        model().select_heads(["nope"])
    with pytest.raises(ValueError, match="at least one"):
        model().select_heads([])
    single = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    with pytest.raises(ValueError, match="single"):
        single.select_heads(["x"])
    assert single.select_heads(["logits"]) is single
