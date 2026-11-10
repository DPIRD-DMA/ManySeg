"""Regression tests for the fourth pre-release sweep of save/load, quantisation and the Hub: a
weights tag's architecture survives the rebuild, loaded programs take their dict inputs in any
order, a program's fixed size is its primary input's, quantize_weights keeps tied weights tied,
and Hub names with an empty revision are refused."""

import manyseg
import pytest
import timm
import torch
from manyseg import create_model, hub
from manyseg import weights as mw

ACT = {"logits": "softmax"}
KW = {"normalize": [{"type": "none"}], "activations": ACT}


# --- I1: a weights tag's native stem survives the rebuild ----------------------------------

def test_weights_tag_with_projection_adapter_saves_and_loads(tmp_path, monkeypatch):
    sd = timm.create_model("resnet18", in_chans=13, pretrained=False).state_dict()
    calls = []

    def fake(url):
        calls.append(url)
        return {k: v.clone() for k, v in sd.items()}

    monkeypatch.setattr(mw, "_download", fake)
    torch.manual_seed(0)
    m = create_model("resnet18", pretrained="ssl4eo_s2_moco", in_channels=13,
                     channel_adapter="projection", num_classes=2).eval()
    assert isinstance(m.backbone.input_adapter, torch.nn.Identity)
    path = manyseg.save(m, tmp_path / "m.safetensors", **KW)
    n = len(calls)
    loaded = manyseg.load(path)
    assert len(calls) == n  # nothing downloaded on load
    assert isinstance(loaded.backbone.input_adapter, torch.nn.Identity)
    assert loaded.backbone.stem_conv().in_channels == 13
    assert loaded.build_config["pretrained"] == "ssl4eo_s2_moco"
    x = torch.randn(1, 13, 64, 64)
    with torch.no_grad():
        assert torch.equal(m(x), loaded(x))
    # attribution survives, and the loaded model saves again
    assert [r["weights"] for r in mw.attribution(loaded)] == \
        [r["weights"] for r in mw.attribution(m)]
    manyseg.save(loaded, tmp_path / "again.safetensors", **KW)


# --- I2, I3: loaded multi-input programs ---------------------------------------------------

def _two_input_program(tmp_path, b_alignment=None, b_size=64, a_size=64, dynamic=True):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    torch.manual_seed(0)
    b = {"in_channels": 2, **({"alignment": b_alignment} if b_alignment else {})}
    m = create_model("resnet18", pretrained=False, num_classes=2,
                     inputs={"a": {"in_channels": 3}, "b": b}).eval()
    x = {"b": torch.randn(2, 2, b_size, b_size), "a": torch.randn(2, 3, a_size, a_size)}
    ep = quantize_pt2e(m, x, [x], reduce_range=True, dynamic=dynamic)
    path = manyseg.save(ep, tmp_path / "p.pt2", source=m, **KW)
    return m, x, manyseg.load(path)


def test_loaded_program_takes_dict_keys_in_any_order(tmp_path):
    _, x, loaded = _two_input_program(tmp_path)
    assert list(loaded.input_shapes) == ["b", "a"]
    with torch.no_grad():
        want = loaded(x)
        got = loaded({"a": x["a"], "b": x["b"]})
    assert torch.equal(want, got)


def test_loaded_program_names_missing_and_unexpected_keys(tmp_path):
    _, x, loaded = _two_input_program(tmp_path)
    with pytest.raises(ValueError, match=r"missing \['b'\]"):
        loaded({"a": x["a"]})
    with pytest.raises(ValueError, match=r"unexpected \['c'\]"):
        loaded({**x, "c": x["a"]})
    with pytest.raises(ValueError, match="takes a dict"):
        loaded(x["a"])


def test_single_input_program_refuses_a_dict(tmp_path):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    torch.manual_seed(0)
    m = create_model("resnet18", pretrained=False, num_classes=2).eval()
    x = torch.randn(2, 3, 64, 64)
    path = manyseg.save(quantize_pt2e(m, x, [x], reduce_range=True), tmp_path / "s.pt2",
                        source=m, **KW)
    loaded = manyseg.load(path)
    assert loaded(x).shape == (2, 2, 64, 64)
    with pytest.raises(ValueError, match="single tensor"):
        loaded({"x": x})


def test_fixed_size_is_the_primary_inputs(tmp_path):
    # the secondary input comes first in the example dict and has another size
    _, x, loaded = _two_input_program(tmp_path, b_alignment="same_extent", b_size=32,
                                      a_size=96, dynamic=False)
    assert loaded.fixed_size == (96, 96)
    assert loaded.tiling_spec()["max_size"] == (96, 96)
    assert loaded.input_shapes == {"b": [2, 2, 32, 32], "a": [2, 3, 96, 96]}
    assert loaded({"a": x["a"], "b": x["b"]}).shape == (2, 2, 96, 96)


def test_fixed_size_when_the_first_input_is_dynamic():
    from manyseg.quantize import _fixed_size

    shapes = {"b": [None, 2, None, None], "a": [None, 3, 96, 96]}
    assert _fixed_size(shapes, "a") == [96, 96]
    assert _fixed_size(shapes, "b") is None
    assert _fixed_size({"x": [2, 3, 64, 64]}, "x") == [64, 64]


# --- I4: quantize_weights keeps tied weights tied ------------------------------------------

def _unique_params(module):
    seen = {}
    for t in module.state_dict(keep_vars=True).values():
        seen.setdefault(id(t), t)
    return sum(t.numel() * t.element_size() if not hasattr(t, "tensor_data_names") else
               sum(getattr(t, p).numel() * getattr(t, p).element_size()
                   for p in t.tensor_data_names) for t in seen.values())


def test_quantize_weights_keeps_shared_encoders_tied(tmp_path):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_weights

    torch.manual_seed(0)
    m = create_model("resnet18", pretrained=False, num_classes=2,
                     inputs={"a": {"in_channels": 3},
                             "b": {"in_channels": 3, "backbone": "shared:a"}}).eval()
    enc = m.backbone.encoders
    assert enc["a"] is not enc["b"]
    q = quantize_weights(m)
    qa, qb = q.backbone.encoders["a"], q.backbone.encoders["b"]
    assert qa.stem_conv().weight is qb.stem_conv().weight
    assert hasattr(qa.stem_conv().weight, "tensor_data_names")
    # BatchNorm state stays per encoder
    bn_a = [b for b in qa.modules() if isinstance(b, torch.nn.BatchNorm2d)][0]
    bn_b = [b for b in qb.modules() if isinstance(b, torch.nn.BatchNorm2d)][0]
    assert bn_a is not bn_b and bn_a.running_mean is not bn_b.running_mean
    assert _unique_params(q) < _unique_params(m) / 3

    x = {"a": torch.randn(1, 3, 64, 64), "b": torch.randn(1, 3, 64, 64)}
    qpath = manyseg.save(q, tmp_path / "q.safetensors", **KW)
    fpath = manyseg.save(m, tmp_path / "f.safetensors", **KW)
    assert qpath.stat().st_size < fpath.stat().st_size / 3
    loaded = manyseg.load(qpath)
    la, lb = loaded.backbone.encoders["a"], loaded.backbone.encoders["b"]
    assert la.stem_conv().weight is lb.stem_conv().weight
    assert _unique_params(loaded) == _unique_params(q)
    with torch.no_grad():
        assert torch.equal(q(x), loaded(x))
        assert (q(x).argmax(1) == m(x).argmax(1)).float().mean() > 0.9


# --- I5: an empty Hub revision ---------------------------------------------------------------

def test_hub_name_with_empty_revision_is_refused():
    with pytest.raises(ValueError, match="hf-hub:<org>/<name>"):
        hub.split_name("hf-hub:org/name@")
    assert hub.split_name("hf-hub:org/name") == ("org/name", None)
    assert hub.split_name("hf-hub:org/name@v1") == ("org/name", "v1")


@pytest.mark.parametrize("repo_id", ["name", "a/b/c", "org/", "/name"])
def test_push_to_hub_names_the_repository_id_form(repo_id, monkeypatch):
    from manyseg import io as mio

    def no_save(*a, **k):
        raise AssertionError("saved before checking the name")

    monkeypatch.setattr(mio, "save", no_save)
    torch.manual_seed(0)
    m = create_model("resnet18", pretrained=False, num_classes=2)
    with pytest.raises(ValueError, match="push_to_hub takes a repository id 'org/name', got") as e:
        manyseg.push_to_hub(m, repo_id, **KW)
    assert "hf-hub:" not in str(e.value)


@pytest.mark.parametrize("repo_id", ["hf-hub:org/name", "org/name@", "hf-hub:name@dev"])
def test_push_to_hub_refuses_load_names(repo_id):
    with pytest.raises(ValueError, match="revision="):
        hub._check_repo_id(repo_id)
