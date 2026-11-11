"""Regression tests for the third pre-release sweep of save/load, export, quantisation, Hub upload
and describe: loaded programs keep their saved state, quantize_pt2e shares nothing with the float
model, time-series normalisation band counts, loaded models keep their pretrained tags, describe's
stage parameters, lazy submodules, push_to_hub repository ids, the partial normalize="weights"
error, quantised model cards, calibration shapes and the program's device check."""

import subprocess
import sys
from types import SimpleNamespace

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg import io as mio

ACT = {"logits": "softmax"}
KW = {"normalize": [{"type": "none"}], "activations": ACT}


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("num_classes", 2)
    return create_model(backbone="resnet18", pretrained=False, **kw).eval()


def _program(**kw):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    m = _model()
    x = torch.randn(2, 3, 64, 64)
    return m, x, quantize_pt2e(m, x, [x], reduce_range=True, **kw)


def _state(ep):
    return {k: (t.device, t.dtype, t.detach().clone()) for k, t in ep.state_dict.items()}


def _same_state(a, b):
    return a.keys() == b.keys() and all(
        a[k][0] == b[k][0] and a[k][1] == b[k][1] and torch.equal(a[k][2], b[k][2]) for k in a)


# --- I1: the execution module is a copy; .program stays as loaded --------------------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_program_loaded_on_mps_saves_again(tmp_path):
    m, x, ep = _program()
    path = manyseg.save(ep, tmp_path / "q.pt2", source=m, **KW)
    loaded = manyseg.load(path, map_location="mps")
    assert {t.device.type for t in loaded.program.state_dict.values()} == {"cpu"}
    again = manyseg.save(loaded, tmp_path / "again.pt2", source=m, **KW)
    assert manyseg.load(again).program.state_dict.keys() == loaded.program.state_dict.keys()
    out = loaded(x.to("mps"))
    assert out.device.type == "mps"


def test_program_state_unchanged_by_its_module(tmp_path):
    from manyseg.quantize import QuantizedProgram

    m, x, ep = _program()
    path = manyseg.save(ep, tmp_path / "q.pt2", source=m, **KW)
    loaded = manyseg.load(path)
    before = _state(loaded.program)
    loaded.module.to(torch.float64)  # the module may be moved or cast...
    assert _same_state(before, _state(loaded.program))  # ...without touching the program
    meta = manyseg.read_metadata(path)
    q = QuantizedProgram(loaded.program, meta, "meta")  # built on another device
    assert next(iter(q.module.buffers())).device.type == "meta"
    assert _same_state(before, _state(loaded.program))
    manyseg.save(loaded, tmp_path / "again.pt2", source=m, **KW)


# --- I2: quantize_pt2e shares no tensors with the float model -------------------------------

def test_quantize_pt2e_shares_nothing_with_the_float_model():
    m, x, ep = _program()
    own = {t.data_ptr() for t in [*m.parameters(), *m.buffers()]}
    assert not [k for k, t in ep.state_dict.items() if t.data_ptr() in own]
    with torch.no_grad():
        before = ep.module()(x).clone()
        m.heads.logits.conv.bias.add_(5.0)
        assert torch.equal(ep.module()(x), before)
    assert not m.training  # its mode is untouched


def test_export_model_documents_shared_parameters():
    from manyseg.export import export_model

    assert "shares its parameters" in export_model.__doc__


# --- I3: band counts include time series (channels per date) --------------------------------

def test_timeseries_normalisation_band_count_is_checked(tmp_path):
    m = _model(inputs={"ts": {"kind": "timeseries", "in_channels": 4}})
    bad = {"ts": [{"type": "zscore", "mean": [1, 2, 3], "std": [1, 1, 1]}]}
    with pytest.raises(ValueError, match="3|4"):
        manyseg.save(m, tmp_path / "bad.safetensors", normalize=bad, activations=ACT)
    good = {"ts": [{"type": "zscore", "mean": [1, 2, 3, 4], "std": [1, 1, 1, 1]}]}
    path = manyseg.save(m, tmp_path / "good.safetensors", normalize=good, activations=ACT)
    assert manyseg.read_metadata(path)["normalize"] == good


# --- I4: a loaded model keeps its create_model config ---------------------------------------

def _tagged(monkeypatch, **kw):
    import timm
    from manyseg import weights as mw

    sd = timm.create_model("resnet18", in_chans=13, pretrained=False).state_dict()
    monkeypatch.setattr(mw, "_download", lambda url: {k: v.clone() for k, v in sd.items()})
    return create_model("resnet18", pretrained="ssl4eo_s2_moco", **kw).eval()


def test_loaded_model_keeps_pretrained_tag(tmp_path, monkeypatch):
    m = _tagged(monkeypatch, in_channels=13, heads={
        "seg": {"type": "segmentation", "out_channels": 2},
        "r": {"type": "regression", "out_channels": 1}})
    acts = {"seg": "softmax"}
    path = manyseg.save(m, tmp_path / "a.safetensors", normalize="weights", activations=acts)
    loaded = manyseg.load(path)
    assert loaded.build_config == m.build_config
    again = manyseg.save(loaded, tmp_path / "b.safetensors", normalize="weights",
                         activations=acts)
    meta = manyseg.read_metadata(again)
    assert meta["create_model"]["pretrained"] == "ssl4eo_s2_moco"
    assert meta["normalize"] == manyseg.read_metadata(path)["normalize"]
    # head selection after loading still saves and loads
    loaded.select_heads(["seg"])
    third = manyseg.save(loaded, tmp_path / "c.safetensors", normalize="weights",
                         activations=acts)
    assert list(manyseg.load(third).heads) == ["seg"]


# --- I5: describe counts a shared decoder once ----------------------------------------------

def test_stage_params_sum_to_model_params_with_shared_decoder():
    from manyseg.describe import model_graph

    m = _model(num_classes=None, inputs={"a": {"in_channels": 4}, "b": {"in_channels": 4}},
               heads={"sa": {"type": "segmentation", "out_channels": 2, "input": "a"},
                      "sb": {"type": "segmentation", "out_channels": 2, "input": "b",
                             "share": "sa"}})
    g = model_graph(m)
    assert sum(g.stage_params.values()) == sum(p.numel() for p in m.parameters())


# --- I6: submodules reachable after import manyseg ------------------------------------------

def test_submodules_reachable_lazily():
    code = ("import sys, manyseg\n"
            "assert 'torchao' not in sys.modules\n"
            "assert callable(manyseg.describe.describe)\n"
            "assert callable(manyseg.export.dynamic_shapes)\n"
            "assert callable(manyseg.quantize.check_deployment)\n"
            "assert 'torchao' not in sys.modules\n"
            "try:\n    manyseg.nope\nexcept AttributeError:\n    pass\n"
            "else:\n    raise SystemExit('no AttributeError')\n")
    subprocess.run([sys.executable, "-c", code], check=True)


# --- I7: push_to_hub refuses load-style names before saving ---------------------------------

@pytest.mark.parametrize("repo_id", ["hf-hub:org/name", "org/name@dev", "hf-hub:org/name@dev"])
def test_push_to_hub_refuses_load_names_before_saving(repo_id, monkeypatch):
    def no_save(*a, **k):
        raise AssertionError("saved before checking the name")

    monkeypatch.setattr(mio, "save", no_save)
    with pytest.raises(ValueError, match="revision="):
        manyseg.push_to_hub(_model(), repo_id, **KW)


# --- I8: normalize="weights" names the untagged inputs --------------------------------------

def test_partial_weights_normalize_names_inputs(tmp_path, monkeypatch):
    m = _tagged(monkeypatch, num_classes=2,
                inputs={"s2": {"in_channels": 13},
                        "dem": {"in_channels": 1, "backbone": "resnet18", "pretrained": False}})
    with pytest.raises(ValueError, match=r"\['dem'\].*for each input"):
        manyseg.save(m, tmp_path / "m.safetensors", normalize="weights", activations=ACT)
    with pytest.raises(ValueError, match="was not built with remote-sensing"):
        manyseg.save(_model(), tmp_path / "m.safetensors", normalize="weights",
                     activations=ACT)


# --- I9: model card of a quantised model ----------------------------------------------------

def test_card_names_quantisation(tmp_path):
    pytest.importorskip("torchao")
    from manyseg.hub import card
    from manyseg.quantize import quantize_weights

    q = quantize_weights(_model(), "int8_dynamic")
    meta = manyseg.read_metadata(manyseg.save(q, tmp_path / "q.safetensors", **KW))
    text = card(meta, "org/name")
    assert "int8_dynamic" in text and "manyseg[quant]" in text
    float_meta = manyseg.read_metadata(manyseg.save(_model(), tmp_path / "f.safetensors", **KW))
    assert "manyseg[quant]" not in card(float_meta, "org/name")


# --- I10: calibration shapes are checked against the export ---------------------------------

@pytest.mark.parametrize(("kw", "size", "match"), [
    ({"dynamic": False}, 96, "fixes it at the example's 64.*dynamic=True"),
    ({}, 16, r"outside the export's range \[\d+, 4096\]"),
    ({"max_size": 128}, 256, "max_size="),
])
def test_calibration_shape_errors(kw, size, match):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    m, x = _model(), torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError, match=match):
        quantize_pt2e(m, x, [torch.randn(2, 3, size, size)], reduce_range=True, **kw)


def test_calibration_within_dynamic_bounds_is_accepted():
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_pt2e

    m, x = _model(), torch.randn(2, 3, 64, 64)
    quantize_pt2e(m, x, [x, torch.randn(2, 3, 96, 96)], reduce_range=True, max_size=128)


# --- related: the device check compares the resolved device, index included -----------------

def test_program_device_is_resolved_and_compared_exactly(tmp_path):
    m, x, ep = _program()
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=m, **KW))
    assert loaded.device == torch.device("cpu")
    loaded._check("x", x)
    loaded.device = torch.device("cuda", 1)  # as if loaded on the second GPU
    on_gpu0 = SimpleNamespace(device=torch.device("cuda", 0), ndim=4, shape=(2, 3, 64, 64))
    with pytest.raises(ValueError, match="cuda:1.*cuda:0"):
        loaded._check("x", on_gpu0)
    if torch.backends.mps.is_available():
        on_mps = manyseg.load(tmp_path / "q.pt2", map_location="mps")
        assert on_mps.device == torch.device("mps", 0)
        on_mps._check("x", x.to("mps"))
