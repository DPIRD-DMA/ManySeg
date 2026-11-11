"""Regression tests for the second pre-release sweep of save/load, quantisation, export and Hub
upload: shared encoders with normalize="weights", quantised half-precision models, None names,
compare_outputs' eval mode, quantize_pt2e's reduce_range with a custom quantizer, push_to_hub of
programs, quantize_weights that quantises nothing, the input kind default, .pt2 suffixes for
float models, and export_model's ``dynamic``."""

import warnings

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg import io as mio

KW = {"normalize": [{"type": "none"}], "activations": {"logits": "softmax"}}


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("num_classes", 3)
    return create_model(backbone="resnet18", pretrained=False, **kw).eval()


# --- I1: normalize="weights" follows "shared:<input>" ----------------------------------------

def test_weights_normalize_follows_shared_encoder(tmp_path, monkeypatch):
    import timm
    from manyseg import weights as mw

    sd = timm.create_model("resnet18", in_chans=13, pretrained=False).state_dict()
    monkeypatch.setattr(mw, "_download", lambda url: {k: v.clone() for k, v in sd.items()})
    m = create_model("resnet18", pretrained="ssl4eo_s2_moco", num_classes=2,
                     inputs={"a": {"in_channels": 13},
                             "b": {"in_channels": 13, "backbone": "shared:a"}})
    spec = mio._weights_normalize(m.build_config)
    assert set(spec) == {"a", "b"} and spec["a"] == spec["b"]
    path = manyseg.save(m, tmp_path / "m.safetensors", normalize="weights",
                        activations={"logits": "softmax"})
    assert manyseg.read_metadata(path)["normalize"] == spec
    # A shared input behind a learned projection has its own bands: the weights' spec is wrong.
    p = create_model("resnet18", pretrained="ssl4eo_s2_moco", num_classes=2,
                     inputs={"a": {"in_channels": 13},
                             "b": {"in_channels": 2, "backbone": "shared:a",
                                   "channel_adapter": "projection"}})
    with pytest.raises(ValueError, match="learned projection"):
        manyseg.save(p, tmp_path / "p.safetensors", normalize="weights",
                     activations={"logits": "softmax"})


# --- I2: quantize_weights on a half / bfloat16 model saves ----------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantized_half_model_saves_and_loads(tmp_path, dtype):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_weights

    q = quantize_weights(_model().to(dtype))
    path = manyseg.save(q, tmp_path / "q.safetensors", **KW)
    loaded = manyseg.load(path)
    x = torch.randn(1, 3, 64, 64, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), q(x))


# --- I3: None names are "no names" ------------------------------------------------------------

def test_none_class_and_band_names_are_skipped(tmp_path):
    m = create_model("resnet18", pretrained=False,
                     heads={"seg": {"type": "segmentation", "out_channels": 2},
                            "h": {"type": "regression", "out_channels": 1}},
                     inputs={"a": {"in_channels": 2}, "b": {"in_channels": 3}})
    path = manyseg.save(m, tmp_path / "m.safetensors",
                        normalize={"a": [{"type": "none"}], "b": [{"type": "none"}]},
                        activations={"seg": "softmax"},
                        classes={"seg": ["x", "y"], "h": None},
                        bands={"a": ["r", "g"], "b": None})
    meta = manyseg.read_metadata(path)
    assert meta["classes"] == {"seg": ["x", "y"]} and meta["bands"] == {"a": ["r", "g"]}
    spec = manyseg.load(path).tiling_spec()
    assert spec["bands"] == {"a": ["r", "g"]}


# --- I4: compare_outputs runs modules in eval mode -------------------------------------------

def test_compare_outputs_uses_eval_mode_and_restores_it():
    from manyseg.quantize import compare_outputs

    m = _model().train()
    bn = m.backbone.model.bn1
    before = bn.running_mean.clone()
    r = compare_outputs(m, m, [torch.randn(2, 3, 64, 64)])
    assert r["max_abs_diff"] == 0 and r["argmax_agreement"] == 1
    assert torch.equal(bn.running_mean, before)
    assert m.training and bn.training


def test_compare_outputs_takes_an_exported_module():
    from manyseg.quantize import compare_outputs

    m = _model()
    x = torch.randn(1, 3, 64, 64)
    ep = torch.export.export(m, (x,))
    assert compare_outputs(m, ep.module(), [x])["max_abs_diff"] < 1e-4


# --- I5: reduce_range with a custom quantizer is refused --------------------------------------

def test_quantize_pt2e_refuses_reduce_range_with_quantizer():
    pytest.importorskip("torchao")
    from manyseg.quantize import _x86_quantizer, quantize_pt2e

    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="reduce_range"):
        quantize_pt2e(_model(), x, [x], quantizer=_x86_quantizer(False), reduce_range=True)


# --- I6: push_to_hub refuses programs up front -------------------------------------------------

def test_push_to_hub_refuses_programs(monkeypatch):
    import huggingface_hub
    from manyseg.hub import push_to_hub

    def no_hub(*a, **k):
        raise AssertionError("contacted the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", no_hub)
    x = torch.randn(1, 3, 64, 64)
    ep = torch.export.export(_model(), (x,))
    with pytest.raises(TypeError, match="DenseModel"):
        push_to_hub(ep, "org/name", **KW)


# --- I8: quantize_weights that quantises nothing ----------------------------------------------

def test_quantize_weights_warns_when_nothing_is_quantised(tmp_path):
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_weights

    with pytest.warns(UserWarning, match="quantised no layer"):
        q = quantize_weights(_model(), conv=False)
    assert getattr(q, "quantization", None) is None
    path = manyseg.save(q, tmp_path / "m.safetensors", **KW)
    assert "quantization" not in manyseg.read_metadata(path)


def test_quantize_weights_does_not_warn_when_it_quantises():
    pytest.importorskip("torchao")
    from manyseg.quantize import quantize_weights

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        q = quantize_weights(_model())
    assert q.quantization == {"mode": "int8_weight_only", "conv": True}


# --- I9: an input without "kind" is a raster ---------------------------------------------------

def test_card_input_kind_defaults_to_raster(tmp_path):
    from manyseg.hub import card

    m = create_model("resnet18", pretrained=False, num_classes=2,
                     inputs={"s2": {"in_channels": 4}, "s1": {"in_channels": 2}})
    path = manyseg.save(m, tmp_path / "m.safetensors",
                        normalize={"s2": [{"type": "none"}], "s1": [{"type": "none"}]},
                        activations={"logits": "softmax"})
    rows = [r for r in card(manyseg.read_metadata(path), "org/x").splitlines()
            if r.startswith("| `s")]
    assert len(rows) == 2 and all("| raster |" in r for r in rows)
    assert {v["kind"] for v in m.tiling_spec()["inputs"].values()} == {"raster"}


# --- I11: a DenseModel does not save to .pt2 ---------------------------------------------------

def test_save_refuses_pt2_suffix_for_a_dense_model(tmp_path):
    with pytest.raises(ValueError, match="quantize_pt2e"):
        manyseg.save(_model(), tmp_path / "m.pt2", **KW)
    assert not (tmp_path / "m.pt2").exists()


# --- I12: one name for dynamic H/W across export_model, dynamic_shapes, quantize_pt2e -----------

def test_export_model_dynamic_false_fixes_hw():
    from manyseg.export import dynamic_shapes, export_model

    m = _model()
    x = torch.randn(2, 3, 64, 64)
    assert dynamic_shapes(m, x, dynamic=False) is None
    ep = export_model(m, x, dynamic=False)
    assert list(ep.module()(x).shape) == [2, 3, 64, 64]
    with pytest.raises(Exception):  # noqa: B017 - torch's guard error type varies by version
        ep.module()(torch.randn(2, 3, 96, 64))
    ep = export_model(m, x, dynamic=False, dynamic_batch=True)
    assert ep.module()(torch.randn(3, 3, 64, 64)).shape[0] == 3


def test_export_model_refuses_dynamic_hw_for_window_attention(monkeypatch):
    from manyseg import export

    monkeypatch.setattr(export, "_windowed", lambda model: "WindowAttention")
    with pytest.raises(ValueError, match="dynamic=False"):
        export.export_model(_model(), torch.randn(1, 3, 64, 64))


def test_weights_normalize_slices_early_fusion_inputs(tmp_path, monkeypatch):
    """Early fusion stacks the inputs in front of one encoder: each input gets its own share
    of the weights' per-band statistics, not all of them."""
    import timm
    from manyseg import weights as mw

    sd = timm.create_model("resnet18", in_chans=13, pretrained=False).state_dict()
    monkeypatch.setattr(mw, "_download", lambda url: {k: v.clone() for k, v in sd.items()})
    m = create_model("resnet18", pretrained="ssl4eo_s2_moco", num_classes=2,
                     inputs={"rgb": {"in_channels": 3}, "rest": {"in_channels": 10}},
                     fusion={"type": "early"})
    full = mw.get_weights("resnet18", "ssl4eo_s2_moco")
    path = manyseg.save(m, tmp_path / "m.safetensors", normalize="weights",
                        activations={"logits": "softmax"})
    spec = manyseg.read_metadata(path)["normalize"]
    z = {n: next(s for s in steps if s["type"] == "zscore") for n, steps in spec.items()}
    assert z["rgb"]["mean"] == list(full.mean)[:3] and z["rest"]["mean"] == list(full.mean)[3:]
    assert z["rgb"]["std"] == list(full.std)[:3] and z["rest"]["std"] == list(full.std)[3:]
    # the weights tag may be set on any input, as create_model reads it
    m2 = create_model("resnet18", pretrained=False, num_classes=2, fusion={"type": "early"},
                      inputs={"rgb": {"in_channels": 3},
                              "rest": {"in_channels": 10, "pretrained": "ssl4eo_s2_moco"}})
    assert mio._weights_normalize(m2.build_config) == spec


def test_save_checks_normalisation_band_counts(tmp_path):
    """A per-band mean of the wrong length would pass ManyPred's validate_spec and fail only
    at prediction time; save refuses it."""
    m = create_model("resnet18", pretrained=False, in_channels=4, num_classes=2)
    ok = {"type": "zscore", "mean": [0.0] * 4, "std": [1.0] * 4}
    manyseg.save(m, tmp_path / "ok.safetensors", normalize=ok, activations={"logits": "softmax"})
    manyseg.save(m, tmp_path / "s.safetensors", normalize={"type": "scale", "value": 255},
                 activations={"logits": "softmax"})
    with pytest.raises(ValueError, match="mean has 3 values but the input has 4 bands"):
        manyseg.save(m, tmp_path / "bad.safetensors",
                     normalize={"type": "zscore", "mean": [0.0] * 3, "std": [1.0] * 4},
                     activations={"logits": "softmax"})
    mi = create_model("resnet18", pretrained=False, num_classes=2,
                      inputs={"a": {"in_channels": 2}, "b": {"in_channels": 5}})
    with pytest.raises(ValueError, match="input 'b'.*max has 2 values but the input has 5"):
        manyseg.save(mi, tmp_path / "mi.safetensors", activations={"logits": "softmax"},
                     normalize={"a": {"type": "minmax", "min": [0, 0], "max": [1, 1]},
                                "b": {"type": "minmax", "min": 0, "max": [1, 1]}})
