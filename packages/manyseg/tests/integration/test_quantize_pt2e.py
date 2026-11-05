"""manyseg.quantize.quantize_pt2e: static int8 (x86 inductor quantizer) through torch.export."""

import io
import json
import pathlib
import sys
import zipfile

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg.export import export_model
from manyseg.quantize import (
    QuantizedProgram,
    _safe_expression,
    compare_outputs,
    model_size,
    quantize_pt2e,
)

pytestmark = pytest.mark.slow
pytest.importorskip("torchao")  # the PT2E path needs the manyseg[quant] extra


def _batches(n, hw=(64, 96), c=3):
    return [torch.randn(1, c, *hw) for _ in range(n)]


@pytest.fixture(scope="module")
def float_and_int8():
    torch.manual_seed(0)
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(4))
    return model, q


def test_pt2e_returns_exported_program(float_and_int8):
    _, q = float_and_int8
    assert isinstance(q, torch.export.ExportedProgram)
    weights = {t.dtype for t in q.state_dict.values() if t.ndim == 4}
    assert weights == {torch.int8}


def test_pt2e_dynamic_hw_shapes(float_and_int8):
    model, q = float_and_int8
    for hw in [(64, 96), (97, 131), (200, 65)]:
        assert q.module()(torch.randn(1, 3, *hw)).shape == (1, 3, *hw)


def test_pt2e_agreement_and_size(float_and_int8):
    model, q = float_and_int8
    r = compare_outputs(model, q, _batches(3, (97, 131)))
    assert r["argmax_agreement"] > 0.9
    # int8 weights, and the float originals convert_pt2e leaves registered are dropped
    assert model_size(q) < 0.35 * model_size(export_model(model, torch.randn(1, 3, 64, 96)))


def test_pt2e_dict_inputs():
    model = create_model(inputs={"a": {"in_channels": 4}, "b": {"in_channels": 2}},
                         backbone="resnet18", pretrained=False, num_classes=2).eval()

    def mk(h, w):
        return {"a": torch.randn(1, 4, h, w), "b": torch.randn(1, 2, h, w)}

    q = quantize_pt2e(model, mk(64, 96), [mk(64, 96) for _ in range(3)])
    assert q.module()(mk(97, 131)).shape == (1, 2, 97, 131)
    assert compare_outputs(model, q, [mk(97, 131)])["argmax_agreement"] > 0.9


@pytest.mark.parametrize("backbone,decoder", [
    ("resnet18", "unetplusplus"), ("resnet18", "fastai_unet"), ("resnet18", "fpn"),
    ("resnet18", "deeplabv3plus"), ("resnet18", "upernet"), ("resnet18", "segformer"),
    ("convnext_atto", "unet"), ("mobilenetv3_small_100", "unet"), ("caformer_s18", "unet"),
    ("pvt_v2_b0", "unet"), ("vit_tiny_patch16_224", "unet"),
])
def test_pt2e_backbones_and_decoders(backbone, decoder):
    model = create_model(backbone=backbone, decoder=decoder, pretrained=False,
                         num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2))
    x = [torch.randn(1, 3, 97, 131)]
    assert q.module()(x[0]).shape == (1, 3, 97, 131)
    assert compare_outputs(model, q, x)["argmax_agreement"] > 0.8


def test_pt2e_multi_head_dict_output():
    model = create_model(backbone="resnet18", pretrained=False, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "height": {"type": "regression", "out_channels": 1},
        "scene": {"type": "classification", "out_channels": 4, "scope": "image"}}).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2))
    out = q.module()(torch.randn(1, 3, 97, 131))
    assert out["seg"].shape == (1, 3, 97, 131) and out["scene"].shape == (1, 4)
    r = compare_outputs(model, q, _batches(2, (97, 131)))
    assert r["seg/argmax_agreement"] > 0.9


def test_pt2e_dynamic_batch():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(2, 3, 64, 96), [torch.randn(2, 3, 64, 96)],
                      dynamic_batch=True)
    assert q.module()(torch.randn(3, 3, 97, 131)).shape == (3, 3, 97, 131)


def test_pt2e_save_load_round_trip(float_and_int8, tmp_path):
    _, q = float_and_int8
    torch.export.save(q, tmp_path / "int8.pt2")
    loaded = torch.export.load(tmp_path / "int8.pt2").module()
    x = torch.randn(1, 3, 97, 131)
    torch.testing.assert_close(loaded(x), q.module()(x))


def test_pt2e_compiles_with_freezing(float_and_int8):
    """The deployment path: inductor lowers the q/dq graph to oneDNN int8 kernels. Compiled int8
    is not numerically the eager q/dq reference (documented), so only shape and finiteness."""
    import torch._inductor.config as inductor_config

    _, q = float_and_int8
    with inductor_config.patch(freezing=True), torch.no_grad():
        out = torch.compile(q.module())(torch.randn(1, 3, 97, 131))
    assert out.shape == (1, 3, 97, 131) and out.isfinite().all()


def test_pt2e_fixed_size():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2), dynamic=False)
    assert q.module()(torch.randn(1, 3, 64, 96)).shape == (1, 3, 64, 96)


def test_pt2e_window_attention_needs_fixed_size():
    """Swin cannot export with dynamic H/W: the error names the fixed-size way out, which
    works."""
    model = create_model(backbone="swin_tiny_patch4_window7_224", pretrained=False,
                         num_classes=2).eval()
    ex = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError, match="dynamic=False"):
        quantize_pt2e(model, ex, [ex])
    q = quantize_pt2e(model, ex, [ex], dynamic=False)
    assert q.module()(ex).shape == (1, 2, 224, 224)


def test_pt2e_unrelated_window_size_attribute_is_not_rejected():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    model.decoder.window_size = 7
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(1))
    assert q.module()(torch.randn(1, 3, 97, 131)).shape == (1, 3, 97, 131)


def test_pt2e_rejects_empty_calibration():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    with pytest.raises(ValueError, match="calibration"):
        quantize_pt2e(model, torch.randn(1, 3, 64, 96), [])


# --- save / load ---------------------------------------------------------------------------

_NORM = [{"type": "scale", "value": 10000}]


def _args(model):
    return {"activations": {h: "softmax" for h in model.heads}, "normalize": _NORM}


def _reduced_head_model():
    return create_model(backbone="resnet18", pretrained=False, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "coarse": {"type": "segmentation", "out_channels": 2, "scale": 0.25},
        "label": {"type": "classification", "scope": "image", "out_channels": 4},
        "height": {"type": "regression", "out_channels": 1}}).eval()


@pytest.fixture(scope="module")
def saved(float_and_int8, tmp_path_factory):
    model, q = float_and_int8
    return manyseg.save(q, tmp_path_factory.mktemp("pt2") / "q.pt2", source=model, **_args(model))


def test_save_round_trip_same_outputs(float_and_int8, saved):
    _, q = float_and_int8
    loaded = manyseg.load(saved)
    assert isinstance(loaded, QuantizedProgram) and not isinstance(loaded, torch.nn.Module)
    for hw in [(64, 96), (97, 131), (65, 200)]:
        x = torch.randn(1, 3, *hw)
        assert torch.equal(q.module()(x), loaded(x))
    assert compare_outputs(q, loaded, [torch.randn(1, 3, 97, 131)])["max_abs_diff"] == 0


def test_saved_program_still_works_after_save(float_and_int8, saved):
    """Saving strips the sample inputs from a copy, not from the caller's program."""
    _, q = float_and_int8
    assert q.example_inputs is not None and saved.exists()


def test_saved_size_and_metadata(float_and_int8, saved):
    model, q = float_and_int8
    assert saved.stat().st_size < 0.35 * model_size(export_model(model, torch.randn(1, 3, 64, 96)))
    meta = manyseg.read_metadata(saved)
    assert meta["format"] == 3
    r = meta["quantization"]
    assert r["route"] == "pt2e" and r["batch_size"] == 1 and r["fixed_size"] is None
    assert r["input_shapes"] == {"x": [1, 3, None, None]}
    assert r["reduce_range"] in (True, False)


def test_saved_tiling_spec_matches_float(tmp_path):
    model = _reduced_head_model()
    args = {"normalize": _NORM, "activations": {"seg": "softmax", "coarse": "sigmoid",
                                                "label": "softmax"},
            "classes": {"seg": ["a", "b", "c"]}, "bands": ["r", "g", "b"]}
    want = manyseg.load(manyseg.save(model, tmp_path / "f.safetensors", **args)).tiling_spec()
    ep = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2))
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **args))
    spec = loaded.tiling_spec()
    assert spec == want
    assert spec["outputs"]["coarse"]["scale"] == 0.25
    assert spec["normalize"] == _NORM
    assert spec["outputs"]["seg"]["classes"] == ["a", "b", "c"]
    assert "classes" not in spec["outputs"]["coarse"]
    assert spec["bands"] == ["r", "g", "b"]
    x = torch.randn(1, 3, 97, 131)
    out, ref = loaded(x), ep.module()(x)
    assert set(out) == set(ref) and all(torch.equal(out[k], ref[k]) for k in ref)
    assert out["coarse"].shape[-2:] == (25, 33)


def test_saved_multi_input(tmp_path):
    model = create_model(inputs={"a": {"in_channels": 4}, "b": {"in_channels": 2}},
                         backbone="resnet18", pretrained=False, num_classes=2).eval()

    def mk(h, w):
        return {"a": torch.randn(1, 4, h, w), "b": torch.randn(1, 2, h, w)}

    ep = quantize_pt2e(model, mk(64, 96), [mk(64, 96) for _ in range(2)])
    args = {**_args(model), "normalize": {"a": _NORM, "b": _NORM}}
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **args))
    x = mk(97, 131)
    assert torch.equal(ep.module()(x), loaded(x))
    assert loaded.input_shapes == {"a": [1, 4, None, None], "b": [1, 2, None, None]}
    assert loaded.tiling_spec()["inputs"] == model.tiling_spec()["inputs"]


def test_saved_batch_size_is_exposed_and_enforced(float_and_int8, saved, tmp_path):
    loaded = manyseg.load(saved)
    assert loaded.batch_size == 1
    with pytest.raises(ValueError, match="batch size 1.*dynamic_batch=True"):
        loaded(torch.randn(2, 3, 64, 96))
    model, _ = float_and_int8
    calib = [torch.randn(2, 3, 64, 96) for _ in range(2)]
    ep = quantize_pt2e(model, calib[0], calib, dynamic_batch=True)
    dyn = manyseg.load(manyseg.save(ep, tmp_path / "d.pt2", source=model, **_args(model)))
    assert dyn.batch_size is None
    for n in (1, 2, 3):
        assert dyn(torch.randn(n, 3, 65, 97)).shape == (n, 3, 65, 97)


@pytest.mark.parametrize("backbone,size", [("swin_tiny_patch4_window7_224", 224),
                                           ("davit_tiny", 224)])
def test_saved_window_attention_records_fixed_size(tmp_path, backbone, size):
    model = create_model(backbone=backbone, pretrained=False, num_classes=2).eval()
    example = torch.randn(1, 3, size, size)
    ep = quantize_pt2e(model, example, [example], dynamic=False)
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    assert loaded.fixed_size == (size, size)
    assert loaded.tiling_spec()["max_size"] == (size, size)
    x = torch.randn(1, 3, size, size)
    assert torch.equal(ep.module()(x), loaded(x))
    with pytest.raises(ValueError, match="fixed input shape"):
        loaded(torch.randn(1, 3, size - 32, size))


def test_fixed_size_overrides_max_size(tmp_path):
    """A CNN quantised at a fixed size: the spec says so, though its float model has no limit."""
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    x = torch.randn(1, 3, 64, 64)
    ep = quantize_pt2e(model, x, [x], dynamic=False)
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    assert loaded.fixed_size == (64, 64) and loaded.tiling_spec()["max_size"] == (64, 64)
    assert model.tiling_spec()["max_size"] is None


def test_saved_program_runs_on_plain_callable_protocol(saved):
    """What ManyPred relies on: no .to / .eval to call, a tiling_spec() on the callable."""
    loaded = manyseg.load(saved)
    assert callable(loaded) and callable(loaded.tiling_spec)
    assert not hasattr(loaded, "eval") and not hasattr(loaded, "to")


def test_compiled_loaded_module(saved):
    loaded = manyseg.load(saved)
    x = torch.randn(1, 3, 64, 96)
    old = torch._inductor.config.freezing
    torch._inductor.config.freezing = True
    try:
        assert torch.compile(loaded.module)(x).shape == (1, 3, 64, 96)
    finally:
        torch._inductor.config.freezing = old


def test_file_holds_no_pickle(saved):
    with zipfile.ZipFile(saved) as z:
        for n in z.namelist():
            assert not (n.endswith(".pkl") or z.read(n)[:1] == b"\x80"), n
        sample = next(n for n in z.namelist() if "/sample_inputs/" in n)
        assert z.getinfo(sample).file_size == 0


def _rewrite(src, dst, change):
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst, "w") as b:
        for n in a.namelist():
            data = change(n, a.read(n))
            if data is not None:
                b.writestr(n, data)
    return dst


class _Payload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (pathlib.Path.touch, (pathlib.Path(self.marker),))


def _crafted_sample_inputs(tmp_path, marker):
    buf = io.BytesIO()
    torch.save(_Payload(marker), buf)
    return lambda n, data: buf.getvalue() if "/sample_inputs/" in n else data


def test_load_refuses_pickled_sample_inputs_without_running_them(saved, tmp_path):
    marker = tmp_path / "ran"
    bad = _rewrite(saved, tmp_path / "bad.pt2", _crafted_sample_inputs(tmp_path, marker))
    with pytest.raises(ValueError, match="run code"):
        manyseg.load(bad)
    assert not marker.exists()


def test_load_refuses_entries_marked_pickled(saved, tmp_path):
    def change(n, data):
        if n.endswith("model_weights_config.json"):
            cfg = json.loads(data)
            next(iter(cfg["config"].values()))["use_pickle"] = True
            return json.dumps(cfg)
        return data

    with pytest.raises(ValueError, match="run code"):
        manyseg.load(_rewrite(saved, tmp_path / "bad.pt2", change))


def test_load_refuses_legacy_weights_entry(saved, tmp_path):
    def change(n, data):
        return b"junk" if n.endswith("/data/weights/weight_0") else data

    with zipfile.ZipFile(saved) as z:
        root = z.namelist()[0].split("/")[0]
    bad = _rewrite(saved, tmp_path / "bad.pt2", change)
    with zipfile.ZipFile(bad, "a") as z:
        z.writestr(f"{root}/data/weights/model.pt", b"x")
    with pytest.raises(ValueError, match="run code"):
        manyseg.load(bad)


def test_plain_torch_export_file_is_a_clear_error(float_and_int8, tmp_path):
    _, q = float_and_int8
    p = tmp_path / "plain.pt2"
    torch.export.save(q, p)
    with pytest.raises(ValueError, match="source=float_model"):
        manyseg.load(p)


def test_save_errors(float_and_int8, tmp_path):
    model, q = float_and_int8
    with pytest.raises(ValueError, match=r"\.pt2"):
        manyseg.save(q, tmp_path / "q.safetensors", source=model, **_args(model))
    with pytest.raises(ValueError, match="not quantised"):
        manyseg.save(export_model(model, torch.randn(1, 3, 64, 96)), tmp_path / "f.pt2",
                     source=model, **_args(model))
    other = create_model(backbone="resnet18", pretrained=False, num_classes=7).eval()
    with pytest.raises(ValueError, match="source="):
        manyseg.save(q, tmp_path / "o.pt2", source=other, **_args(other))
    assert not list(tmp_path.glob("*.pt2"))


def test_load_program_without_torchao_names_the_extra(saved, monkeypatch):
    monkeypatch.setitem(sys.modules, "torchao", None)
    assert manyseg.read_metadata(saved)["quantization"]["route"] == "pt2e"
    with pytest.raises(ImportError, match=r"manyseg\[quant\]"):
        manyseg.load(saved)


def _crafted_expression(marker):
    done = []

    def change(n, data):
        if not n.endswith("models/model.json"):
            return data

        def swap(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "expr_str" and not done:
                        node[k] = f"__import__('pathlib').Path({str(marker)!r}).touch() or 1"
                        done.append(1)
                    else:
                        swap(v)
            elif isinstance(node, list):
                for v in node:
                    swap(v)

        cfg = json.loads(data)
        swap(cfg)
        assert done
        return json.dumps(cfg)

    return change


def test_load_refuses_code_in_graph_expressions_without_running_it(saved, tmp_path):
    """sympify evaluates these strings, so a file with no pickle at all could run code."""
    marker = tmp_path / "ran"
    with pytest.raises(ValueError, match="run code"):
        manyseg.load(_rewrite(saved, tmp_path / "bad.pt2", _crafted_expression(marker)))
    assert not marker.exists()


@pytest.mark.parametrize("expr,ok", [
    ("Symbol('s53', positive=True, integer=True)", True),
    ("Add(Symbol('s0', positive=True, integer=True), Integer(1))", True),
    ("Max(Integer(1), FloorDiv(Symbol('s1', integer=True), Integer(32)))", True),
    ("StrictLessThan(Integer(-1), Symbol('s1'))", True),
    ("s0 + 1", True),
    ("__import__('os')", False),
    ("eval('1')", False),
    ("open(0)", False),
    ("(open)(0)", False),
    ("Integer(1).__class__", False),
    ("Integer.__mro__", False),
    ("Symbol('a').func", False),
    ("Symbol('a; b')", False),
    ("Integer(1)[0]", False),
    ("lambda: 1", False),
])
def test_expression_allowlist(expr, ok):
    assert _safe_expression(expr) is ok


def test_unserialisable_dynamic_program_is_a_clear_error(tmp_path):
    """vit: dynamic H/W use math.ceil, which torch.export.save can't write."""
    model = create_model(backbone="vit_tiny_patch16_224", pretrained=False, num_classes=3).eval()
    ep = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(1))
    with pytest.raises(ValueError, match="dynamic=False"):
        manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model))
    assert not (tmp_path / "q.pt2").exists()


def test_fixed_size_keeps_a_multiple_of_pad_multiple(tmp_path):
    """ManyPred rounds patches down to pad_multiple, which must not change a fixed size."""
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    x = torch.randn(1, 3, 97, 97)
    ep = quantize_pt2e(model, x, [x], dynamic=False)
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    assert loaded.fixed_size == (97, 97) and loaded.tiling_spec()["pad_multiple"] == 1


def test_wrong_size_error_states_both_sizes(saved, tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    x = torch.randn(1, 3, 64, 64)
    ep = quantize_pt2e(model, x, [x], dynamic=False)
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    with pytest.raises(ValueError, match=r"\[1, 3, 64, 64\].*\[1, 3, 32, 64\]"):
        loaded(torch.randn(1, 3, 32, 64))


def test_program_device_attribute_and_mismatch_error(saved):
    loaded = manyseg.load(saved)
    assert loaded.device == torch.device("cpu")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="map_location"):
            loaded(torch.randn(1, 3, 64, 96, device="cuda"))


def test_spec_passes_manypred_validator(saved):
    spec_mod = pytest.importorskip("manypred.spec")
    spec_mod.validate_tiling_spec(manyseg.load(saved).tiling_spec())


def test_non_square_fixed_program_saves_and_runs(tmp_path):
    """ManyPred's patches are square, so it can't tile this one, but it stays usable directly."""
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    x = torch.randn(1, 3, 64, 96)
    ep = quantize_pt2e(model, x, [x], dynamic=False)
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    assert loaded.fixed_size == (64, 96)
    assert torch.equal(ep.module()(x), loaded(x))


def test_fixed_batch_above_one(tmp_path):
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    x = torch.randn(3, 3, 64, 96)
    ep = quantize_pt2e(model, x, [x])
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    assert loaded.batch_size == 3
    assert loaded(torch.randn(3, 3, 65, 97)).shape == (3, 2, 65, 97)
    with pytest.raises(ValueError, match="batch size 3"):
        loaded(torch.randn(1, 3, 65, 97))


def _edit_json(suffix, edit):
    def change(n, data):
        if not n.endswith(suffix):
            return data
        cfg = json.loads(data)
        edit(cfg)
        return json.dumps(cfg)

    return change


def _first_target(node, value):
    """Set the first graph node's target."""
    if isinstance(node, dict):
        if isinstance(node.get("target"), str):
            node["target"] = value
            return True
        return any(_first_target(v, value) for v in node.values())
    return isinstance(node, list) and any(_first_target(v, value) for v in node)


@pytest.mark.parametrize("target", ["#ns:op", "torch.load", "builtins.eval", "_operator.call",
                                    "torch.ops.__class__.x"])
def test_load_refuses_unknown_graph_operators(saved, tmp_path, target):
    bad = _rewrite(saved, tmp_path / "bad.pt2",
                   _edit_json("models/model.json", lambda cfg: _first_target(cfg, target)))
    with pytest.raises(ValueError, match="graph operator"):
        manyseg.load(bad)


def test_load_refuses_oversized_empty_entry(saved, tmp_path):
    """torch allocates what a zero-byte entry declares, without reading anything."""
    def change(n, data):
        if n.endswith("model_weights_config.json"):
            cfg = json.loads(data)
            entry = next(iter(cfg["config"].values()))
            entry["tensor_meta"]["sizes"] = [{"as_int": 2**40}]
            first["path"] = entry["path_name"]
            return json.dumps(cfg)
        return data

    first = {}
    bad = _rewrite(saved, tmp_path / "bad.pt2", change)
    with zipfile.ZipFile(bad) as z, zipfile.ZipFile(tmp_path / "bad2.pt2", "w") as out:
        for n in z.namelist():
            out.writestr(n, b"" if n.endswith(f"/{first['path']}") else z.read(n))
    with pytest.raises(ValueError, match="declares more"):
        manyseg.load(tmp_path / "bad2.pt2")


def test_load_refuses_entry_on_another_device(saved, tmp_path):
    def edit(cfg):
        next(iter(cfg["config"].values()))["tensor_meta"]["device"]["type"] = "cuda"

    with pytest.raises(ValueError, match="on cuda"):
        manyseg.load(_rewrite(saved, tmp_path / "bad.pt2",
                              _edit_json("model_weights_config.json", edit)))


def test_save_refuses_source_of_another_architecture(float_and_int8, tmp_path):
    """Same output shape, so only the weights can tell."""
    _, q = float_and_int8
    other = create_model(backbone="mobilenetv3_small_100", pretrained=False,
                         num_classes=3).eval()
    with pytest.raises(ValueError, match="no weights of shape"):
        manyseg.save(q, tmp_path / "o.pt2", source=other, **_args(other))
    assert not (tmp_path / "o.pt2").exists()


def test_save_warns_on_source_with_other_weights(float_and_int8, tmp_path):
    _, q = float_and_int8
    torch.manual_seed(99)
    other = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    with pytest.warns(UserWarning, match="correlate"):
        manyseg.save(q, tmp_path / "o.pt2", source=other, **_args(other))


@pytest.mark.parametrize("backbone,decoder,dynamic,hw", [
    ("resnet18", "segformer", True, (64, 96)),
    ("resnet18", "upernet", True, (64, 96)),
    ("convnext_atto", "unet", True, (64, 96)),
    ("mobilenetv3_small_100", "unet", True, (64, 96)),
    ("caformer_s18", "unet", True, (64, 96)),
    ("pvt_v2_b0", "unet", True, (64, 96)),
    ("vit_tiny_patch16_224", "unet", False, (224, 224)),
])
def test_save_load_across_architectures_passes_the_archive_check(tmp_path, backbone, decoder,
                                                                 dynamic, hw):
    """The check is an allowlist, so real programs must get through it."""
    torch.manual_seed(0)
    model = create_model(backbone=backbone, decoder=decoder, pretrained=False,
                         num_classes=3).eval()
    x = torch.randn(1, 3, *hw)
    ep = quantize_pt2e(model, x, [x], dynamic=dynamic)
    loaded = manyseg.load(manyseg.save(ep, tmp_path / "q.pt2", source=model, **_args(model)))
    assert torch.equal(ep.module()(x), loaded(x))
