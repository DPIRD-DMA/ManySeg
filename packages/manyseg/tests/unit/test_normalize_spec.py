"""manyseg.save refuses normalisation specs ManyPred could not run, at save time."""

import manyseg
import pytest
import torch
from manyseg import create_model
from manyseg._normalize_spec import check_normalize, check_steps

GOOD_STEPS = [
    [],
    {"type": "none"},
    {"type": "scale", "value": 10000},
    {"type": "scale", "value": [255, 255.0, 1e4]},
    [{"type": "scale", "value": 10000}, {"type": "zscore", "mean": [0.5, 0.4], "std": [0.2, 0.1]}],
    {"type": "zscore", "mean": 0.5, "std": 0.2},
    {"type": "zscore", "scope": "tile"},
    [{"type": "zscore", "scope": "scene"}, {"type": "zscore", "scope": "tile"}],
    {"type": "minmax", "min": 0, "max": 1},
    {"type": "minmax", "min": [0, 1], "max": 2, "clip": False},
    {"type": "percentile"},
    {"type": "percentile", "low": 1, "high": 99, "scope": "scene", "clip": True},
    {"type": "db"},
    {"type": "db", "eps": 1e-6},
    {"type": "clip", "min": 0},
    {"type": "clip", "max": [1, 2]},
    {"type": "clip", "min": 0.0, "max": 1.0},
]

BAD_STEPS = [
    [{"type": "scael", "value": 1}],  # unknown type
    [{"value": 1}],  # no type
    {},
    "scale",
    False,
    [None],
    [{"type": ["scale"], "value": 1}],
    {"type": "scale", "valeu": 1},  # unknown option
    {"type": "scale"},
    {"type": "scale", "value": 0},
    {"type": "scale", "value": [1, 0]},
    {"type": "scale", "value": "x"},
    {"type": "scale", "value": []},
    {"type": "scale", "value": float("nan")},
    {"type": "zscore"},
    {"type": "zscore", "mean": 0.5},
    {"type": "zscore", "std": 0.5},
    {"type": "zscore", "mean": 0.5, "std": 0.2, "scope": "tile"},
    {"type": "zscore", "scope": "image"},
    {"type": "zscore", "mean": 0.5, "std": 0},
    {"type": "zscore", "mean": 0.5, "std": -1},
    {"type": "minmax", "min": 0},
    {"type": "minmax", "max": 1},
    {"type": "minmax", "min": 1, "max": 1},
    {"type": "minmax", "min": [0, 0, 0], "max": [1, 1]},
    {"type": "minmax", "min": 0, "max": 1, "clip": 1},
    {"type": "percentile", "low": 98, "high": 2},
    {"type": "percentile", "high": 101},
    {"type": "db", "eps": 0},
    {"type": "clip"},
    {"type": "clip", "min": 2, "max": 1},
    {"type": "none", "value": 1},
    [{"type": "zscore", "scope": "tile"}, {"type": "zscore", "scope": "scene"}],  # scene first
]


@pytest.mark.parametrize("spec", GOOD_STEPS, ids=repr)
def test_good_steps_pass(spec):
    check_steps(spec)


@pytest.mark.parametrize("spec", BAD_STEPS, ids=repr)
def test_bad_steps_fail(spec):
    with pytest.raises(ValueError):
        check_steps(spec)


@pytest.mark.parametrize("spec", GOOD_STEPS + BAD_STEPS, ids=repr)
def test_agrees_with_manypred(spec):
    """ManySeg can't import ManyPred, so it keeps its own copy of the rules. Any drift fails."""
    validate_spec = pytest.importorskip("manypred.normalize").validate_spec

    def accepted(check):
        try:
            check(spec)
        except Exception:
            return False
        return True

    assert accepted(check_steps) == accepted(validate_spec)


def test_per_input_specs():
    good = {"a": {"type": "none"}, "b": [{"type": "scale", "value": 2}]}
    check_normalize(good, ["a", "b"])
    with pytest.raises(ValueError, match="unknown inputs.*'c'"):
        check_normalize({**good, "c": {"type": "none"}}, ["a", "b"])
    with pytest.raises(ValueError, match=r"'b'.*\{'type': 'none'\}"):  # a forgotten input
        check_normalize({"a": {"type": "none"}}, ["a", "b"])
    with pytest.raises(ValueError, match="scael"):
        check_normalize({"a": {"type": "none"}, "b": {"type": "scael"}}, ["a", "b"])
    for one_spec in ({"type": "none"}, [{"type": "scale", "value": 2}], []):
        with pytest.raises(ValueError, match="input_name"):
            check_normalize(one_spec, ["a", "b"])


def test_a_named_single_input_also_takes_a_dict_of_inputs():
    check_normalize({"a": {"type": "none"}}, ["a"])
    with pytest.raises(ValueError, match="input_name"):
        check_normalize({"type": "none"}, ["a"])


def test_an_unnamed_input_takes_steps_not_names():
    check_normalize([{"type": "scale", "value": 2}], None)
    with pytest.raises(ValueError, match="no 'type'"):
        check_normalize({"image": {"type": "none"}}, None)


KW = {"activations": {"logits": "softmax"}}


def test_save_refuses_a_bad_spec_and_writes_no_file(tmp_path):
    model = create_model("resnet18", pretrained=False, num_classes=2)
    path = tmp_path / "m.safetensors"
    with pytest.raises(ValueError, match="scael"):
        manyseg.save(model, path, normalize=[{"type": "scael", "value": 1}], **KW)
    assert not path.exists()


def test_save_checks_a_multi_input_model_against_its_rasters(tmp_path):
    model = create_model(
        "resnet18", pretrained=False, num_classes=2,
        inputs={"a": {"in_channels": 3}, "b": {"in_channels": 2},
                "meta": {"kind": "vector", "num_features": 2}})
    path = tmp_path / "m.safetensors"
    with pytest.raises(ValueError, match="input_name"):
        manyseg.save(model, path, normalize=[{"type": "scale", "value": 2}], **KW)
    with pytest.raises(ValueError, match="'b'"):
        manyseg.save(model, path, normalize={"a": {"type": "none"}}, **KW)
    with pytest.raises(ValueError, match="unknown inputs.*'meta'"):
        manyseg.save(model, path, normalize={"a": [], "b": [], "meta": []}, **KW)
    spec = {"a": {"type": "scale", "value": 2}, "b": {"type": "none"}}
    assert manyseg.read_metadata(manyseg.save(model, path, normalize=spec, **KW))["normalize"] \
        == spec


def test_load_stays_lenient_about_a_stored_spec(tmp_path, monkeypatch):
    """Files written before the check still open: it's only the save that refuses."""
    model = create_model("resnet18", pretrained=False, num_classes=2)
    monkeypatch.setattr("manyseg.io.check_normalize", lambda *a: None)
    bad = [{"type": "scael"}]
    path = manyseg.save(model, tmp_path / "m.safetensors", normalize=bad, **KW)
    monkeypatch.undo()
    assert manyseg.load(path).tiling_spec()["normalize"] == bad
    assert isinstance(manyseg.load(path)(torch.zeros(1, 3, 32, 32)), torch.Tensor)
