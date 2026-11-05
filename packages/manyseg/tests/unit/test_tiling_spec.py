"""DenseModel.tiling_spec(): plain data describing inputs, outputs and size constraints for
tiled inference (ManyPred)."""

import json

import manyseg
import pytest
from manyseg import create_model


def build(**kw):
    return create_model(backbone="resnet18", pretrained=False, **kw).eval()


def test_single_input_model():
    spec = build(in_channels=3, num_classes=4).tiling_spec()
    assert spec["inputs"] == {}
    assert spec["outputs"] == {"logits": {"scope": "pixel", "kind": "segmentation", "scale": 1.0}}
    assert spec["pad_multiple"] == 32 and spec["max_size"] is None and spec["normalize"] is None
    json.dumps(spec)  # plain data


def test_heads_scopes_and_scales():
    spec = build(in_channels=3, heads={
        "seg": {"type": "segmentation", "out_channels": 3},
        "height": {"type": "regression", "out_channels": 1, "scale": 0.25},
        "scene": {"type": "classification", "out_channels": 5, "scope": "image"},
    }).tiling_spec()
    assert spec["outputs"]["seg"] == {"scope": "pixel", "kind": "segmentation", "scale": 1.0}
    assert spec["outputs"]["height"]["scale"] == 0.25
    assert spec["outputs"]["scene"] == {"scope": "image", "kind": "classification"}


def test_multi_input_alignment_time_series_and_vectors():
    spec = build(
        inputs={
            "s2": {"in_channels": 4, "backbone": "resnet18"},
            "s1": {"in_channels": 2, "backbone": "resnet18", "alignment": "same_extent"},
            "ts": {"kind": "timeseries", "in_channels": 3, "backbone": "resnet18"},
            "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}},
        },
        fusion={"type": "intermediate", "op": "concat"},
        num_classes=2,
    ).tiling_spec()
    assert spec["inputs"] == {
        "s2": {"kind": "raster", "scale": 1.0},
        "s1": {"kind": "raster", "scale": None},
        "ts": {"kind": "timeseries", "scale": 1.0},
        "meta": {"kind": "vector"},
    }
    assert list(spec["inputs"])[0] == "s2"  # the primary input comes first
    json.dumps(spec)


def test_single_output_lists_only_the_returned_head():
    model = build(in_channels=3, heads={
        "mask": {"type": "segmentation", "out_channels": 2},
        "aux": {"type": "regression", "out_channels": 1, "scale": 0.25},
    })
    model.single_output = "mask"  # as num_classes= models are built
    assert list(model.tiling_spec()["outputs"]) == ["mask"]


def test_pad_to_stride_false_needs_no_padding_multiple():
    model = build(in_channels=3, num_classes=2, pad_to_stride=False)
    assert model.tiling_spec()["pad_multiple"] == 1


def test_single_time_series_input():
    spec = build(inputs={"ts": {"kind": "timeseries", "in_channels": 3}},
                 num_classes=2).tiling_spec()
    assert spec["inputs"] == {"ts": {"kind": "timeseries", "scale": 1.0}}


def test_fixed_size_backbone():
    model = create_model(backbone="hiera_tiny_224", pretrained=False, in_channels=3,
                         num_classes=2, backbone_kwargs={"img_size": (128, 192)})
    assert model.tiling_spec()["max_size"] == (128, 192)


SOFTMAX = {"activations": {"logits": "softmax"}, "normalize": []}


def test_no_names_stored():
    model = build(in_channels=3, num_classes=2)
    assert model.tiling_spec()["bands"] is None
    assert "classes" not in model.tiling_spec()["outputs"]["logits"]


def test_single_tensor_model_keeps_its_names(tmp_path):
    model = build(in_channels=3, num_classes=2)
    path = manyseg.save(model, tmp_path / "m.safetensors", **SOFTMAX, classes=["bg", "water"],
                        bands=["r", "g", "b"])
    spec = manyseg.load(path).tiling_spec()
    assert spec["outputs"]["logits"]["classes"] == ["bg", "water"]
    assert spec["bands"] == ["r", "g", "b"]
    json.dumps(spec)


def test_names_only_on_returned_heads(tmp_path):
    model = build(in_channels=3, heads={
        "mask": {"type": "segmentation", "out_channels": 2},
        "aux": {"type": "segmentation", "out_channels": 3}})
    path = manyseg.save(model, tmp_path / "m.safetensors", normalize=[],
                        activations={"mask": "softmax", "aux": "softmax"},
                        classes={"mask": ["a", "b"], "aux": ["x", "y", "z"]})
    loaded = manyseg.load(path)
    assert loaded.tiling_spec()["outputs"]["aux"]["classes"] == ["x", "y", "z"]
    loaded.select_heads(["mask"])
    outputs = loaded.tiling_spec()["outputs"]
    assert list(outputs) == ["mask"] and outputs["mask"]["classes"] == ["a", "b"]


def test_list_of_bands_for_one_named_input_is_keyed_by_it(tmp_path):
    model = build(num_classes=2, inputs={
        "img": {"in_channels": 2}, "doy": {"kind": "vector", "num_features": 1}})
    path = manyseg.save(model, tmp_path / "m.safetensors", **SOFTMAX, bands=["a", "b"])
    assert manyseg.load(path).tiling_spec()["bands"] == {"img": ["a", "b"]}


def test_one_channel_head_may_name_both_classes(tmp_path):
    model = build(in_channels=3, num_classes=1)
    path = manyseg.save(model, tmp_path / "m.safetensors", normalize=[],
                        activations={"logits": "sigmoid"}, classes=["bg", "water"])
    assert manyseg.load(path).tiling_spec()["outputs"]["logits"]["classes"] == ["bg", "water"]


@pytest.mark.parametrize("names", [{"classes": [0, 1]}, {"classes": {"logits": "ab"}},
                                   {"bands": [1, 2, 3]}, {"bands": "rgb"}])
def test_save_refuses_names_that_are_not_strings(tmp_path, names):
    model = build(in_channels=3, num_classes=2)
    with pytest.raises(ValueError):
        manyseg.save(model, tmp_path / "m.safetensors", **SOFTMAX, **names)


def test_names_stored_before_they_were_checked_stay_valid():
    model = build(in_channels=3, num_classes=2)
    model.saved_metadata = {"classes": [0, 1], "bands": [1, 2, 3]}
    spec = model.tiling_spec()
    assert spec["outputs"]["logits"]["classes"] == ["0", "1"] and spec["bands"] == ["1", "2", "3"]
