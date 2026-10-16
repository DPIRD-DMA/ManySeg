"""DenseModel.tiling_spec(): plain data describing inputs, outputs and size constraints for
tiled inference (ManyPred)."""

import json

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
        "seg": {"type": "segmentation", "channels": 3},
        "height": {"type": "regression", "channels": 1, "scale": 0.25},
        "scene": {"type": "classification", "channels": 5, "scope": "image"},
    }).tiling_spec()
    assert spec["outputs"]["seg"] == {"scope": "pixel", "kind": "segmentation", "scale": 1.0}
    assert spec["outputs"]["height"]["scale"] == 0.25
    assert spec["outputs"]["scene"] == {"scope": "image", "kind": "classification"}


def test_multi_input_alignment_time_series_and_vectors():
    spec = build(
        inputs={
            "s2": {"channels": 4, "backbone": "resnet18"},
            "s1": {"channels": 2, "backbone": "resnet18", "alignment": "same_extent"},
            "ts": {"kind": "timeseries", "channels": 3, "backbone": "resnet18"},
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
        "mask": {"type": "segmentation", "channels": 2},
        "aux": {"type": "regression", "channels": 1, "scale": 0.25},
    })
    model.single_output = "mask"  # as num_classes= models are built
    assert list(model.tiling_spec()["outputs"]) == ["mask"]


def test_pad_to_stride_false_needs_no_padding_multiple():
    model = build(in_channels=3, num_classes=2, pad_to_stride=False)
    assert model.tiling_spec()["pad_multiple"] == 1


def test_single_time_series_input():
    spec = build(inputs={"ts": {"kind": "timeseries", "channels": 3}}, num_classes=2).tiling_spec()
    assert spec["inputs"] == {"ts": {"kind": "timeseries", "scale": 1.0}}


def test_fixed_size_backbone():
    model = create_model(backbone="hiera_tiny_224", pretrained=False, in_channels=3,
                         num_classes=2, backbone_kwargs={"img_size": (128, 192)})
    assert model.tiling_spec()["max_size"] == (128, 192)
