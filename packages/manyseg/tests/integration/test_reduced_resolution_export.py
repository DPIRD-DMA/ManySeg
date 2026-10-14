"""Reduced decoder out_stride and scaled heads under torch.compile, torch.export and ONNX.

Sizes stay symbolic: the head's output size is ``(h + f - 1) // f`` and cropping is negative
padding (``manyseg.utils.spatial``), so one exported program serves every input size.
"""

import pytest
import torch
from manyseg import create_model
from manyseg.export import export_model

pytestmark = pytest.mark.slow

SCALED = {"x1": {"type": "segmentation", "channels": 2},
          "x2": {"type": "segmentation", "channels": 2, "scale": 0.5},
          "x4": {"type": "regression", "channels": 1, "scale": 0.25},
          "x8": {"type": "regression", "channels": 1, "scale": 0.125}}

CONFIGS = {
    "unet_s2": dict(decoder="unet", decoder_kwargs={"out_stride": 2}, num_classes=3),
    "unet_s4_scaled": dict(decoder="unet", decoder_kwargs={"out_stride": 4}, heads=SCALED),
    "unetpp_s2": dict(decoder="unetplusplus", decoder_kwargs={"out_stride": 2}, num_classes=3),
    "fastai_s2": dict(decoder="fastai_unet", decoder_kwargs={"out_stride": 2}, num_classes=3),
    "fastai_s4_scaled": dict(decoder="fastai_unet", decoder_kwargs={"out_stride": 4},
                             heads=SCALED),
    "segformer_scaled": dict(decoder="segformer", heads=SCALED),
    "task_decoders": dict(heads={
        "seg": {"type": "segmentation", "channels": 2, "scale": 0.25},
        "edges": {"type": "segmentation", "channels": 1,
                  "decoder": {"type": "unet", "out_stride": 2}},
        "density": {"type": "regression", "channels": 1, "scale": 0.125,
                    "decoder": {"type": "fastai_unet", "out_stride": 4}}},
        decoder_kwargs={"out_stride": 4}),
}


def build(name):
    return create_model(backbone="resnet18", pretrained=False, **CONFIGS[name]).eval()


def assert_same(a, b, tol=1e-4):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b:
            assert a[k].shape == b[k].shape, k
            torch.testing.assert_close(a[k], b[k], atol=tol, rtol=tol)
    else:
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_export_dynamic_hw(name):
    model = build(name)
    ep = export_model(model, torch.randn(1, 3, 64, 96))
    for hw in [(64, 96), (97, 131), (200, 65)]:
        x = torch.randn(1, 3, *hw)
        assert_same(ep.module()(x), model(x))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_compile_fullgraph_dynamic(name):
    model = build(name)
    compiled = torch.compile(model, fullgraph=True, dynamic=True, backend="aot_eager")
    with torch.no_grad():
        for hw in [(64, 96), (97, 131)]:
            x = torch.randn(1, 3, *hw)
            assert_same(compiled(x), model(x))


@pytest.mark.parametrize("name", ["unet_s4_scaled", "fastai_s2"])
def test_compile_inductor_matches_eager(name):
    model = build(name)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    x = torch.randn(1, 3, 65, 97)
    with torch.no_grad():
        assert_same(compiled(x), model(x), tol=2e-3)


def test_export_multi_input_scaled_follows_primary_grid():
    inputs = {"optical": {"channels": 4}, "sar": {"channels": 2}}
    model = create_model(inputs=inputs, backbone="resnet18", pretrained=False, heads=SCALED,
                         decoder_kwargs={"out_stride": 2}).eval()
    ep = export_model(model, {"optical": torch.randn(1, 4, 64, 96),
                              "sar": torch.randn(1, 2, 64, 96)})
    x = {"optical": torch.randn(1, 4, 97, 131), "sar": torch.randn(1, 2, 97, 131)}
    out = ep.module()(x)
    assert out["x8"].shape[-2:] == (13, 17)
    assert_same(out, model(x))


@pytest.mark.parametrize("name", ["unet_s4_scaled", "fastai_s2"])
def test_onnx_dynamic_matches_onnxruntime(name, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = build(name)
    path = tmp_path / "m.onnx"
    h = torch.export.Dim("h", min=33, max=2048)
    w = torch.export.Dim("w", min=33, max=2048)
    torch.onnx.export(model, (torch.randn(1, 3, 64, 96),), str(path), dynamo=True,
                      dynamic_shapes={"x": {2: h, 3: w}}, input_names=["x"])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for hw in [(64, 96), (65, 97), (129, 47)]:
        x = torch.randn(1, 3, *hw)
        got = sess.run(None, {"x": x.numpy()})
        ref = model(x)
        refs = list(ref.values()) if isinstance(ref, dict) else [ref]
        assert len(got) == len(refs)
        for g, r in zip(got, refs):
            torch.testing.assert_close(torch.from_numpy(g), r, atol=1e-3, rtol=1e-3)
