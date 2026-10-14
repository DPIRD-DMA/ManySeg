"""Metadata-conditioned models under torch.export / ONNX / torch.compile. Marked slow."""

import pytest
import torch
from manyseg import create_model
from manyseg.export import dynamic_shapes, export_model

pytestmark = pytest.mark.slow

FIELDS = {
    "doy": {"type": "cyclic", "period": 365.25},
    "elev": {"type": "continuous", "mean": 40.0, "std": 10.0},
    "latlon": {"type": "coords", "frequencies": 2},
    "sensor": {"type": "categorical", "num": 3},
}
HEADS = {"seg": {"type": "segmentation", "channels": 2},
         "area": {"type": "regression", "channels": 1, "scope": "image"}}
MODES = {"film": "film", "head": {"mode": "head", "pixel": True}, "channels": "channels"}


def meta(n=1, seed=0, missing=False):
    g = torch.Generator().manual_seed(seed)
    m = torch.stack([torch.rand(n, generator=g) * 365, 40 + 10 * torch.randn(n, generator=g),
                     torch.rand(n, generator=g) * 180 - 90, torch.rand(n, generator=g) * 360 - 180,
                     torch.randint(0, 3, (n,), generator=g).float()], dim=1)
    if missing:
        m[:, 1] = float("nan")
    return m


def build(mode, rasters=None, **kw):
    rasters = rasters or {"img": {"channels": 3}}
    model = create_model(backbone="resnet18", pretrained=False, heads=HEADS,
                         inputs={**rasters, "meta": {"kind": "vector", "fields": FIELDS,
                                                     "conditioning": MODES[mode]}}, **kw)
    with torch.no_grad():  # non-identity injection so the export must carry the metadata path
        for n, p in model.conditioner.named_parameters():
            if n.startswith(("film", "head_bias")):
                p.normal_(0, 0.3)
    return model.eval()


def check(fn, model, xs):
    for x in xs:
        got, ref = fn(x), model(x)
        for k in ref:
            torch.testing.assert_close(got[k], ref[k], atol=1e-4, rtol=1e-4)


def test_dynamic_shapes_give_vectors_no_spatial_dims():
    model = build("film")
    ex = {"img": torch.randn(1, 3, 64, 96), "meta": meta(), "meta_mask": torch.ones(1, 4)}
    spec = dynamic_shapes(model, ex)["x"]
    assert spec["meta"] is None and spec["meta_mask"] is None and set(spec["img"]) == {2, 3}
    spec = dynamic_shapes(model, ex, dynamic_batch=True)["x"]
    assert spec["meta"] == {0: spec["img"][0]}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_export_dynamic_hw_with_vector_input(mode):
    model = build(mode)
    ep = export_model(model, {"img": torch.randn(1, 3, 64, 96), "meta": meta()})
    check(ep.module(), model, [{"img": torch.randn(1, 3, *hw), "meta": meta(seed=s, missing=s)}
                               for s, hw in enumerate([(97, 131), (200, 65)])])


def test_export_with_mask_dynamic_batch_and_fusion():
    model = build("film", rasters={"a": {"channels": 4}, "b": {"channels": 2}})
    ex = {"a": torch.randn(2, 4, 64, 96), "b": torch.randn(2, 2, 64, 96), "meta": meta(2),
          "meta_mask": torch.ones(2, 4)}
    ep = export_model(model, ex, dynamic_batch=True)
    xs = []
    for n, hw in [(1, (97, 131)), (3, (64, 64))]:
        mask = torch.ones(n, 4)
        mask[0, 2] = 0
        xs.append({"a": torch.randn(n, 4, *hw), "b": torch.randn(n, 2, *hw),
                   "meta": meta(n, seed=n), "meta_mask": mask})
    check(ep.module(), model, xs)


def test_onnx_with_vector_input_matches_onnxruntime(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = build("film")
    ex = {"img": torch.randn(1, 3, 64, 96), "meta": meta()}
    path = tmp_path / "m.onnx"
    # a trailing dict in ``args`` is read as keyword arguments by torch.onnx.export
    torch.onnx.export(model, (), str(path), kwargs={"x": ex}, dynamo=True,
                      dynamic_shapes=dynamic_shapes(model, ex))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    assert len(names) == 2
    for s, hw in enumerate([(64, 96), (97, 131)]):
        x = {"img": torch.randn(1, 3, *hw), "meta": meta(seed=s, missing=s)}
        outs = sess.run(None, dict(zip(names, (x["img"].numpy(), x["meta"].numpy()))))
        ref = model(x)
        for got, k in zip(outs, ref):  # outputs in head order
            torch.testing.assert_close(torch.from_numpy(got), ref[k], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_compile_fullgraph_dynamic(mode):
    model = build(mode, decoder="fpn")
    compiled = torch.compile(model, fullgraph=True, dynamic=True, backend="aot_eager")
    with torch.no_grad():
        check(compiled, model, [{"img": torch.randn(1, 3, *hw), "meta": meta(seed=s, missing=s)}
                                for s, hw in enumerate([(64, 96), (97, 131)])])


def test_compile_training_step_with_field_dropout():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"channels": 3},
        "meta": {"kind": "vector", "fields": FIELDS, "dropout": 0.3}}).train()
    compiled = torch.compile(model, fullgraph=True, dynamic=True, backend="aot_eager")
    compiled({"img": torch.randn(2, 3, 64, 64), "meta": meta(2)}).mean().backward()
    assert all(p.grad is not None for p in model.conditioner.parameters())
