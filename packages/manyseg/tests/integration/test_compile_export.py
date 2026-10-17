"""torch.compile / torch.export / ONNX contracts. Marked slow."""

import pytest
import torch
from manyseg import create_model

pytestmark = pytest.mark.slow


def model_(backbone="resnet18", **kw):
    return create_model(backbone=backbone, pretrained=False, num_classes=3, **kw).eval()


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_atto", "efficientnet_b0"])
def test_compile_fullgraph_matches_eager(backbone, device):
    model = model_(backbone).to(device)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    x = torch.randn(1, 3, 65, 97, device=device)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x), atol=2e-3, rtol=2e-3)


SIZES10 = [(64, 96), (96, 128), (65, 97), (160, 33), (200, 129), (257, 301), (123, 77),
           (320, 320), (64, 64), (511, 129)]


def test_dynamic_compile_graph_count(device):
    """One graph for many sizes on CUDA.

    On CPU PyTorch's fake conv kernel guards on the size-dependent oneDNN/native backend choice
    (torch._subclasses.fake_impls, `_select_conv_backend`), which adds a bounded number of graphs
    for small inputs. That is outside ManySeg; we only require it stays bounded.
    """
    import torch._dynamo as dynamo

    model = model_().to(device)
    dynamo.reset()
    calls = {"n": 0}

    def backend(gm, example_inputs):
        calls["n"] += 1
        return gm.forward

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
    with torch.no_grad():
        for hw in SIZES10:
            assert compiled(torch.randn(1, 3, *hw, device=device)).shape[-2:] == hw
    limit = 1 if device.type == "cuda" else 8
    assert calls["n"] <= limit, calls["n"]


# Dynamic export range starts at 33: at <= 32 px the stride-32 feature map is 1x1 and
# torch.export specialises size-1 dimensions.
def test_torch_export_dynamic_hw():
    model = model_()
    h = torch.export.Dim("h", min=33, max=2048)
    w = torch.export.Dim("w", min=33, max=2048)
    x = torch.randn(1, 3, 64, 96)
    ep = torch.export.export(model, (x,), dynamic_shapes={"x": {2: h, 3: w}})
    for hw in [(64, 96), (65, 97), (128, 33)]:
        xi = torch.randn(1, 3, *hw)
        torch.testing.assert_close(ep.module()(xi), model(xi), atol=1e-4, rtol=1e-4)


def test_torch_export_multihead():
    model = create_model(
        backbone="resnet18", pretrained=False,
        heads={"seg": {"type": "segmentation", "out_channels": 2},
               "cls": {"type": "classification", "out_channels": 4, "scope": "image"}},
    ).eval()
    ep = torch.export.export(model, (torch.randn(1, 3, 64, 64),))
    out = ep.module()(torch.randn(1, 3, 64, 64))
    assert out["seg"].shape == (1, 2, 64, 64) and out["cls"].shape == (1, 4)


def test_onnx_export_dynamic_matches_onnxruntime(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = model_()
    path = tmp_path / "m.onnx"
    h = torch.export.Dim("h", min=33, max=2048)
    w = torch.export.Dim("w", min=33, max=2048)
    torch.onnx.export(
        model, (torch.randn(1, 3, 64, 96),), str(path), dynamo=True,
        dynamic_shapes={"x": {2: h, 3: w}}, input_names=["x"], output_names=["logits"],
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for hw in [(64, 96), (65, 97)]:
        x = torch.randn(1, 3, *hw)
        (got,) = sess.run(None, {"x": x.numpy()})
        torch.testing.assert_close(torch.from_numpy(got), model(x), atol=1e-3, rtol=1e-3)


_OLD_TORCH = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) < (2, 11)


@pytest.mark.parametrize("backbone", [
    "vit_tiny_patch16_224", "eva02_tiny_patch14_224",
    # Swin's dynamic fullgraph compile fails on torch 2.9 (any timm); passes from 2.11.
    pytest.param("swin_tiny_patch4_window7_224", marks=pytest.mark.xfail(
        _OLD_TORCH, reason="Swin dynamic fullgraph compile needs torch >= 2.11", strict=False)),
    "pvt_v2_b0"])
def test_transformer_compile_fullgraph_dynamic(backbone):
    model = model_(backbone)
    compiled = torch.compile(model, fullgraph=True, dynamic=True, backend="aot_eager")
    with torch.no_grad():
        for hw in [(64, 96), (97, 131)]:
            x = torch.randn(1, 3, *hw)
            torch.testing.assert_close(compiled(x), model(x), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("backbone", ["vit_tiny_patch16_224", "eva02_tiny_patch14_224",
                                      "pvt_v2_b0"])
def test_transformer_export_dynamic(backbone):
    model = model_(backbone)
    h = torch.export.Dim("h", min=33, max=1024)
    w = torch.export.Dim("w", min=33, max=1024)
    ep = torch.export.export(model, (torch.randn(1, 3, 64, 96),),
                             dynamic_shapes={"x": {2: h, 3: w}})
    for hw in [(64, 96), (97, 131)]:
        x = torch.randn(1, 3, *hw)
        torch.testing.assert_close(ep.module()(x), model(x), atol=1e-4, rtol=1e-4)


def test_swin_dynamic_export_known_limitation():
    """timm Swin with dynamic H/W hangs in torch.export (see architecture_review §2.4).
    Probe in a subprocess with a timeout; if it starts working, this test tells us."""
    import subprocess
    import sys

    code = ("import torch; from manyseg import create_model; "
            "m=create_model(backbone='swin_tiny_patch4_window7_224',pretrained=False,"
            "num_classes=2).eval(); h=torch.export.Dim('h',min=33,max=1024); "
            "w=torch.export.Dim('w',min=33,max=1024); "
            "torch.export.export(m,(torch.randn(1,3,64,96),),dynamic_shapes={'x':{2:h,3:w}})")
    try:
        r = subprocess.run([sys.executable, "-c", code], timeout=120, capture_output=True)
    except subprocess.TimeoutExpired:
        pytest.xfail("swin dynamic export still hangs (known timm/torch.export limitation)")
    if r.returncode != 0:
        pytest.xfail("swin dynamic export fails: " + r.stderr.decode()[-300:])


@pytest.mark.parametrize("decoder", ["fpn", "deeplabv3plus", "upernet", "segformer",
                                     "fastai_unet", "unetplusplus"])
def test_decoder_export_and_compile_dynamic(decoder):
    model = model_(decoder=decoder)
    h = torch.export.Dim("h", min=33, max=1024)
    w = torch.export.Dim("w", min=33, max=1024)
    ep = torch.export.export(model, (torch.randn(1, 3, 64, 96),),
                             dynamic_shapes={"x": {2: h, 3: w}})
    compiled = torch.compile(model, fullgraph=True, dynamic=True, backend="aot_eager")
    with torch.no_grad():
        for hw in [(64, 96), (97, 131)]:
            x = torch.randn(1, 3, *hw)
            ref = model(x)
            torch.testing.assert_close(ep.module()(x), ref, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(compiled(x), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("ftype", ["early", "intermediate", "late"])
def test_multi_input_export_dynamic(ftype):
    inputs = {"optical": {"in_channels": 4, "backbone": "resnet18"},
              "sar": {"in_channels": 2, "backbone": "resnet18"}}
    model = create_model(inputs=inputs, fusion={"type": ftype}, backbone="resnet18",
                         pretrained=False, num_classes=2).eval()
    h = torch.export.Dim("h", min=33, max=1024)
    w = torch.export.Dim("w", min=33, max=1024)
    x = {"optical": torch.randn(1, 4, 64, 96), "sar": torch.randn(1, 2, 64, 96)}
    ep = torch.export.export(model, (x,), dynamic_shapes={
        "x": {"optical": {2: h, 3: w}, "sar": {2: h, 3: w}}})
    xi = {"optical": torch.randn(1, 4, 97, 131), "sar": torch.randn(1, 2, 97, 131)}
    torch.testing.assert_close(ep.module()(xi), model(xi), atol=1e-4, rtol=1e-4)
