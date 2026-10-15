"""Deployment: torch.export with dynamic H/W, and ONNX checked with onnxruntime (skipped if
onnxscript / onnxruntime are not installed).

torch.compile needs no export: `torch.compile(model, dynamic=True)` gives one graph across
input sizes. For GPU inference, bf16 autocast + compile was 1.6x * 1.46x faster than eager
fp32 on an RTX 4090 (E014); the first call per shape compiles (8-15 s), so it is not run here:

    fast = torch.compile(model, dynamic=True)
    with torch.autocast("cuda", torch.bfloat16):
        fast(x)
"""

import importlib.util
import tempfile
from pathlib import Path

import torch

from manyseg import create_model
from manyseg.export import dynamic_shapes, export_model


def main():
    model = create_model(backbone="resnet18", pretrained=False,  # True downloads weights
                         in_channels=4, num_classes=3).eval()
    example = torch.randn(1, 4, 64, 96)
    out_dir = Path(tempfile.mkdtemp())

    # torch.export: H and W are dynamic (min = the model's divisor + 1, max 4096).
    ep = export_model(model, example)  # dynamic_batch=True also frees the batch dimension
    x = torch.randn(1, 4, 97, 131)     # a size not seen at export time
    with torch.no_grad():
        ref = model(x)
    torch.testing.assert_close(ep.module()(x), ref, atol=1e-4, rtol=1e-4)
    torch.export.save(ep, out_dir / "model.pt2")  # reload with torch.export.load(...)
    print("torch.export: ok at 97x131 ->", out_dir / "model.pt2")

    # ONNX through the dynamo exporter (needs onnxscript), with the same dynamic H/W spec.
    if importlib.util.find_spec("onnxscript") is None:
        print("onnxscript not installed: skipping ONNX (pip install 'manyseg[onnx]' onnxruntime)")
        return
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(model, (example,), str(onnx_path), dynamo=True, verbose=False,
                      dynamic_shapes=dynamic_shapes(model, example),
                      input_names=["x"], output_names=["logits"])
    if importlib.util.find_spec("onnxruntime") is None:
        print("onnxruntime not installed: skipping the ONNX check ->", onnx_path)
        return
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"x": x.numpy()})
    torch.testing.assert_close(torch.from_numpy(got), ref, atol=1e-3, rtol=1e-3)
    print("ONNX: onnxruntime matches eager at 97x131 ->", onnx_path)


if __name__ == "__main__":
    main()
