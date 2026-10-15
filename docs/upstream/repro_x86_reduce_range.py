"""PT2E x86 int8 + torch.compile (freezing) silently diverges from its own eager reference on
CPUs without VNNI unless reduce_range=True. Standalone: torch, torchao, timm.

    python repro_x86_reduce_range.py

Observed (torch 2.14.1, torchao 0.18.0, timm 1.0.30, AMD Ryzen 9 5950X = Zen 3, AVX2, no
VNNI): with the default reduce_range=False the compiled stage-4 ResNet-34 features differ from
the eager quantise/dequantise reference by ~50% (mean relative difference); with
reduce_range=True by ~6%. Expected: the compiled int8 graph should match its eager reference
(or the quantizer should default to / require reduce_range on such CPUs).
"""

import timm
import torch
import torch.nn.functional as F
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import x86_inductor_quantizer as xiq

torch.manual_seed(0)
torch.set_num_threads(8)
print("torch", torch.__version__, "| VNNI:", torch.cpu._is_vnni_supported(),
      "| capability:", torch.backends.cpu.get_cpu_capability())

# Stock ImageNet ResNet-34 trunk (any CNN backbone with real weights shows it).
model = timm.create_model("resnet34.tv_in1k", pretrained=True, features_only=True,
                          out_indices=(4,)).eval()

# Smooth, natural-image-like inputs (low-frequency structure), ImageNet-normalised. Pure white
# noise shows a smaller effect (~12% vs ~5%).
x = F.interpolate(torch.rand(32, 3, 14, 14), size=(224, 224), mode="bilinear")
x = (x - 0.45) / 0.226

for reduce_range in (False, True):
    q = xiq.X86InductorQuantizer().set_global(
        xiq.get_default_x86_inductor_quantization_config(reduce_range=reduce_range))
    prepared = prepare_pt2e(torch.export.export(model, (x,)).module(), q)
    prepared(x)  # calibration
    converted = convert_pt2e(prepared)
    with torch.no_grad():
        eager = converted(x)[0]  # quantise/dequantise reference
        torch._dynamo.reset()
        torch._inductor.config.freezing = True
        compiled = torch.compile(converted)(x)[0]  # x86 inductor int8 kernels
    rel = float((compiled - eager).abs().mean() / eager.abs().mean())
    print(f"reduce_range={reduce_range!s:5}: compiled vs eager mean relative difference {rel:.3f}")
