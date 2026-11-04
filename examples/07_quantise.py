"""int8 quantisation with `manyseg.quantize` (needs torchao: pip install 'manyseg[quant]').

* `quantize_pt2e`: static int8 weights and activations for CPU, calibrated on a few batches,
  returned as an ExportedProgram with dynamic H/W.
* `quantize_weights`: int8 weights on the eager module (CPU or GPU), no calibration.
Random weights give near-flat logits, so the agreement printed here is pessimistic: measure
`compare_outputs` for a trained model on your own validation data.
"""

import torch
from manyseg import create_model
from manyseg.export import export_model
from manyseg.quantize import compare_outputs, model_size, quantize_pt2e, quantize_weights


def main():
    try:
        import torchao  # noqa: F401
    except ImportError:
        print("torchao is not installed: skipping (pip install 'manyseg[quant]')")
        return
    torch.manual_seed(0)
    model = create_model(backbone="resnet18", pretrained=False,  # True downloads weights
                         in_channels=4, num_classes=3).eval()
    example = torch.randn(1, 4, 64, 64)
    # Use real, representative tiles. The batch size must match `example` unless
    # quantize_pt2e(..., dynamic_batch=True); H/W may differ.
    calibration = [torch.randn(1, 4, 64, 96) for _ in range(4)]
    test = [torch.randn(1, 4, 97, 131) for _ in range(2)]        # dynamic H/W: any size works

    mb = 1 / 2**20
    # Static int8 (PT2E): export -> prepare -> calibrate -> convert -> export.
    ep = quantize_pt2e(model, example, calibration)
    float_ep = export_model(model, example)
    print(f"pt2e int8:   {model_size(ep) * mb:.1f} MB vs {model_size(float_ep) * mb:.1f} MB "
          "exported float")
    print("  ", {k: round(v, 3) for k, v in compare_outputs(model, ep, test).items()})
    # manyseg.save(ep, "model_int8.pt2", source=model, normalize=..., activations=...) keeps the
    # metadata; manyseg.load gives a callable with tiling_spec(), not an nn.Module.
    # Eager ep.module() simulates int8 in float and is slower than float; for speed compile it
    # (and check compare_outputs on that module):
    #     torch._inductor.config.freezing = True
    #     fast = torch.compile(ep.module())

    # int8 weight-only on the eager module.
    q = quantize_weights(model)  # or "int8_dynamic" (also int8 activations in Linear layers)
    print(f"weight-only: {model_size(q) * mb:.1f} MB vs {model_size(model) * mb:.1f} MB "
          "float state_dict")
    print("  ", {k: round(v, 3) for k, v in compare_outputs(model, q, test).items()})
    # manyseg.save(q, "model_int8.safetensors", normalize=..., activations=...) keeps the size
    # saving and the metadata; manyseg.load gives an nn.Module with int8 weights.


if __name__ == "__main__":
    main()
