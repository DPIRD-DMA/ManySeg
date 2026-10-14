# ManySeg

Composable dense-prediction models for PyTorch: any timm backbone, any channel count, multiple
inputs and outputs. Compile- and export-friendly, and checked against an independent
benchmark.

> Status: pre-alpha (Phases 1–4 in progress). APIs may change. See `docs/roadmap.md`.

This monorepo holds two independent packages:

| Package | Purpose | Depends on |
| --- | --- | --- |
| [`manyseg`](packages/manyseg) | The model library | `torch`, `timm` |
| [`manyseg-bench`](packages/manyseg-bench) | Implementation-neutral benchmarking of segmentation models (ManySeg, SMP, fastai, ...) | `torch`, `numpy`; model libraries are optional |

`manyseg` never imports `manyseg-bench`. A test enforces this.

## Quick start

```python
import torch
from manyseg import create_model

model = create_model(backbone="convnext_tiny", in_channels=4, decoder="unet", num_classes=3)
model(torch.randn(2, 4, 257, 301)).shape   # (2, 3, 257, 301): odd and non-square sizes are fine
```

Multiple inputs and outputs:

```python
model = create_model(
    inputs={
        "optical": {"channels": 4, "backbone": "convnext_tiny"},
        "sar": {"channels": 2, "backbone": "resnet18", "alignment": "same_extent"},
    },
    fusion={"type": "intermediate", "op": "gated"},
    decoder="unet",
    heads={
        "water": {"type": "segmentation", "channels": 2},
        "water_area": {"type": "regression", "channels": 1, "scope": "image"},
    },
)
out = model({"optical": optical, "sar": sar_at_half_resolution})
```

## What works today

* **Backbones.** timm CNNs (ResNet, EfficientNet, ConvNeXt, MobileNetV3/V4, RegNet,
  EdgeNeXt), hierarchical transformers and hybrids (Swin, PVTv2, MambaOut, FastViT,
  CaFormer), and plain ViTs (ViT, DeiT, EVA-02, DINOv2) through a pyramid neck. They are
  configured by introspection, without allow-lists.
* **Input channels.** Any channel count, with `repeat` (timm), `mean`, `rgb_plus_zero`,
  `rgb_plus_random` and `projection` adapters for pretrained stems.
* **Decoders.** `unet` (parameter- and FLOP-identical to SMP's Unet), `fastai_unet`
  (identical to fastai's DynamicUnet), `fpn`, `deeplabv3plus`, `upernet`, `segformer`.
* **Fusion.** Early, intermediate (`concat`, `add`, `gated`, `attention`, `cross_attention`)
  and late, with shared or independent encoders and explicit `same_grid` / `same_extent`
  alignment.
* **Heads.** Named heads for pixel or image segmentation, regression and classification.
* **Spatial.** Dynamic, odd and non-square sizes with an explicit pad-and-crop policy.
  Spatial alignment is tested numerically.
* **Deployment.** `torch.compile(fullgraph=True, dynamic=True)` (one graph across 10 sizes
  on CUDA), `torch.export` with dynamic H/W, ONNX with onnxruntime parity, AMP,
  channels_last. Swin's dynamic export is a known timm limitation.

## Development

```bash
uv sync --all-packages            # Python 3.12 workspace
uv run pytest packages/manyseg -n 4                         # unit
uv run pytest packages/manyseg -m slow -n 4                 # compile / export / ONNX
uv run pytest packages/manyseg-bench -n 4
```

Benchmarks:

```bash
export MANYSEG_BENCH_DATA=/path/to/data
uv run manyseg-bench prepare oxford_pet
uv run manyseg-bench train --provider smp --arch unet --encoder resnet34 --seeds 0,1,2
uv run manyseg-bench profile --provider manyseg --size 512,512 --compile
uv run manyseg-bench report experiments/
```

## Documentation

* `docs/architecture_review.md`: what we learned from SMP, fastai, timm, MMSeg, TerraTorch,
  TorchGeo and HF, including why fastai's U-Net differs from SMP's.
* `docs/api_design.md`, `docs/testing_strategy.md`, `docs/benchmark_design.md`, `docs/roadmap.md`.
* `experiments/`: one directory per experiment, with hypothesis, method, results and
  decision. Negative results are kept.

## License

MIT. Upstream libraries are acknowledged in `docs/architecture_review.md`. Ported designs
(e.g. the fastai-style decoder) cite their source in the module docstring.
