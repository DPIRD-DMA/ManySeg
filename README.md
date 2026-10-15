<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="ManySeg" src="docs/assets/logo-light.svg" width="420">
  </picture>
</p>

Composable dense-prediction models for PyTorch: any timm backbone, any channel count, multiple
inputs and outputs. Compile- and export-friendly, and checked against an independent
benchmark.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
    <img alt="ManySeg architecture: per-input encoders feed a fusion block, metadata conditions the decoder through FiLM, shared or per-head decoders feed named heads including instance outputs; every stage is swappable" src="docs/assets/architecture-light.svg" width="860">
  </picture>
</p>

Each stage is configured independently. The row under the line lists the options for each
stage. The diagram shows the model built by the multi-input example below.

> Status: pre-alpha. The model library is feature-complete for the original plan (FSDP2 and a
> ViT metadata-token mode remain); experiments and release preparation are ongoing. APIs may
> change. See `docs/roadmap.md`.

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
from manyseg.instances import instance_heads

model = create_model(
    inputs={
        "optical": {"channels": 4, "backbone": "convnext_tiny"},
        "sar": {"channels": 2, "backbone": "resnet18", "alignment": "same_extent"},
        "meta": {"kind": "vector", "fields": {                   # non-raster conditioning
            "doy": {"type": "cyclic", "period": 365.25},
            "sun_elev": {"type": "continuous", "mean": 45.0, "std": 15.0},
            "latlon": {"type": "coords"},
        }},
    },
    fusion={"type": "intermediate", "op": "gated"},
    decoder="unet",
    heads={
        **instance_heads(decoders={"boundary": "fpn"}),  # extent, boundary, distance
        "water_area": {"type": "regression", "channels": 1, "scope": "image"},
    },
)
meta = model.conditioner.encoders["meta"].pack({"doy": doy, "sun_elev": sun_elev, "latlon": latlon})
out = model({"optical": optical, "sar": sar_at_half_resolution, "meta": meta})
# out["extent"], out["boundary"], out["distance"]: (B, 1, H, W); out["water_area"]: (B, 1)
```

## What works today

* **Backbones.** timm CNNs (ResNet, EfficientNet, ConvNeXt, MobileNetV3/V4, RegNet,
  EdgeNeXt), hierarchical transformers and hybrids (Swin, PVTv2, MambaOut, FastViT,
  CaFormer), and plain ViTs (ViT, DeiT, EVA-02, DINOv2) through a pyramid neck. They are
  configured by introspection, without allow-lists.
* **Input channels.** Any channel count, with `repeat` (timm), `mean`, `rgb_plus_zero`,
  `rgb_plus_random` and `projection` adapters for pretrained stems.
* **Decoders.** `unet` (parameter- and FLOP-identical to SMP's Unet), `unetplusplus`,
  `fastai_unet` (identical to fastai's DynamicUnet), `fpn`, `deeplabv3plus`, `upernet`,
  `segformer`. One decoder can be shared by all heads, or a head can have its own. `unet`'s
  optional `input_cross` (fastai's full-resolution input stage) recovers most of fastai's
  gain on remote sensing at +14% FLOPs (E012).
* **Output resolution.** The U-Net-family decoders take `out_stride` 1 (default), 2 or 4. The
  stages finer than that are not built: `fastai_unet` at stride 2 drops its full-resolution
  stages, about 40% of its FLOPs. Pixel heads take `"scale"` 1, 1/2, 1/4 or 1/8 and return
  `ceil(H·s) × ceil(W·s)`. They skip upsampling when the decoder stride already matches.
  `manyseg.utils` has helpers for targets at head scale (label mode with `ignore_index`, area
  mean, count-preserving sum), for upsampling predictions and for mapping coordinates.
* **Fusion.** Early, intermediate (`concat`, `add`, `gated`, `attention`, `cross_attention`)
  and late, with shared or independent encoders and explicit `same_grid` / `same_extent`
  alignment.
* **Metadata.** Non-raster inputs (date, angles, lat/lon, sensor id, tabular floats) as
  `"kind": "vector"` inputs, encoded by field type, with missing-value handling. They
  condition the model by FiLM on the features entering each decoder (default), through image
  heads, or as extra input channels. Pretrained encoders are not modified.
* **Heads.** Named heads for pixel or image segmentation, regression and classification,
  with optional uncertainty (mean + log-variance) for regression.
* **Instances and points.** Boundary + distance instance heads (`manyseg.instances`) and
  centre-heatmap, offset and density heads (`manyseg.points`), with target generation and
  post-processing to instance labels, points, counts and panoptic labels.
* **Spatial.** Dynamic, odd and non-square sizes with an explicit pad-and-crop policy.
  Spatial alignment is tested numerically.
* **Deployment.** `torch.compile(fullgraph=True, dynamic=True)` (one graph across 10 sizes
  on CUDA), `torch.export` with dynamic H/W, ONNX with onnxruntime parity, AMP,
  channels_last. Swin's dynamic export is a known timm limitation. For inference, use bf16
  autocast plus `torch.compile`: on an idle RTX 4090 this was 1.6× (bf16) × 1.46× (compile)
  faster than eager fp32 (E014).
* **Quantisation.** `manyseg.quantize` (optional `manyseg[quant]` extra, torchao): static int8
  through PT2E with dynamic H/W, returned as an `ExportedProgram`, and int8 weight-only or
  dynamic quantisation of the eager model. A ResNet-34 U-Net shrinks to 0.27x. Reports give the
  serialized size and the agreement with the float model.

## What the experiments found

All experiments are pre-registered, multi-seed and written up in `experiments/`, negative
results included. Highlights:

* `fastai_unet`'s edge over SMP-style `unet` on Sen1Floods11 comes from reinjecting the raw
  input at full resolution. `unet` + `input_cross` recovers 81–83% of it at +14% FLOPs, against
  fastai's 4× (E006, E012). On Oxford Pet the recipe matters more than the decoder (E002, E003).
* Decoder `out_stride=2` costs no accuracy for `unet` and saves 11–18% memory (E006).
* Boundary + distance heads double instance PQ over extent + connected components (E010).
  Point detection needs a validation-calibrated peak threshold: F1 0.40 → 0.59 (E011b).
* Negative results: aligned upsampling (E001b), fusion beyond early stacking on Sen1Floods11
  (E005), and the `projection` stem adapter (E004). Metadata FiLM helps less than 2 SE on held-out
  flood events (E009).
* Running: CloudSEN12+ cloud and shadow masking (auxiliary layers, metadata on unseen locations,
  `input_cross` replication; E013).

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
uv run manyseg-bench prepare oxford_pet   # or sen1floods11, ai4boundaries, oam_tcd, cloudsen12
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
