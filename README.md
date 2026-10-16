<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="ManySeg" src="docs/assets/logo-light.svg" width="420">
  </picture>
</p>

<p align="center">
Segmentation and other dense-prediction models for PyTorch, built from
<a href="https://github.com/huggingface/pytorch-image-models">timm</a> encoders.<br>
Any number of bands, several inputs at different resolutions, metadata, and many outputs in one model.
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
    <img alt="ManySeg architecture: per-input encoders feed a fusion block, metadata conditions the decoder through FiLM, shared or per-head decoders feed named heads including instance outputs; every stage is swappable" src="docs/assets/architecture-light.svg" width="860">
  </picture>
</p>

> **Status: pre-alpha.** The API may still change before the first release.

## Install

ManySeg is not on PyPI yet. Install it from GitHub (Python ≥ 3.10, PyTorch ≥ 2.9):

```bash
pip install "manyseg @ git+https://github.com/DPIRD-DMA/ManySeg#subdirectory=packages/manyseg"
```

Optional extras: `[quant]` (int8 quantisation), `[onnx]` (ONNX export), `[instances]`
(instance post-processing). After the first release this becomes `pip install manyseg`.

## Quick start

```python
import torch
from manyseg import create_model

model = create_model(backbone="convnext_tiny", in_channels=4, num_classes=3)  # U-Net by default
logits = model(torch.randn(2, 4, 257, 301))   # (2, 3, 257, 301): any size, odd or non-square
print(model.describe())                       # what was built, stage by stage
```

`backbone` is a timm model name: ResNet, EfficientNet, ConvNeXt, Swin, PVTv2, ViT, DINOv2 and
most other timm families that expose multi-scale features. 46 families are tested
([compatibility table](docs/backbone_compatibility.md)); a few only accept inputs up to a fixed
size. Pretrained weights are downloaded by timm and adapted to your band count.

## Common tasks

**Multispectral imagery.** Set `in_channels` (e.g. 13 for Sentinel-2). The pretrained RGB stem
is adapted automatically.

**Pick a speed/accuracy trade-off.** Presets are measured configurations:

```python
from manyseg.presets import preset

model = create_model(**preset("balanced"), in_channels=4, num_classes=2)
```

| preset | encoder + decoder | relative speed | accuracy |
|---|---|---|---|
| `fast` | PVTv2-b0 + SegFormer head | 2.4× | slightly below the default on Pet |
| `compact` | PVTv2-b0 + quarter-width U-Net | 2.2× | matches or beats the default |
| `balanced` | PVTv2-b0 + half-width U-Net | 2.1× | matches or beats the default |
| `accurate` | PVTv2-b0 + U-Net | 1.5× | matches or beats the default |

Relative to the default ResNet-34 U-Net, on an RTX 4090 with bf16 + `torch.compile`; accuracy
on two datasets. Validate on your own data.

**Several inputs, possibly at different resolutions**, e.g. optical + radar:

```python
model = create_model(
    inputs={
        "optical": {"channels": 4, "backbone": "convnext_tiny"},
        "sar": {"channels": 2, "backbone": "resnet18", "alignment": "same_extent"},  # half res OK
    },
    fusion={"type": "intermediate", "op": "concat"},
    num_classes=2,
)
out = model({"optical": optical, "sar": sar})
```

**Several outputs.** Name your heads; the model returns a dict:

```python
model = create_model(
    backbone="resnet50", in_channels=4,
    heads={
        "landcover": {"type": "segmentation", "channels": 8},
        "height":    {"type": "regression", "channels": 1},                  # per pixel
        "flooded":   {"type": "classification", "channels": 2, "scope": "image"},
    },
)
```

**Image classification or regression.** Use only image heads: no decoder is built, and the
model is a classifier that still accepts several inputs, metadata and extra heads.

```python
model = create_model(backbone="resnet50", in_channels=12, heads={
    "labels": {"type": "classification", "channels": 19, "scope": "image"}})
```

If you also have pixel labels, training with an extra segmentation head can improve the
classifier. Drop it before deployment with `model.select_heads(["labels"])`.

**Metadata** (acquisition date, sun angle, location, sensor) as extra inputs:

```python
inputs = {
    "image": {"channels": 4},
    "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25},
                                          "latlon": {"type": "coords"}}},
}
```

Missing values are handled. In our tests metadata rarely helped, so compare against a model
without it on your data.

**Objects and counts.** `manyseg.instances` (touching fields, crowns, buildings) and
`manyseg.points` (trees, animals: centres, counts) provide the heads, training targets and
post-processing.

**Deployment.** Models work with `torch.compile(dynamic=True)`, `torch.export` and ONNX with
dynamic input sizes (except Swin-family encoders, a timm limitation), and with int8
quantisation (`manyseg.quantize`). For fast GPU inference use bf16
autocast plus `torch.compile`.

Each of these has a runnable script in [`examples/`](examples/); they run on CPU in seconds.

## Tips from our benchmarks

* **Small batches** (≤ 4 per GPU): keep the encoder's BatchNorm layers in eval mode during
  training (frozen running statistics; their weights still train).
* **Memory-bound?** `decoder_kwargs={"out_stride": 2}` costs no accuracy for the U-Net and
  saves 11–18% memory.
* **Water and other per-pixel spectral targets:** `decoder_kwargs={"input_cross": True}`
  feeds the raw bands to the last decoder stage. It did not help pet or cloud segmentation.
* **Mixed precision is safe:** bf16, fp16 and int8 weight-only showed no accuracy loss.
* **Fusion:** stacking all bands into one input was as good as fancier fusion on the datasets
  we tried; start there.

Every claim here comes from a pre-registered, multi-seed experiment in
[`experiments/`](experiments/), where negative results are kept too.

## Documentation

* [`examples/`](examples/): runnable scripts for each task above.
* [`docs/api_design.md`](docs/api_design.md): the full API (inputs, fusion, decoders, heads,
  metadata, resolution, export, quantisation).
* [`experiments/`](experiments/): the evidence behind the defaults and tips.

## Benchmarking (`manyseg-bench`)

A separate package for comparing segmentation libraries (ManySeg, segmentation_models_pytorch,
fastai, timm) under identical data, augmentation and training. `manyseg` does not depend on it.
See [`docs/benchmark_design.md`](docs/benchmark_design.md).

## Contributing

```bash
uv sync --all-packages
uv run pytest packages/manyseg -n 4            # fast tests
uv run pytest packages/manyseg -m slow -n 4    # compile, export, ONNX
```

See [`docs/roadmap.md`](docs/roadmap.md) and [`docs/testing_strategy.md`](docs/testing_strategy.md).

## License

MIT. Designs ported from other libraries (e.g. the fastai-style U-Net) cite their source in
the code; see [`docs/architecture_review.md`](docs/architecture_review.md).
