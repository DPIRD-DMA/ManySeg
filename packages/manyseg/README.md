# manyseg

Composable dense-prediction models for PyTorch: most [timm](https://github.com/huggingface/pytorch-image-models)
backbones (46 families tested), any channel count, multiple inputs and outputs. Compile- and export-friendly, and
checked against an independent benchmark ([`manyseg-bench`](https://pypi.org/project/manyseg-bench/)).

```bash
pip install manyseg                 # torch + timm only
pip install "manyseg[quant]"        # + torchao int8 quantisation
pip install "manyseg[onnx]"         # + ONNX export (add onnxruntime to check parity)
pip install "manyseg[instances]"    # + scipy / scikit-image post-processing
```

```python
import torch
from manyseg import create_model

model = create_model(backbone="convnext_tiny", in_channels=4, decoder="unet", num_classes=3)
model(torch.randn(2, 4, 257, 301)).shape   # (2, 3, 257, 301): odd and non-square sizes are fine
print(model.describe())                    # what was actually built, stage by stage
```

* **Backbones:** timm CNNs, hierarchical transformers and plain ViTs (through a pyramid neck),
  configured by introspection.
* **Inputs:** any channel count (pretrained stem adapters), several rasters with early,
  intermediate or late fusion and explicit `same_grid` / `same_extent` alignment, and
  non-raster metadata (date, angles, lat/lon) through FiLM.
* **Decoders:** `unet` (identical to SMP's), `unetplusplus`, `fastai_unet` (identical to
  fastai's), `fpn`, `deeplabv3plus`, `upernet` and `segformer`, shared or per head, with
  `out_stride` and an optional full-resolution `input_cross`.
* **Heads:** segmentation, regression (with uncertainty) and classification, at pixel or image
  scope and at reduced scale. Includes instance (boundary + distance) and point/density presets.
* **Deployment:** `torch.compile(dynamic=True)`, `torch.export` and ONNX with dynamic H/W
  (except Swin-family encoders, a timm limitation), and int8 quantisation (`manyseg.quantize`).

Every design default comes from a pre-registered, multi-seed experiment, and negative results
are kept. Documentation, examples and experiments are on
[GitHub](https://github.com/DPIRD-DMA/ManySeg). Status: pre-alpha, and APIs may change. MIT
licence.
