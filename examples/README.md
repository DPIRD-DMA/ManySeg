# Examples

Short scripts that run on CPU with random weights: the numbered ones in a few seconds, the
walkthroughs that train briefly (`instance_fields`, `point_counting`) in 10–20 s
(`pretrained=False`; each
script shows where to switch pretrained weights on). They use a GPU if one is available.

| Script | Shows |
| --- | --- |
| [`01_single_input.py`](01_single_input.py) | 4-band U-Net on a timm backbone, odd non-square input, one training step, `model.describe()` |
| [`02_multispectral.py`](02_multispectral.py) | 13-band Sentinel-2-like input with the `repeat` stem adapter, and when to pick another adapter |
| [`03_multi_modal.py`](03_multi_modal.py) | The README model: optical + half-resolution SAR (`same_extent`) + metadata via FiLM, `describe(sample)`, a Graphviz PNG |
| [`04_multi_task.py`](04_multi_task.py) | Segmentation + image-level regression with uncertainty (Gaussian NLL) + a head with its own decoder |
| [`05_instances_points.py`](05_instances_points.py) | Boundary + distance instance heads and centre-heatmap + density heads: targets, labels, peaks, counts |
| [`06_export.py`](06_export.py) | `torch.export` with dynamic H/W, ONNX checked with onnxruntime, a `torch.compile` note |
| [`07_quantise.py`](07_quantise.py) | `manyseg.quantize`: PT2E static int8 and int8 weight-only, `model_size`, `compare_outputs` |
| [`08_classification_regression.py`](08_classification_regression.py) | Image-level models with no decoder: a multi-label classifier; then two S2 resolutions + S1 + day of year, a yield head with uncertainty and a pixel head together |

Longer walkthroughs:

| Script | Shows |
| --- | --- |
| [`instance_fields.py`](instance_fields.py) | Field delineation: 30 training steps on synthetic touching fields, then instances |
| [`point_counting.py`](point_counting.py) | Heatmap + offset + density heads: peaks, counts, instances by offset grouping |
| [`metadata_conditioning.py`](metadata_conditioning.py) | Metadata fields, missing values, the three injection modes, export with a vector input |

## Running

From the repository root:

```bash
uv sync --all-packages --all-extras
uv run python examples/01_single_input.py
```

With pip, `pip install manyseg` is enough for 01-05 and 08. Optional parts skip with a message when
their dependency is missing: ONNX in 06 needs `manyseg[onnx]` and `onnxruntime`, 07 needs
`manyseg[quant]` (torchao), and the PNG in 03 needs the Graphviz `dot` binary.

All examples run in the slow test tier (`packages/manyseg/tests/examples`):

```bash
uv run pytest packages/manyseg/tests/examples -m slow -n 4
```
