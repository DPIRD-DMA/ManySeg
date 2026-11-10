# E000: Measurements for the architecture review

Status: **archived** (2026-10-14). Not an experiment with a hypothesis: these are the one-off
probe scripts behind the measured numbers in the architecture review
(`docs/architecture_review.md`), kept so those numbers can be
traced and rerun. They predate the library and the benchmark, import timm, fastai and
segmentation-models-pytorch directly, and run their measurements at import (they are scripts,
not modules).

| File | What it measures |
| --- | --- |
| [`probe.py`](probe.py) | `features_only` output per timm family: feature shapes, layout (NCHW / NHWC) and reductions |
| [`probe2.py`](probe2.py) | The same encoders at odd and non-square input sizes |
| [`probe3.py`](probe3.py) | `torch.export` with dynamic sizes per encoder (`python probe3.py <encoder> <mode>`) |
| [`cmp.py`](cmp.py) | fastai's `DynamicUnet` against SMP's `Unet` on ResNet-34: layers and parameter counts |
| [`fl.py`](fl.py) | The same two decoders: FLOPs per part, in training mode |
| [`timm_audit.md`](timm_audit.md) | The results of `probe*.py` (timm 1.0.30, CPU, `pretrained=False`) |

Run from the repository root with the benchmark's extras installed
(`uv sync --all-packages --all-extras`), e.g. `uv run python experiments/E000_review_measurements/probe2.py`.
