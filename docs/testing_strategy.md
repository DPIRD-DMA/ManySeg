# Testing strategy

The functional contracts of the `manyseg` library are developed test-first. Scientific
claims (accuracy, speed) are established by `manyseg-bench` experiments, not by unit tests.

## Test tiers

| Tier | Location / marker | Runs | Budget |
| --- | --- | --- | --- |
| Unit | `packages/*/tests/unit`, `packages/manyseg-bench/tests` | every commit, CPU | < 2 min with `-n 8` |
| Slow integration | `-m slow` (`tests/integration`) | pre-merge, nightly | < 5 min |
| Network | `-m network` | nightly (needs weight cache) | small downloads |
| Hardware | `device` fixture: `cpu`, `cuda`, `mps` | wherever the device exists | n/a |
| Benchmark | `manyseg-bench` CLI, never in pytest | on demand, logged under `experiments/` | per experiment budget |

`pytest` excludes `slow` and `network` by default (`addopts` in each `pyproject.toml`). To run
everything:

```bash
uv run pytest packages/manyseg -m "slow or network or not slow" -n 8
```

## Device policy

The `device` fixture parametrises over `cpu`, `cuda` and `mps`. An unavailable device is
**skipped and reported**, never replaced by another device. The CI matrix runs CPU on Linux
and MPS on macOS-arm64. CUDA runs on the self-hosted GPU runner.

## What is tested (and how)

Shape assertions alone are not enough. Each contract has a numerical or behavioural check:

| Contract | Test style |
| --- | --- |
| Channel adapters | Exact algebraic identities: 3→3 is identity; `mean` preserves the response to channel-constant inputs for all C; `repeat` (the timm rule) preserves it only for C=1 or C%3=0, which documents its bias; `rgb_plus_zero` equals the RGB model on the first 3 channels. |
| Pretrained adaptation | The adapter is applied to (mocked) pretrained weights and every other tensor is unchanged. Network test: `repeat` reproduces timm's own `in_chans` loading bit-for-bit. |
| Projection adapter | At init, the C-channel model equals the RGB model on `x[:, :3]`. |
| Dynamic sizes | Same instance, many sizes, including odd, non-square and smaller than the stride. Output size equals input size. |
| Padding semantics | `model(x) == model(zero_pad(x))[..., :H, :W]` exactly. |
| Spatial alignment | Gradient-centroid test: the input sensitivity of output pixel (i, j) is centred on (i, j). `aligned` upsampling must be under 1.5 px. The known bias of nearest upsampling is bounded as a regression guard (E001). |
| Gradients | Every parameter receives a finite, non-zero gradient. `gradcheck` in float64 on decoder blocks. |
| Serialization | `state_dict` round-trip reproduces outputs exactly. |
| Mixed precision | bf16 autocast on CPU; fp16 autocast on accelerators; outputs finite. |
| Devices | Each accelerator's output matches CPU within 1e-3. |
| channels_last | Matches contiguous within 1e-4. |
| `torch.compile` | `fullgraph=True, dynamic=True` matches eager. Graph count: 1 on CUDA for 10 sizes; bounded on CPU (PyTorch's oneDNN backend-selection guards). |
| `torch.export` | Dynamic H/W (`Dim`, min 33) with outputs matching eager at unseen sizes. Multi-head dict outputs. |
| ONNX | Dynamo exporter with dynamic H/W; onnxruntime output matches eager within 1e-3. |
| Library independence | `manyseg` source never references `manyseg_bench`. Bench providers import model libraries lazily. |
| Bench metrics | Hand-computed confusion-matrix cases, ignore index, absent classes, accumulation invariance, boundary F1 with shifts. |
| Bench loader | Seeded determinism; eval loader is the identity and keeps order; augmented images and masks stay aligned. |
| Bench providers | Contract conformance (plain Tensor `(N, K, H, W)`), parameter partition, pretrained fingerprints identical across providers. |
| Bench training | Loss decreases on synthetic data for every provider; freezing leaves encoder conv weights unchanged. |

## Regression policy

Every bug fix gets a test that fails before the fix. Examples so far:

* Inductor could not lower `%` on symbolic sizes, so the padding arithmetic now uses floor
  division (covered by the compile tests).
* torch.export could not prove `h <= 32·ceil(h/32)` for slice-cropping, so cropping now uses
  negative padding (covered by `test_torch_export_dynamic_hw`).

## Parametrisation

Backbone families are parametrised with the smallest timm variant of each family
(`resnet18`, `efficientnet_b0`, `convnext_atto`, `mobilenetv3_small_100`, `regnetx_002`),
using `pretrained=False`. Unit tests therefore need no network access and stay fast. The
backbone capability registry (Phase 4) adds a generated compatibility matrix over the wider
timm zoo, run under `-m slow`.
