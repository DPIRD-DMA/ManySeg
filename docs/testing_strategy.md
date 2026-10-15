# Testing strategy

The functional contracts of the `manyseg` library are developed test-first. Scientific
claims (accuracy, speed) are established by `manyseg-bench` experiments, not by unit tests.

## Test tiers

| Tier | Location / marker | Runs | Budget |
| --- | --- | --- | --- |
| Unit | `packages/*/tests/unit`, `packages/manyseg-bench/tests` | every commit, CPU | < 2 min with `-n 8` |
| Slow integration | `-m slow` (`tests/integration`) | pre-merge, nightly | < 5 min |
| Examples | `-m slow` (`tests/examples`): every `examples/*.py` as a CPU subprocess | pre-merge, nightly | ~30 s with `-n 4` |
| Network | `-m network` | nightly (needs weight cache) | small downloads |
| Hardware | `device` fixture: `cpu`, `cuda`, `mps` | wherever the device exists | n/a |
| Benchmark | `manyseg-bench` CLI, never in pytest | on demand, logged under `experiments/` | per experiment budget |

`pytest` excludes `slow` and `network` by default (`addopts` in each `pyproject.toml`). To run
everything:

```bash
uv run pytest packages/manyseg -m "slow or network or not slow" -n 8
```

## Device policy

The `device` fixture parametrises over `cpu`, `cuda` and `mps`. A device counts only if it
can actually allocate. An unavailable or unusable device is **skipped and reported**, never
replaced by another device.

* CI runs CPU on Linux (Python 3.10 and 3.12) and on macOS-arm64.
* GitHub's macOS VMs report MPS as available, but every allocation fails, so **MPS tests need
  a physical Apple Silicon machine** (`uv run pytest packages/manyseg -k mps`).
* CUDA tests run on the development workstation (RTX 4090).

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
| DDP | Two CPU ranks (gloo): replicas stay identical after a step with different data per rank. |
| Export helper | `export_model` round-trips at unseen sizes and batch sizes, with dict inputs and with families whose guards the solver cannot prove. |
| Quantisation | torchao weight-only / dynamic on CNN, depthwise and Linear-heavy backbones: odd sizes, argmax agreement > 0.95, < 0.35x size. Slow tier: PT2E static int8 with dynamic H/W, batch and dict inputs and outputs, every decoder, save/load, compile with freezing, fixed-size fallback, the Swin error. The missing-torchao error is tested by blocking the import. |
| Metadata conditioning | Field encoders: cyclic continuity across the period, coords across the dateline and poles, exact normalisation, out-of-range categories become unknown. FiLM and head modes at init equal the unconditioned model with the same weights (`atol=rtol=0`, every decoder). A metadata-only synthetic task is fitted by the conditioned model and not by the plain one. Missing values: all-missing, NaN, partial mask, field dropout. Every decoder × {film, head, channels} at odd sizes with dict outputs and task decoders. Slow tier: export (dynamic H/W, dynamic batch, mask), ONNX parity, compile fullgraph. |
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

Test isolation: an autouse fixture resets Dynamo after each test. Many tests compile
different models through the same `DenseModel.forward` code object, and without the reset
the per-code-object cache hits Dynamo's recompile limit.

## Parametrisation

Backbone families are parametrised with the smallest timm variant of each family
(`resnet18`, `efficientnet_b0`, `convnext_atto`, `mobilenetv3_small_100`, `regnetx_002`),
using `pretrained=False`. Unit tests therefore need no network access and stay fast. The
backbone capability registry (Phase 4) adds a generated compatibility matrix over the wider
timm zoo, run under `-m slow`.
