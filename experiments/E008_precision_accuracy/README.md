# E008: Accuracy cost of bf16, fp16 and int8 (pre-registered)

Status: **planned** (registered 2026-10-15, before any run).

## Question
E014 measured speed: bf16 was 1.61× faster than fp32, compile added 1.46×, and
`manyseg.quantize` int8 is about 4× smaller. What accuracy do these cost on *trained* models?
Every variant is evaluated on the same trained weights, so the comparison is paired and needs
no new training per variant.

## Protocol
* Models: U-Net, ResNet-34 `tv_in1k`, recipe `fastai_like`, trained with `--save-model` on
  Oxford Pet (20 epochs) and Sen1Floods11 (stacked S1+S2, 60 epochs, batch 16). Seeds 0–2, so
  6 checkpoints, trained on the local RTX 4090 (accuracy does not depend on contention).
* Each checkpoint is evaluated on val and test with `manyseg-bench evaluate`, under these
  variants:

  | Variant | Device | What |
  | --- | --- | --- |
  | `fp32` | GPU | reference |
  | `bf16` | GPU | autocast bf16 |
  | `fp16` | GPU | autocast fp16 |
  | `int8_pt2e` | CPU | `quantize_pt2e` (x86 quantizer), calibrated on 32 training batches (no augmentation), eager quantise/dequantise graph |
  | `int8_pt2e_compiled` | CPU | the same program after `torch.compile` with inductor freezing (the deployed form; the quantisation agent found it can differ from the eager form) |
  | `int8_weight_only` | CPU | `quantize_weights(model)` |

* Reported per variant: mIoU, boundary F1, and argmax agreement with `fp32` on the same
  images, as paired Δ vs `fp32` (mean ± SD over seeds). CPU latency is reported only as
  indicative, since the machine is shared.
* Budget: 6 training runs (about 1 GPU-hour locally) plus 36 evaluations (CPU int8 dominates,
  about 1–2 h at 4 threads).

## Decision rule
* A variant counts as **safe** if its mean test-mIoU drop vs `fp32` is ≤ 0.3 points on **both**
  datasets and no single seed drops more than 1 point. The docs then recommend it without
  qualification (bf16 already is, on speed grounds; E014).
* Otherwise the docs give the measured drop next to the variant and recommend validating on
  the user's own data. For int8, the docs recommend whichever of `int8_pt2e_compiled` and
  `int8_weight_only` is safe. If neither is, they say int8 needs quantisation-aware training,
  which is out of scope.

## Amendment (2026-10-15 11:45, after the first pass)
The first pass found that `int8_pt2e_compiled` lost about 6 mIoU on every Pet seed (agreement
94–95%), while the eager `int8_pt2e` reference and Sen1Floods11 were lossless. Cause: this
machine's CPU (AMD Ryzen 9 5950X, Zen 3) has **no VNNI**. On such CPUs, inductor's
u8 × s8 int8 kernels saturate their 16-bit intermediate sums unless activations use a reduced
range (torchao's docs advise this). On a 256-image Pet test subset, `reduce_range=True` raised
compiled agreement from 0.9501 to 0.9949, matching eager. This is a **library bug**:
`manyseg.quantize`'s default quantizer assumed VNNI. It is fixed: `quantize_pt2e` now
defaults to `reduce_range` when the CPU lacks VNNI. Both PT2E variants are re-evaluated with
the fixed library on all 6 checkpoints (`runs/eval_rr/`). The first-pass records stay in
`runs/eval/` as the evidence for the bug. The decision uses the fixed library, which is what
users get.
