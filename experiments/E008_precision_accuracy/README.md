# E008: Accuracy cost of bf16, fp16 and int8 (pre-registered)

Status: **complete** (2026-10-15; 6 checkpoints × 6 variants, local RTX 4090 + AMD Ryzen 9 5950X CPU; training code `23e3333`, evaluation `682f85d` / fixed quantizer `2d92e66`). **Every variant is safe with the fixed library. The first pass found a real int8 bug on CPUs without VNNI (−6.5 mIoU), which is fixed.**

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

## Results (paired Δ test mIoU vs fp32 on the same weights; mean ± SD over 3 seeds)

| dataset | variant | n | Δ test mIoU vs fp32 (pts) | worst seed | agreement % |
|---|---|---|---|---|---|
| oxford_pet | bf16 | 3 | -0.00 ± 0.01 | -0.01 | 99.93 |
| oxford_pet | fp16 | 3 | -0.00 ± 0.00 | -0.00 | 99.99 |
| oxford_pet | fp32 | 3 | +0.00 ± 0.00 | +0.00 | 100.00 |
| oxford_pet | int8_pt2e | 3 | -0.09 ± 0.01 | -0.10 | 99.48 |
| oxford_pet | int8_pt2e [first pass, no reduce_range] | 3 | -0.04 ± 0.01 | -0.05 | 99.61 |
| oxford_pet | int8_pt2e_compiled | 3 | -0.10 ± 0.02 | -0.11 | 99.49 |
| oxford_pet | int8_pt2e_compiled [first pass, no reduce_range] | 3 | -6.49 ± 0.60 | -7.17 | 94.50 |
| oxford_pet | int8_weight_only | 3 | -0.05 ± 0.02 | -0.07 | 99.59 |
| sen1floods11 | bf16 | 3 | +0.00 ± 0.00 | +0.00 | 99.98 |
| sen1floods11 | fp16 | 3 | +0.00 ± 0.00 | -0.00 | 100.00 |
| sen1floods11 | fp32 | 3 | +0.00 ± 0.00 | +0.00 | 100.00 |
| sen1floods11 | int8_pt2e | 3 | -0.10 ± 0.08 | -0.18 | 99.74 |
| sen1floods11 | int8_pt2e [first pass, no reduce_range] | 3 | -0.01 ± 0.02 | -0.03 | 99.86 |
| sen1floods11 | int8_pt2e_compiled | 3 | -0.13 ± 0.11 | -0.25 | 99.74 |
| sen1floods11 | int8_pt2e_compiled [first pass, no reduce_range] | 3 | -0.01 ± 0.02 | -0.04 | 99.86 |
| sen1floods11 | int8_weight_only | 3 | +0.03 ± 0.06 | -0.04 | 99.95 |

Agreement is the share of valid test pixels whose argmax matches fp32 on the same device.
Sen1Floods11's Bolivia (held-out event) split behaves the same (within 0.3 points for every
fixed variant).

* **bf16 / fp16 autocast:** no measurable effect (≤ 0.01 points, ≥ 99.93% agreement).
* **int8 weight-only** (`quantize_weights`): −0.05 / +0.03 points. The model is about 4× smaller.
  It is not faster (E014).
* **int8 PT2E, compiled (the deployed form):** −0.10 / −0.13 points with the fixed default
  quantizer (`reduce_range` on CPUs without VNNI). That is about 0.1 point below full-range int8
  (eager first pass: −0.04 / −0.01), the expected cost of 7-bit activations, and still well
  inside the bar.
* **The bug:** without `reduce_range`, compiled int8 lost **6.5 ± 0.6 points on Pet** (94.5%
  agreement) on this CPU, while its eager reference and Sen1Floods11 were fine. The Pet model's
  activation ranges saturate inductor's u8 × s8 kernels on CPUs without VNNI. This failure is
  silent, and depends on the data and the hardware. It would have shipped as a library default
  if E008 had not evaluated the deployed form.

## Decision
All five variants are **safe** by the pre-registered rule (mean drop ≤ 0.3 points on both
datasets, no seed > 1 point).
* The docs recommend bf16 autocast (with `torch.compile` for inference, E014) without
  qualification.
* For int8 CPU deployment, `quantize_pt2e` + `torch.compile` (freezing) is safe with the fixed
  default. Users who quantise on one CPU and deploy on another should set `reduce_range` for the
  *deployment* CPU (`True` unless it has VNNI). `quantize_weights` is the no-calibration option
  for size alone.
* Keep evaluating the deployed (compiled) form, not only the eager reference.
