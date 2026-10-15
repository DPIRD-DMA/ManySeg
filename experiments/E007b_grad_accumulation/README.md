# E007b: Does gradient accumulation recover batch-16 results at batch 2? (pre-registered)

Status: **complete** (2026-10-15; 30 runs, local RTX 4090, code `1293f92`). Follow-up to E007. **Freezing the encoder's BN statistics is a real small-batch fix on both datasets; gradient accumulation alone is not. The full small-memory recipe matches batch 16 on Sen1Floods11 and is 0.2 points short on Pet.**

## Question
In E007 (MI250X), batch 2 cost 1.3–1.6 val mIoU points against batch 16. Freezing the
encoder's BN statistics recovered more than that on Sen1Floods11, but batch 2 also took 8×
more optimiser steps, so normalisation and step count were confounded. Gradient accumulation
(`grad_accum_steps`, new in the bench recipe) restores the effective batch and step count
while BN still sees micro-batches of 2. Which matters, and can a small-memory GPU match
batch 16?

## Protocol
* Local RTX 4090 (no Pawsey units). Every arm runs here, including the references, because
  arms are never compared across machines. U-Net, ResNet-34 `tv_in1k`, recipe `fastai_like`,
  Pet (20 epochs) and Sen1Floods11 stacked S1+S2 (60 epochs). Seeds 0–2.
* Arms (batch × accumulation, encoder BN): `b16` (16 × 1, train, the reference); `b2` (2 × 1,
  train); `b2-frozen` (2 × 1, frozen); `b2x8` (2 × 8, train); `b2x8-frozen` (2 × 8, frozen).
  That is 30 runs, about 2 h.
* The machine is shared: system RAM is watched, and the run is paused if available memory
  drops below 16 GB. Timing is contended and reported only as indicative.

## Decision rule (pooled seed SD; val mIoU, test must agree)
* **Small-memory recipe:** if `b2x8-frozen` is within 1 pooled SD of `b16`, or better, on
  **both** datasets, the docs recommend "batch 2 × accumulate 8 + `freeze_encoder_bn=True`"
  for GPUs that fit only small batches. Otherwise the docs report the gap.
* **Attribution:** `b2x8` − `b2` (same BN noise, 8× fewer, larger steps) and `b2x8-frozen` −
  `b2x8` (BN freezing at matched steps) are reported as effects. Each is called real only if
  it exceeds 2 pooled SD on both datasets.

## Results (mean ± seed SD, 3 seeds; Pet 20 epochs, Sen1Floods11 60 epochs)

| dataset | arm | n | val mIoU | test mIoU | train img/s |
|---|---|---|---|---|---|
| oxford_pet | b16 | 3 | 0.8209 ± 0.0005 | 0.8220 ± 0.0003 | 568 |
| oxford_pet | b2 | 3 | 0.8079 ± 0.0004 | 0.8093 ± 0.0016 | 107 |
| oxford_pet | b2-frozen | 3 | 0.8092 ± 0.0026 | 0.8139 ± 0.0019 | 106 |
| oxford_pet | b2x8 | 3 | 0.8106 ± 0.0032 | 0.8124 ± 0.0008 | 125 |
| oxford_pet | b2x8-frozen | 3 | 0.8188 ± 0.0007 | 0.8202 ± 0.0008 | 133 |
| sen1floods11 | b16 | 3 | 0.8926 ± 0.0065 | 0.8878 ± 0.0058 | 496 |
| sen1floods11 | b2 | 3 | 0.8858 ± 0.0049 | 0.8942 ± 0.0019 | 108 |
| sen1floods11 | b2-frozen | 3 | 0.9075 ± 0.0011 | 0.9038 ± 0.0010 | 109 |
| sen1floods11 | b2x8 | 3 | 0.8783 ± 0.0090 | 0.8872 ± 0.0026 | 125 |
| sen1floods11 | b2x8-frozen | 3 | 0.9013 ± 0.0006 | 0.8928 ± 0.0011 | 127 |

```
oxford_pet: b2x8-frozen - b16: val -0.21 (1 pooled SD 0.06), test -0.18 -> worse
  accumulation at fixed BN noise: b2x8 - b2: val +0.27 (2 pooled SD 0.45), test +0.31
  BN freezing at matched steps: b2x8-frozen - b2x8: val +0.82 (2 pooled SD 0.46), test +0.78
  batch-2 gap (default): b2 - b16: val -1.29 (2 pooled SD 0.08), test -1.27
sen1floods11: b2x8-frozen - b16: val +0.87 (1 pooled SD 0.46), test +0.49 -> within/better
  accumulation at fixed BN noise: b2x8 - b2: val -0.75 (2 pooled SD 1.45), test -0.69
  BN freezing at matched steps: b2x8-frozen - b2x8: val +2.31 (2 pooled SD 1.28), test +0.55
  batch-2 gap (default): b2 - b16: val -0.68 (2 pooled SD 1.15), test +0.64
```

## Decision
* **Small-memory recipe: narrowly not met.** `b2x8-frozen` beats `b16` on Sen1Floods11 (+0.87
  val, +0.49 test) but is 0.21 val / 0.18 test short on Pet. Seed variance there is tiny
  (1 SD = 0.06), so the rule ("within 1 pooled SD on both") fails. The docs give the numbers:
  batch 2 × 8 with frozen encoder BN gets within 0.2 mIoU of batch 16 on natural images and
  matches or beats it on multispectral data.
* **BN freezing is a real effect (> 2 pooled SD on both datasets, test agrees):** at matched
  optimiser steps it gives +0.82 (Pet) and +2.31 (Sen1Floods11) val mIoU. With E007 (frozen BN
  +2.55 on Sen1Floods11 at batch 2), this is now the documented fix for small-batch training:
  keep the encoder's BatchNorm in eval mode (`freeze_encoder_bn=True` in the bench). The
  default stays off, because at batch 16 nothing shows it is needed.
* **Gradient accumulation alone: no effect** (+0.27 Pet, −0.75 Sen1Floods11, both within noise).
  On its own it does not repair small-batch training, because BN still normalises over 2 images.
  Combined with BN freezing it closes most of the remaining gap on Pet.
* On Sen1Floods11, plain batch 2 with frozen BN (`b2-frozen`, 8× more steps) scored best of all
  (0.9075 val, 0.9038 test). Many small steps are not harmful there once BN is fixed.
