# E007b: Does gradient accumulation recover batch-16 results at batch 2? (pre-registered)

Status: **planned** (registered 2026-10-15, before any run). Follow-up to E007.

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
