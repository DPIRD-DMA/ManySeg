# E007: Normalisation at small batch sizes (pre-registered)

Status: **planned** (registered 2026-10-15, before any run).

## Question
Large remote-sensing tiles often force batches of 1–4 per GPU. BatchNorm statistics are then
noisy, both in the decoder (BN by default, as in SMP) and in a pretrained encoder whose
running statistics keep updating. Does GroupNorm or no norm in the decoder, and/or freezing
the encoder's BN statistics, help at batch 2 without hurting at batch 16?

## Protocol
* U-Net, ResNet-34 `tv_in1k`, recipe `fastai_like` (lr unchanged across batch sizes, so batch 2
  takes 8× more optimiser steps per epoch; linear lr scaling is *not* applied, a stated
  limitation). Oxford Pet (20 epochs) and Sen1Floods11 stacked S1+S2 (60 epochs).
* Factors: decoder `norm` ∈ {`batch` (default), `group`, `none`} × encoder BN ∈ {`train`
  (default), `frozen`: the new recipe option `freeze_encoder_bn`, which keeps running
  statistics at their ImageNet values while affine weights still train} × batch ∈ {2, 16} ×
  2 datasets × seeds 0–2 = 72 runs. Pawsey Setonix MI250X, one GCD per run (`run.sh`
  `EMIT_TASKS`).
* Reported per arm: val and test mIoU, boundary F1, and train throughput.
* Budget: 72 runs, about 10–15 GCD-hours.

## Decision rule
Pooled SD over 3 seeds; "test agrees" means the same direction on test.

* At **batch 2**, an arm counts as helping if its val mIoU − default (`batch` + `train`) > 2
  pooled SD on **both** datasets and test agrees (the roadmap's stopping rule). Five arms are
  compared with one default, so a single marginal pass is reported as such.
* If an arm helps at batch 2 **and** is within 1 pooled SD of the default at batch 16 (or
  better) on both datasets, the docs recommend it for small-batch training. A default change
  (e.g. `freeze_encoder_bn` on in the recipes, or GroupNorm in decoders) also needs it to be
  no worse at batch 16, because SMP parity of `unet` with `norm="batch"` is a design
  commitment. A decoder default therefore stays `batch` regardless, and only the docs change.
* Also reported: the default's batch-2 − batch-16 gap. If it is within 2 pooled SD on both
  datasets, small batches are not a problem for this setup, and the result is negative.
