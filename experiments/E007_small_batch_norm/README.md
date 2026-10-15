# E007: Normalisation at small batch sizes (pre-registered)

Status: **complete** (2026-10-15; 30 runs, Pawsey Setonix MI250X, code `c3b4bda`). **Mixed or negative by the rule: no arm passes on both datasets. Freezing encoder BN helps a lot on Sen1Floods11; GroupNorm in the decoder helps on Pet.**

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

## Amendment (2026-10-15 09:52, before any run finished)
The project's Pawsey GPU allocation is being used too fast, so the design is cut from 72 to
**30 runs**. The array was cancelled about 2 minutes after it started, and no run had finished.
* Batch 2: decoder `norm` ∈ {`batch`, `group`} × encoder BN ∈ {`train`, `frozen`}. `none` is
  dropped as the least plausible arm.
* Batch 16: only the default (`batch` + `train`), for the batch-2 vs batch-16 gap.
* The decision rule's batch-16 check ("within 1 pooled SD of the default at batch 16") is run
  afterwards, only for an arm that passes at batch 2 (3 seeds × 2 datasets for that arm).

## Results (mean ± seed SD, 3 seeds; Δ vs the batch-2 default, in mIoU points)

| Dataset | Arm | Val mIoU | Test mIoU | Val bF1 | Val Δ (2 pooled SD) | Test Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Pet | b16 default (batch, encoder BN train) | 0.8209 ± 0.0007 | 0.8217 ± 0.0002 | 0.787 | | |
| Pet | **b2 default** | 0.8077 ± 0.0011 | 0.8093 ± 0.0010 | 0.763 | — | — |
| Pet | b2 batch + frozen encoder BN | 0.8087 ± 0.0014 | 0.8137 ± 0.0013 | 0.769 | +0.10 (0.25) | +0.44 |
| Pet | b2 group + train | 0.8129 ± 0.0005 | 0.8132 ± 0.0012 | 0.765 | **+0.52** (0.17) | +0.39 |
| Pet | b2 group + frozen | 0.8139 ± 0.0004 | 0.8165 ± 0.0010 | 0.772 | **+0.62** (0.17) | +0.72 |
| Sen1Floods11 | b16 default | 0.8971 ± 0.0031 | 0.8914 ± 0.0040 | 0.562 | | |
| Sen1Floods11 | **b2 default** | 0.8816 ± 0.0076 | 0.8939 ± 0.0016 | 0.554 | — | — |
| Sen1Floods11 | b2 batch + frozen encoder BN | 0.9071 ± 0.0019 | 0.9002 ± 0.0030 | 0.584 | **+2.55** (1.11) | +0.63 |
| Sen1Floods11 | b2 group + train | 0.8748 ± 0.0077 | 0.8784 ± 0.0209 | 0.571 | −0.68 (1.54) | −1.55 |
| Sen1Floods11 | b2 group + frozen | 0.9017 ± 0.0018 | 0.8868 ± 0.0011 | 0.601 | +2.01 (1.10) | −0.71 |

* **Small batches do hurt the default.** At batch 2 vs 16, val mIoU is −1.32 on Pet (≫ 2 SD)
  and −1.55 on Sen1Floods11 (> 2 SD, though test is +0.25 there).
* **By the rule, no arm passes on both datasets with test agreeing:**
  * `batch` + frozen encoder BN passes on Sen1Floods11 (+2.55 val, +0.63 test) but not on
    Pet (+0.10).
  * `group` + `train` passes on Pet (+0.52 / +0.39) but not on Sen1Floods11.
  * `group` + frozen passes on Pet and on Sen1Floods11 val, but Sen1Floods11 test disagrees
    (−0.71).
* **Freezing the encoder's BN statistics is the largest single effect on the multispectral
  dataset.** It stabilises seeds (val SD 0.0076 → 0.0019) and, at batch 2, even beats the
  batch-16 default (0.9071 vs 0.8971 val, 0.9002 vs 0.8914 test). Batch 2 also takes 8× more
  optimiser steps, so this is not a pure normalisation effect.
* Throughput at batch 2 is about a third of batch 16 (81–88 vs 246–273 img/s).

## Decision
* **No default changes**, and the pre-registered batch-16 follow-up is not run, since no arm
  passed at batch 2.
* The docs report the numbers. For small-batch training on multispectral or remote-sensing
  inputs, `freeze_encoder_bn=True` is worth trying (it was strongly positive on Sen1Floods11),
  and decoder `norm="group"` is worth trying on natural images (Pet). Both are stated as
  single-dataset results, not recommendations.
* Pawsey cost: 252 units for the 30 runs, plus 81 for the cancelled 72-run array (64 units/GCD-h).
