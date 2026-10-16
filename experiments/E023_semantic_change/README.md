# E023: Semantic change detection: fused vs per-input land-cover heads (pre-registered)

Status: **done** (registered 2026-10-16 before any run; run at code `3b363c1`). Result (SECOND only):
per-input land-cover heads beat land cover from the fused features; sharing them costs nothing.

## Question
Semantic change maps land cover at both dates plus where it changed. ManySeg can predict the
two land-cover maps from the fused features, or with per-input heads (`"input": "t1"` /
`"t2"`) that read each date's own encoder features, optionally sharing one decoder and head
(`"share"`). Does the per-input design help, and does sharing?

## Data
**SECOND** (`second`): 512×512 aerial RGB pairs, 6 land-cover classes labelled where change
happened, from the torchange copy (train 2,968 / test 1,694; val = the sha1 10% of train).
One dataset only: results are single-dataset and reported as such.

## Protocol
* ResNet-34 (`tv_in1k`) U-Net, recipe `fastai_like`, batch 8, seeds 0–2, local RTX 4090.
  Loss: CE(change) + CE(land cover t1) + CE(land cover t2) on changed pixels.
* Arms (`--tag e023-<arm>`):

| arm | encoder / fusion | land-cover heads |
|---|---|---|
| `stack` | both dates stacked (early fusion) | from the fused features |
| `siam-fused` | Siamese, `concat_diff` | from the fused features |
| `siam-perinput-shared` | Siamese, `concat_diff` | per input, one shared decoder + head |
| `siam-perinput-separate` | Siamese, `concat_diff` | per input, separate decoders + heads |

  4 arms × 3 seeds = 12 runs. Epochs: the largest of 30 / 20 / 15 for which the 12 runs fit in
  6 h at the smoke-timed rate (recorded here before any run). **Smoke:** frozen first epoch
  28 s (`stack`) to 53 s (`siam-perinput-separate`); 30 epochs ≈ 7.8 h, 20 epochs ≈ 5.2 h:
  **20 epochs**.
* Metrics (Yang et al. 2021): primary SECOND **Score** = 0.3 mIoU(change / no change) + 0.7
  SeK; SeK, mIoU and change F1 reported.

## Decision rule
* Per-input heads are recommended for semantic change if the better per-input arm beats
  `siam-fused` (same encoder and fusion, land-cover heads on the fused features) by more than
  2 pooled SD of val Score with test agreeing. `stack` is a separate system-level baseline
  (early fusion), reported but not part of the head decision.
* `share` becomes the documented default for per-input change heads if `siam-perinput-shared`
  is within 2 pooled SD of, or better than, `siam-perinput-separate` (fewer parameters, same
  mapping for both dates); otherwise separate heads are documented.
* Single dataset: any recommendation is marked as resting on SECOND only.

## Amendment (2026-10-16, before any E023 run; Codex review)
* The head decision originally compared per-input heads with the better of `stack` and
  `siam-fused`; a `stack` win would then reflect early fusion, not the head source. It now
  compares against `siam-fused` only (the arms differ only in where the land-cover heads read).
* `run.sh` defaulted to 30 epochs; it now defaults to the registered 20.
* "Test agreeing" (here and in every experiment since E015) is a veto only: arms are compared
  and chosen on validation; test can block an adoption but never select between arms.

## Results

All 12 runs completed (local 4090, 20 epochs, no failures; RAM never below 97 GB available).
Final epoch, mean ± SD over seeds 0–2. SECOND Score = 0.3 mIoU_scd + 0.7 SeK.

| arm | val Score | test Score | test SeK | test mIoU_scd | test change F1 | params | GFLOPs (512²) |
|---|---|---|---|---|---|---|---|
| stack | 0.2560 ± 0.0011 | 0.2662 | 0.1052 | 0.6418 | 0.6044 | 24.4 M | 65 |
| siam-fused | 0.3260 ± 0.0044 | 0.3464 | 0.1924 | 0.7056 | 0.7006 | 25.5 M | 105 |
| siam-perinput-shared | **0.3407 ± 0.0028** | **0.3614** | **0.2070** | **0.7217** | **0.7216** | 28.6 M | 153 |
| siam-perinput-separate | 0.3400 ± 0.0043 | 0.3607 | 0.2063 | 0.7212 | 0.7214 | 31.8 M | 153 |

* **Per-input heads (decision): recommended.** `siam-perinput-shared` − `siam-fused` = +1.47 val
  Score (2 SD 0.73), test +1.51; `siam-perinput-separate` +1.40 (2 SD 0.86), test +1.44. Every
  reported metric moves the same way (SeK +1.5, mIoU_scd +1.6, change F1 +2.1 on test).
* **Sharing (decision): `share` is the documented default.** Shared vs separate: +0.08 val
  (2 SD 0.72), test +0.07, with 3.2 M fewer parameters and one mapping for both dates.
* **System-level context (not part of the head decision):** the Siamese encoder is the larger
  step: `siam-fused` − `stack` = +7.0 val Score (2 SD 0.64), test +8.0, consistent with E022.
* **Cost:** per-input heads add two land-cover decoders' compute (153 vs 105 GFLOPs per pair;
  the shared variant shares weights, not compute) and ~46% longer training per epoch.
* **Caveats:** one dataset (SECOND), one encoder (ResNet-34), 20 epochs (scores were still
  rising slowly at the end). Absolute Scores are below published SECOND results, which use
  stronger encoders, longer schedules and test-time augmentation.
