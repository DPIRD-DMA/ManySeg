# E020: What ManySeg adds to image-level tasks (BigEarthNet v2; pre-registered)

Status: **done** (registered 2026-10-15 before any run; run 2026-10-15/16 at code `7be9202`).
Result: ManySeg matches timm on classification and regression (Q1). Band groups, Sentinel-1
and metadata do not help (Q2–Q4). A segmentation head on the reference maps **helps** the
image classifier (Q5: +2.1 val, +2.3 test macro mAP); post hoc, it looks like regularisation.

## Question
Once ManySeg's classifiers match timm (E019), do the things timm cannot express help an
image-level task? On BigEarthNet v2 (19-class multi-label land cover, Sentinel-2 L2A +
Sentinel-1, 10 countries, dates, pixel reference maps):

* **Q1 parity:** ManySeg S2 classifier vs a plain timm classifier (same data, weights, recipe);
  and the same for class-fraction **regression** (19 outputs).
* **Q2 band groups:** S2 as three native-resolution groups through one shared encoder (E015's
  method; +1.25 on Sen1Floods11, not replicated on PASTIS or CloudSEN12+) vs the 12-band stack.
* **Q3 S1:** adding Sentinel-1 VV/VH as a second input with intermediate fusion (separate
  encoder) vs stacking it as two extra channels vs S2 alone.
* **Q4 metadata:** acquisition day of year and country by FiLM, against the same metadata with
  rows permuted across patches (control).
* **Q5 pixel supervision:** a segmentation head on the reference maps (U-Net decoder) as an
  auxiliary task for the image classifier (which still pools the encoder).

## Data
`manyseg-bench prepare bigearthnet`: the fixed sha1-ordered subset of the official splits,
60,000 train / 15,000 val / 30,000 test patches (of 238k / 122k / 120k; the official metadata
already excludes snow, cloud and shadow). 120×120 at 10 m; 20 m / 60 m bands repeated onto the
10 m grid. Every classification arm (including `seg`) uses the same image labels: the
19 classes present in the mapped reference map, which reproduce the official labels (the
fraction of patches with identical label sets is `label_agreement` in `meta.json`).
Augmentation: h/v flips only (crops would invalidate image labels). Image database: the
community LMDB conversion (see `datasets/bigearthnet.py`); labels and maps are official.

## Protocol
* ResNet-50 (`tv_in1k`), recipe `fastai_like` (1 frozen epoch, encoder LR ×0.1, one-cycle),
  batch 64, 12 epochs, bf16, seeds 0–2, local RTX 4090 (no Pawsey units). Image heads use the
  default init (E019b did not change it). Budget: smoke timing of the frozen first epoch was
  23 s (`s2`), 40 s (`s1s2-fusion`), 46 s (`seg`); about 4.7 h for the 30 runs.
* Arms (`--tag e020-<arm>`):

| arm | provider | dataset | options |
|---|---|---|---|
| `timm-s2` | timm | `bigearthnet` | — |
| `s2` | manyseg | `bigearthnet` | — |
| `groups` | manyseg | `bigearthnet` | `band_groups` (10/20/60 m), shared encoder, intermediate concat |
| `s1s2-stack` | manyseg | `bigearthnet_s1s2` | stacked 14 channels |
| `s1s2-fusion` | manyseg | `bigearthnet_s1s2` | dict inputs, two ResNet-50s, intermediate concat |
| `meta` | manyseg | `bigearthnet_meta` | dict, FiLM (doy cyclic, country categorical) |
| `meta-shuffled` | manyseg | `bigearthnet_meta_shuffled` | as `meta`, rows permuted |
| `seg` | manyseg | `bigearthnet_seg` | U-Net decoder + `seg` head, image head on the encoder |
| `timm-frac` | timm | `bigearthnet_frac` | regression, 19 fractions, MSE |
| `frac` | manyseg | `bigearthnet_frac` | regression, 19 fractions, MSE |

* Metric: val macro average precision (mAP) for classification; RMSE for regression. Test
  (30k), micro mAP, F1 and per-class AP reported.

## Decision rule
* **Q1 parity:** |`s2` − `timm-s2`| ≤ 2 pooled SD of val macro mAP (or ≤ 0.3 points when the
  pooled SD is below 0.15); same for `frac` vs `timm-frac` on RMSE (≤ 2 pooled SD, or ≤ 0.002).
* **Q2–Q5:** an arm *helps* if it beats `s2` by more than 2 pooled SD on val macro mAP and test
  moves the same way. Q4 additionally requires `meta` to beat `meta-shuffled` by 2 pooled SD;
  `meta-shuffled` passing against `s2` would mean the gain is not the information.
* Helpful arms are documented as recommendations for image-level tasks with their cost
  (parameters, throughput). Arms that do not help are reported as negative results.

## Results

All 30 runs completed (4 h 01 min, local 4090, no failures; RAM never below 96 GB available).
Final-epoch (12) metrics, mean ± SD over seeds 0–2. Full tables:
`uv run python experiments/E020_bigearthnet_image_tasks/analyse.py`.

| arm | val mAP macro | test mAP macro | test mAP micro | test F1 macro | params | train img/s |
|---|---|---|---|---|---|---|
| timm-s2 | 0.7344 ± 0.0031 | 0.6632 ± 0.0044 | 0.8384 | 0.6199 | 23.6 M | 2550 |
| s2 | 0.7372 ± 0.0036 | 0.6622 ± 0.0026 | 0.8391 | 0.6238 | 23.6 M | 2504 |
| groups | 0.7347 ± 0.0023 | 0.6611 ± 0.0023 | 0.8384 | 0.6137 | 40.4 M | 609 |
| s1s2-stack | 0.7385 ± 0.0007 | 0.6690 ± 0.0071 | 0.8388 | 0.6272 | 23.6 M | 2368 |
| s1s2-fusion | 0.7354 ± 0.0007 | 0.6646 ± 0.0019 | 0.8377 | 0.6242 | 58.2 M | 1361 |
| meta | 0.7353 ± 0.0038 | 0.6628 ± 0.0022 | 0.8384 | 0.6201 | 23.8 M | 2391 |
| meta-shuffled | 0.7359 ± 0.0030 | 0.6582 ± 0.0007 | 0.8384 | 0.6153 | 23.8 M | 2425 |
| **seg** | **0.7581 ± 0.0010** | **0.6848 ± 0.0016** | **0.8584** | 0.6263 | 32.6 M | 1583 |

Regression (19 class fractions): `frac` val RMSE 0.0821 ± 0.0003, test 0.0836 (R² 0.42);
`timm-frac` 0.0824 ± 0.0007, test 0.0838 (R² 0.43).

* **Q1 parity:** classification `s2` − `timm-s2` = +0.29 val (bar 0.68), −0.10 test: parity.
  Regression `frac` − `timm-frac` = −0.0003 val RMSE (bar 0.0020): parity.
* **Q2 band groups:** −0.25 val (2 SD 0.61): no. At 4× lower training throughput and +17 M
  parameters (the shared encoder carries three per-input BatchNorm sets plus the fusion).
  Across four datasets band grouping cleared its bar only on Sen1Floods11 (E015).
* **Q3 Sentinel-1:** stacked +0.13 val / +0.68 test, separate-encoder fusion −0.18 / +0.24:
  neither passes. S1 adds little to S2 for land-cover classification at 10 m.
* **Q4 metadata:** day of year + country by FiLM −0.19 val; the shuffled control −0.13; meta
  vs shuffled −0.06: no. The fourth dataset where metadata adds nothing (E009, E013, E016).
* **Q5 segmentation head:** +2.09 val (2 SD 0.53), +2.26 test macro mAP, +1.93 test micro
  mAP: **helps**. Per class (test AP), 18 of 19 classes improve; the largest gains are the
  hard ones: moors/heathland +7.3, permanent crops +6.0, industrial +3.4, inland wetlands
  +3.1 (coastal wetlands, 0.3% of patches, −1.1).
* **Why it helps (post hoc, not pre-registered):** every arm except `seg` peaks at epoch 6–8
  and then overfits (final − best val −1.15 to −1.73); `seg` peaks at epoch 10 (−0.11). At epoch 7 `s2` and
  `seg` are tied (0.7506 vs 0.7509 val). So the pixel task is a **regulariser**: it does not
  raise the early peak, it keeps the classifier improving. Against the other arms' best
  epoch (chosen on val, which flatters them), `seg`'s final epoch still leads by +0.5 to +1.0
  val (0.7581 vs 0.7477–0.7527).
* **Cost of `seg`:** +9.0 M decoder parameters and 37% lower training throughput; the
  decoder also runs at inference unless the seg head is dropped
  (`model.select_heads(["logits"])`, added after this experiment, removes it).
* **Caveats:** one encoder (ResNet-50), a 60k / 15k / 30k subset of the official splits,
  12 epochs; the regularisation effect may shrink with early stopping or stronger
  augmentation. The masks are the same reference maps the labels come from, so this needs
  pixel labels that a classification-only dataset may not have.
