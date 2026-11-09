# E059: Do separate encoders help for inputs at different scales? (FLAIR #2: aerial 0.4 m + Sentinel-2 10 m)

Status: **done, positive** (2026-11-09, RTX 3090, torch 2.14.1+cu130, seeds 0–2). Separate
encoders beat stacking for inputs 25× apart in resolution, mostly because stacking hurts.

## Question
E005 (S1 + S2) and E056 (RGB + thermal) found no worthwhile gain from separate encoders over
stacking, but in both the inputs share a grid. Here they don't: aerial imagery at 0.4 m and a
Sentinel-2 time series at 10 m, a 25× gap. Stacking needs S2 upsampled to the aerial grid and
reduced to one date; separate encoders can keep S2 at its own scale and keep its time axis.

## Protocol
* Data: FLAIR #2 (IGN). Aerial patches and labels from FLAIR #1 train (already on the 3090);
  Sentinel-2 L2A super-patches from `flair_sen_train.zip`, fetched for the chosen départements
  only (HTTP range reads of the remote zip, 9.9 GB); patch positions from
  `flair-2_centroids_sp_to_patch.json` (in the FLAIR #2 toy dataset).
* Split by département: 12 of the 40 FLAIR #1 train départements by fixed hash, 8 train
  (D077 D008 D066 D023 D034 D017 D035 D086), 2 val (D009 D029), 2 test (D051 D038). Up to 1,250
  (train) or 1,000 (val, test) patches sampled per département; patches whose S2 crop leaves the
  zone or has fewer than 3 clear dates are dropped. Train 7,417, val 1,932, test 1,463.
* Aerial: 5 bands (R G B NIR elevation), 512 px at 0.2 m, 2×2-averaged to 256 px (0.4 m).
  Labels 1–19 mapped to 0–12 (13–19 merged into "other", trained on, not scored); the label at
  256 px is the top-left of each 2×2 block (0.1 m centre offset).
* Sentinel-2: the 10 × 10 px around the patch centroid (100 m; the patch is 102.4 m, a 2.4%
  footprint mismatch, ~0.1 S2 px at the edges). Dates count as clear when max cloud and snow
  probability in the crop is < 20%. Monthly medians of clear dates (12 months, 83–90% present)
  and an annual median. Alignment check: median correlation between aerial NIR (averaged to
  10 × 10) and S2 B8 is 0.44–0.45 per split; misaligned crops would give about 0.
* Model: ResNet-34 U-Net on the aerial input; ResNet-18 for the S2 encoder where there is one,
  fed the S2 crop upsampled to 64 px. Cross-entropy, AdamW 3e-4 one-cycle, 30 epochs, batch 16,
  bf16, flips and 90° rotations shared by every input of a sample.
* Arms: `aerial` (baseline), `early` (aerial + S2 annual median upsampled to 256 px, 15 channels,
  one encoder), `mid_median` (aerial + S2-median encoder, `same_extent`, concat fusion), `mid_ts`
  (aerial + S2 monthly time-series encoder, ManySeg `timeseries` with `attention`, masked missing
  months, day-of-year dates, `same_extent`, concat fusion).
* Metric: mIoU over the 12 scored classes, from the full confusion matrix (a scored pixel
  predicted "other" counts as a miss), plus per-class IoU.
* Code reviewed by Codex before launch. Fix taken: mIoU first dropped the "other" column too,
  hiding scored pixels predicted as "other". Not fixed (documented above): the 2.4% S2 footprint
  mismatch and top-left label sampling.

## Decision rule
* Screen (seed 0): run seeds 1–2 if the best `mid_*` arm beats `early` by more than 1 mIoU.
* Separate encoders help if `mid_*` − `early` > 2 pooled SD over 3 seeds and val agrees in sign.
* `mid_ts` − `mid_median` separates the time axis from the second encoder itself.

## Results

### Seed 0 (screen)

| arm | test mIoU | val mIoU | params | train min |
|---|---|---|---|---|
| aerial (baseline) | 0.459 | 0.486 | 24.4 M | 23 |
| early | 0.438 | 0.480 | 24.5 M | 23 |
| mid_median | 0.465 | 0.498 | 36.3 M | 29 |
| mid_ts | 0.445 | 0.493 | 36.4 M | 33 |

Test IoU per class (building, pervious, impervious, bare soil, water, coniferous, deciduous,
brushwood, vineyard, herbaceous, agricultural, plowed):

| arm | bld | perv | imp | bare | water | conif | decid | brush | vine | herb | agri | plow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 0.79 | 0.43 | 0.63 | 0.16 | 0.57 | 0.21 | 0.63 | 0.26 | 0.43 | 0.45 | 0.46 | 0.50 |
| early | 0.79 | 0.40 | 0.63 | 0.17 | 0.70 | 0.24 | 0.63 | 0.20 | 0.20 | 0.44 | 0.38 | 0.47 |
| mid_median | 0.79 | 0.43 | 0.67 | 0.15 | 0.70 | 0.26 | 0.64 | 0.27 | 0.31 | 0.43 | 0.42 | 0.52 |
| mid_ts | 0.79 | 0.47 | 0.64 | 0.18 | 0.68 | 0.31 | 0.64 | 0.21 | 0.13 | 0.46 | 0.38 | 0.46 |

* Screen: `mid_median` − `early` = +2.7 test mIoU, above the 1-mIoU screen, so seeds 1–2 run.
* Stacking the upsampled S2 median *hurts* (−2.1 against aerial alone), mostly vineyard and
  agricultural land; giving S2 its own encoder recovers that and gains on water and impervious.
* Against aerial alone the best S2 arm is only +0.6 test / +1.3 val. Vineyard swings widely
  between arms (0.13–0.43), and with two test départements one class moves the mean by ~2
  points, so seed noise is likely large.
* The monthly time series does not beat the annual median at seed 0.

### Seeds 0–2 (mean ± seed SD)

| arm | test mIoU | per seed | val mIoU | per seed |
|---|---|---|---|---|
| aerial (baseline) | 0.466 ± 0.009 | 0.459 / 0.476 / 0.462 | 0.485 ± 0.002 | 0.486 / 0.486 / 0.482 |
| early | 0.443 ± 0.010 | 0.438 / 0.454 / 0.436 | 0.481 ± 0.002 | 0.480 / 0.484 / 0.480 |
| **mid_median** | **0.472 ± 0.014** | 0.465 / 0.463 / 0.488 | 0.498 ± 0.002 | 0.498 / 0.500 / 0.496 |
| mid_ts | 0.460 ± 0.016 | 0.445 / 0.458 / 0.477 | **0.509 ± 0.019** | 0.493 / 0.530 / 0.503 |

Mean test IoU per class over seeds:

| arm | bld | perv | imp | bare | water | conif | decid | brush | vine | herb | agri | plow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 0.79 | 0.43 | 0.64 | 0.17 | 0.65 | 0.20 | 0.63 | 0.28 | 0.42 | 0.45 | 0.44 | 0.50 |
| early | 0.79 | 0.40 | 0.63 | 0.16 | 0.68 | 0.26 | 0.64 | 0.25 | 0.24 | 0.44 | 0.39 | 0.44 |
| mid_median | 0.79 | 0.45 | 0.63 | 0.18 | 0.70 | 0.28 | 0.64 | 0.29 | 0.38 | 0.43 | 0.42 | 0.46 |
| mid_ts | 0.79 | 0.45 | 0.65 | 0.18 | 0.72 | 0.30 | 0.65 | 0.24 | 0.24 | 0.46 | 0.38 | 0.46 |

| comparison | test diff | 2 pooled SD | val diff | 2 pooled SD | verdict |
|---|---|---|---|---|---|
| mid_median − early | +0.029 | 0.024 | +0.017 | 0.004 | **passes** |
| mid_ts − early | +0.017 | 0.026 | +0.027 | 0.027 | does not pass |
| early − aerial | −0.023 | 0.019 | −0.003 | 0.004 | stacking hurts on test |
| mid_median − aerial | +0.006 | 0.023 | +0.013 | 0.004 | val only |
| mid_ts − aerial | −0.006 | 0.026 | +0.024 | 0.027 | no difference |
| mid_ts − mid_median | −0.012 | 0.030 | +0.010 | 0.027 | no difference |

* **Decision: separate encoders beat stacking here** (`mid_median` − `early` = +0.029 test
  against 0.024; val agrees). The first dataset where they do, after E005 and E056, which share a
  grid; here the inputs are 25× apart in resolution.
* **Most of that is stacking doing damage.** Stacking the upsampled S2 median is 0.023 test mIoU
  below aerial alone, mostly vineyard (0.42 → 0.24) and agricultural land. A separate S2 encoder
  avoids the loss but adds little over aerial alone: +0.006 test (inside the noise), +0.013 val.
* S2 helps water (0.65 → 0.68–0.72) and coniferous (0.20 → 0.26–0.30) in every S2 arm.
* **The monthly time series shows no demonstrated improvement over a single median** (−0.012 test,
  +0.010 val, both well inside the noise).
* Practical reading, for this setup (one dataset, two test départements): when one input is far
  coarser than the other, don't stack it onto the fine grid; give it its own encoder. Whether this
  holds for other datasets is untested, and whether S2 is worth adding at all is a smaller,
  noisier gain.

## Caveats
* Two départements per val and test split: one or two classes swing the mean by ~2 points, and
  test and val disagree on `mid_ts`. Seed SD understates the uncertainty from which départements
  were chosen.
* About 7.4k training patches at 0.4 m from 8 départements, ResNet-18 S2 encoder on a 10 × 10
  crop upsampled to 64 px; the published FLAIR #2 gains used the 40 × 40 S2 context around each
  patch, which ManySeg's `same_extent` alignment can't take.
* Parameter counts differ: 24.4 M single encoder against 36.3 M with the S2 encoder.
