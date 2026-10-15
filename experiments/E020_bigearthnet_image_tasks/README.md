# E020: What ManySeg adds to image-level tasks (BigEarthNet v2; pre-registered)

Status: **planned** (registered 2026-10-15 before any run; timing from a 1-epoch smoke run of
three arms only, outside `runs/`).

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
