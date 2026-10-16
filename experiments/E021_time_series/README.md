# E021: Time-series inputs (pre-registered)

Status: **planned** (registered 2026-10-16 before any run; timing from 1-epoch smoke runs only,
outside `runs/`).

## Question
ManySeg now takes `{"kind": "timeseries"}` inputs: one 2D encoder on every date, aggregated per
feature level (masked mean, or L-TAE-style temporal attention that starts as the mean). Before
recommending it:

* **Q1:** does a time-series input beat the simple alternative, stacking all dates as channels
  in one input?
* **Q2:** does temporal attention beat the masked mean (which would make it the default)?
* **Context:** how much do several dates add over one date?

## Data
* **PASTIS** (`pastis_ts`): crop-type semantic segmentation, 19 classes, 128×128 at 10 m,
  8 dates per patch (one per eighth of the Nov 2018 – Oct 2019 season, from the E016 cache),
  10 bands, official fold I (train 1,455 / val 482 / test 496 patches). Only 8 of the 38–61
  available dates and no cloud masks, so not comparable to U-TAE's published numbers.
* **AI4Boundaries Sentinel-2** (`ai4boundaries_s2`): field delineation as boundary + distance
  instances (3 heads; primary metric PQ), 256×256 at 10 m, 6 monthly composites (Mar–Aug
  2019), bands B2/B3/B4/B8, official split (5,319 / 1,140 / 1,139). Per-date validity mask.
* Single-date variants use one fixed date: PASTIS early July (index 5 of 8), AI4Boundaries the
  July composite.

## Protocol
* ResNet-34 (`tv_in1k`) U-Net, recipe `fastai_like`, seeds 0–2, local RTX 4090 (no Pawsey).
  PASTIS: 30 epochs, batch 16 (as E016). AI4Boundaries: batch 8; epochs = the largest of
  30 / 20 / 15 / 10 for which the 12 AI4Boundaries runs fit in 8 h at the smoke-timed rate
  (decided and recorded here before any AI4Boundaries run).
* Arms per dataset (`--tag e021-<dataset>-<arm>`):

| arm | input |
|---|---|
| `single` | one date as an ordinary raster |
| `stack` | all dates stacked as channels (T × C) in one input |
| `mean` | time-series input, masked mean per level |
| `attention` | time-series input, temporal attention with acquisition dates |

  2 datasets × 4 arms × 3 seeds = 24 runs. PASTIS smoke: ~10–13 s per frozen epoch for every
  arm (about 2 h for its 12 runs).

## Decision rule
* **Q1:** the time-series input is recommended if `mean` or `attention` beats `stack` by more
  than 2 pooled SD on val (mIoU on PASTIS, PQ on AI4Boundaries) on **both** datasets, and test
  moves the same way. Otherwise the docs say stacking dates as channels is as good on these
  data.
* **Q2:** `attention` becomes the default aggregation if it beats `mean` by more than 2 pooled SD
  on val on both datasets with test agreeing; otherwise `mean` stays the default.
* `single` is context only (no decision).
