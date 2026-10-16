# E021: Time-series inputs (pre-registered)

Status: **done** (registered 2026-10-16 before any run; PASTIS run at `7b3a250`, AI4Boundaries at
`7b3a250` with the cache prepared after the mask-name fix). Result: several dates help a lot on
both datasets; the time-series input does **not** beat stacking dates as channels on both
(better on AI4Boundaries, worse on PASTIS); attention does not beat mean.

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
  **AI4Boundaries smoke (2026-10-16, idle GPU):** frozen first epoch 25 s (`single`), 27 s
  (`stack`), 35 s (`mean`), 40 s (`attention`); with ~1.6× for full fine-tuning plus ~25 s of
  validation (instance post-processing) per epoch, ~76 s per epoch on average, so 30 epochs ×
  12 runs ≈ 7.7 h: **30 epochs**.
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

## Results

All 24 runs completed (local 4090, no failures). Final epoch, mean ± SD over seeds 0–2. Full
tables: `uv run python experiments/E021_time_series/analyse.py`.

| arm | PASTIS val mIoU | PASTIS test | AI4B val PQ | AI4B test | train img/s (PASTIS / AI4B) |
|---|---|---|---|---|---|
| single | 0.3261 ± 0.0020 | 0.3251 | 0.0990 ± 0.0047 | 0.1001 | 814 / 302 |
| stack | **0.4123 ± 0.0035** | **0.3984** | 0.1550 ± 0.0065 | 0.1564 | 488 / 275 |
| mean | 0.3694 ± 0.0098 | 0.3717 | 0.1713 ± 0.0101 | 0.1738 | 410 / 182 |
| attention | 0.3759 ± 0.0026 | 0.3786 | **0.1782 ± 0.0023** | **0.1802** | 356 / 153 |

* **Q1 (time series vs dates as channels): not met.** On AI4Boundaries `attention` beats `stack`
  (+2.33 val PQ, 2 SD 0.97; test +2.38) and `mean` nearly does (+1.64, 2 SD 1.70). On PASTIS
  `stack` beats both (mean −4.29, 2 SD 1.47; attention −3.65, 2 SD 0.61; test −2.7 / −2.0).
  The time-series input is not recommended over stacking in general.
* **Q2 (attention vs mean): not met** (PASTIS +0.65, 2 SD 1.43; AI4B +0.69, 2 SD 1.47). `mean`
  stays the default aggregation.
* **Context:** multiple dates help on both datasets (stack − single: PASTIS +8.6 val mIoU,
  AI4B +5.6 val PQ).
* **Reading (post hoc):** stacking lets the first convolution compare dates per pixel; the
  time-series encoder only combines dates after each level. PASTIS crop types depend on
  spectral change between specific dates (early cross-date cues), field boundaries on
  AI4Boundaries are spatial structure seen more clearly by encoding each month with
  pretrained RGB-like filters. Neither explanation was tested.
* **Docs:** use several dates; stacking dates as channels is the cheaper strong baseline (one
  encoder pass); the time-series input is worth trying (attention was best on AI4Boundaries)
  and is the option when dates vary in number, are missing, or come with acquisition dates.
* **Caveats:** ResNet-34 only; PASTIS uses 8 of 38–61 dates; AI4Boundaries composites are
  regular monthly, so no missing-date masking was exercised on real data.
