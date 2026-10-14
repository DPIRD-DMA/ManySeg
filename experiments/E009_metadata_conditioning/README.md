# E009: Metadata conditioning on Sen1Floods11 (pre-registered)

Status: **planned** (registered 2026-10-14 before any E009 run). The roadmap reserved E006
(decoder Pareto study), E007 and E008 and E010 is taken, so this is E009. Queue:
`experiments/queue3_metadata.sh`.

## Questions
1. Does conditioning on acquisition metadata (FiLM, the default) improve flood segmentation on
   **flood events the model has never seen**?
2. Does FiLM beat the naive early-fusion baseline of metadata as extra input channels
   (`channels` mode)?
3. Shortcut check: is any gain due to lat/lon identifying the event? (Same as 1 with lat/lon
   removed, and the same comparison on the official, event-sharing split.)

## What metadata exists (checked 2026-10-14)
* Per chip: lat/lon of the chip centre from the GeoTIFF tie point and pixel scale (EPSG:4326).
* Per event (`Sen1Floods11_Metadata.geojson`, 12 events, 11 with hand labels): S1 and S2
  acquisition **dates** (no times), orbit direction, relative orbit number, country.
* Not available: acquisition times, sun or view angles, incidence angle, cloud cover. The
  GeoTIFF tags hold only georeferencing and band names.
* Hence the fields are `latlon` (coords), `doy` (cyclic, S2 date) and `orbit` (categorical,
  ascending/descending). Every field except lat/lon is **constant within an event**, and
  lat/lon clusters by event. On the official split, which shares all 10 non-Bolivia events
  between train, val and test, metadata can therefore identify the event. A positive result
  on that split would say nothing about generalisation.

## Protocol
* Data: Sen1Floods11 hand-labelled S1 + S2. **Leave-events-out split** (`--split events:5:F`):
  the official train+val+test chips (431, 10 events) are pooled and split by event. Fold F
  tests on 2 events and validates on 2 others. Across the 5 folds, every event is tested
  exactly once. Bolivia (15 chips, its own event) is also reported for every run.
  Raster normalisation statistics are computed from each fold's training chips only. They are
  recorded per run as `norm_stats`. The official-split arms use `meta.json` (official train).
* Model: ManySeg, `--input-mode dict`, intermediate fusion (concat) of S1 and S2, each with
  resnet34 `tv_in1k` (default `repeat` adapter), U-Net decoder. Recipe `fastai_like`,
  60 epochs, batch 16, 256² random-resized crops with flips, full 512² evaluation, last-epoch
  model. The metadata embedding uses the defaults (`dim` 64, field dropout 0). No tuning.
* Arms:
  | tag | dataset | metadata | split |
  |---|---|---|---|
  | none | `sen1floods11` | none | events:5:F |
  | film | `sen1floods11_meta` | latlon + doy + orbit, FiLM | events:5:F |
  | channels | `sen1floods11_meta` | latlon + doy + orbit, as extra S2 channels | events:5:F |
  | film_nogeo | `sen1floods11_meta_nogeo` | doy + orbit, FiLM | events:5:F |
  | none_official | `sen1floods11` | none | official |
  | film_official | `sen1floods11_meta` | latlon + doy + orbit, FiLM | official |
* `head` mode is not run because Sen1Floods11 has no image-level target.
* Seeds 0, 1 and 2 for every fold. Runs are ordered fold → seed → arm, so a partial queue stays
  balanced.
* Metrics: water IoU (class 1) and mIoU on the held-out-event test split (`test`), on
  `bolivia`, and on val. Params. The primary metric is the test water IoU, averaged over the
  5 folds for each seed.

## Budget
6 arms: 4 × 5 folds × 3 seeds + 2 × 3 seeds = **66 runs**. At E005's speed (about 4–5 min per
run) that is about 5–6 GPU hours. No reruns except for crashes.

## Decision rule
* **Q1 (FiLM helps on unseen events)**: yes only if the mean paired difference (film − none,
  paired by fold and seed, 15 pairs) in test water IoU exceeds 2× its standard error **and** is
  positive in at least 4 of 5 folds. Bolivia must not point the other way by more than its own
  2× SE (indicative only, n = 15 chips).
* **Q2**: the same rule for film − channels.
* **Q3 (shortcut)**: lat/lon is treated as a shortcut if film_nogeo ≥ film on held-out events
  (within 1 SE), or if the official-split gain (film_official − none_official) exceeds the
  held-out-event gain by more than 2× the pooled SE. In either case, the docs will recommend
  against lat/lon conditioning without geographic splits.
* Expected outcome, stated in advance: with only 8 training events, doy and orbit take 8
  distinct values in training. A held-out event's metadata is therefore an extrapolation, and
  we expect no gain (or a small loss) on held-out events but a gain on the official split. A
  null result here means "Sen1Floods11 is too metadata-poor to test conditioning". It does not
  mean "conditioning does not work". The follow-up then uses a metadata-rich dataset (see
  below).
* Stopping: abort only for a systemic failure (the preflight fails, or more than 3 FAILED lines
  in fold 0).

## Follow-up dataset (proposed, not downloaded)
FLAIR #1 (IGN, French aerial imagery at 0.2 m, 13-class land cover, Etalab 2.0 licence). Every
patch ships metadata: acquisition date and time, camera, flight altitude and patch-centroid
coordinates. The official test set consists of held-out départements (a geographic split), and
the FLAIR baseline already conditions a U-Net on that metadata. A small toy subset exists for a
pipeline check. Needs user approval for the download size.
