# E016: Acquisition date and S2 band groups for single-date crop types (PASTIS)

Status: **planned** (drafted and reviewed 2026-10-15, registered before any run). Review decisions: the date rule stays as written (no cloud-aware choice); the per-month analysis is dropped (see Protocol).

E009 (Sen1Floods11) and E013 (CloudSEN12+) found no benefit from metadata conditioning. On
both, the metadata was largely redundant with the image: a flood or a cloud looks the same in
any season. Single-date crop-type mapping is a case where the date should carry real signal.
The label is one crop per season, but each image shows one moment of it. Bare soil in
November could be about to become winter wheat; in May it means a spring crop (maize,
sunflower, soybean) that has not emerged yet. A green canopy in April could be wheat or
barley; in August it is more likely maize. If conditioning helps anywhere, it should help here.

The same runs also replicate E015's band-grouping result (Q2) on a second dataset, sharing the
no-metadata baseline.

## Data
`pastis` (`packages/manyseg-bench/src/manyseg_bench/datasets/pastis.py`,
`docs/benchmark_design.md`): PASTIS (Garnot & Landrieu, ICCV 2021, CC BY 4.0), 2,433
patches of 128×128 px at 10 m from four Sentinel-2 tiles in France, 10 L2A bands (B2, B3,
B4, B5, B6, B7, B8, B8A, B11, B12; the 20 m bands were bilinearly upsampled by the authors).
Each patch gives **8 single-date samples**: the acquisition nearest the centre of each eighth
of the 365-day season window 2018-11-01..2019-10-31, taken from within that eighth. 19 classes (background + 18 crops);
void (out-of-nomenclature or mostly-outside parcels) is ignored. Official fold I: train
folds 1–3, val fold 4, test fold 5. All dates of a patch stay in its fold.

Prepared cache (2026-10-15, 6.3 GiB float16, memory-mapped):

| split | patches | samples | T30UXV / T31TFJ / T31TFM / T32ULU patches | void px |
| --- | --- | --- | --- | --- |
| train | 1,455 | 11,640 | 297 / 365 / 452 / 341 | 9.4% |
| val | 482 | 3,856 | 111 / 118 / 131 / 122 | 9.8% |
| test | 496 | 3,968 | 123 / 140 / 140 / 93 | 10.2% |

Labelled-pixel class fractions (train): background 0.441, meadow 0.201, corn 0.099, soft
winter wheat 0.074, grapevine 0.032, winter barley 0.022, rapeseed 0.019, leguminous fodder
0.017, soybeans 0.013, durum wheat 0.012, orchard 0.012, fruits/vegetables/flowers 0.011,
sunflower 0.010, beet 0.009, triticale 0.008, spring barley 0.007, mixed cereal 0.005,
sorghum 0.004, potatoes 0.003. Val and test are within ±0.015 of these. mIoU will be driven by
many rare classes (13 of 19 below 2% of pixels).

Points the reviewer should know:
* **The date identifies the tile.** Each tile has its own acquisition calendar (38–61 dates),
  and every patch of a tile shares it, so all patches of a tile get the same 8 dates
  (e.g. T30UXV 2018-11-23, 2019-01-17, 02-21, 04-12, 05-27, 07-11, 08-25, 10-09; the other
  tiles' dates are 1–14 days away). The exact doy therefore reveals the region and its crop
  mix. Q1c controls for this.
* **Clouds.** THEIA only dropped acquisitions with > 90% tile cloud cover. 15% of samples
  have patch-mean blue reflectance > 0.2 (mostly cloud, some snow), concentrated on a few
  dates: T30UXV 2019-10-09 (65% of its patches), T32ULU 2019-01-15 (67%), T31TFJ
  2019-02-19 (63%), T30UXV 2019-02-21 (46%). These dates are the same for every arm. The
  doy can tell the model "this date tends to be cloudy" for a tile; this is a property of the
  date, not a label leak. A cloud-aware date choice (lowest patch-level blue reflectance within
  each eighth of the season; input-derived, deterministic) is a possible change **for the
  reviewer to decide before running**; it would also break the tile-wide shared dates.
  **Review decision: keep the fixed rule.** Cloud cover is realistic and identical for every
  arm, and a cloud-aware choice would break the shared per-tile calendars that Q1c relies on.
* 35 S2 series in the release have no metadata entry and no annotation; they are skipped.

## Questions and hypotheses
**Q1 — acquisition date (metadata conditioning).** Baseline `e016-none`.
* **Q1a (main).** Does conditioning on day of year (FiLM) improve mIoU over the same model
  without metadata? Hypothesis: yes, with the gain concentrated in crops that one image
  confuses: winter cereals (soft/durum wheat, barley, triticale, mixed cereal) and summer
  crops (corn, sunflower, soybeans, sorghum).
* **Q1b (control: is it the date?).** `pastis_meta_shuffled` permutes the doy column across
  all samples (fixed seed). It keeps the marginal date distribution and breaks the date ↔
  image link. It must **not** show the gain.
* **Q1c (control: tile identity).** `pastis_meta_tileshuffled` permutes doy only among samples
  of the same tile. It keeps the tile information (see above) and breaks the seasonal link.
  If the Q1a gain is phenological, this arm should lose most of it. This arm is an addition to
  the original brief, made after the date lists turned out to be per tile.
* **Q1d (secondary).** FiLM vs metadata as broadcast input channels (`channels`). E009 found
  FiLM better (+1.3 points, 2.9 SE).

**Q2 — Sentinel-2 band groups (E015 replication).** E015 (Sen1Floods11) found that splitting
the S2 bands into resolution groups through **one shared encoder** (per-input BatchNorm, the
library default for `shared:` encoders since commit `0a4025d`) beat the stacked bands by
+1.25 val / +1.29 test mIoU, and that native resolution itself added nothing. Does that
replicate on PASTIS? `e016-groups-shared`: 10 m group B2, B3, B4, B8 (indices 0, 1, 2, 6,
factor 1) and 20 m group B5, B6, B7, B8A, B11, B12 (indices 3, 4, 5, 7, 8, 9, factor 2:
block-averaged back to 20 m), the 20 m input through the 10 m input's encoder
(`"band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}}`), intermediate concat fusion,
no metadata. The band order is wavelength order (confirmed by the dataset author in
pastis-benchmark issues #17 and #19), and checked on the data: horizontal lag-1
autocorrelation is 0.93–0.95 for indices 0, 1, 2, 6 and 0.98 for the six upsampled 20 m bands.
Hypothesis: a gain of similar size to E015's.

## Protocol
* Local RTX 4090 (shared workstation: contention is recorded per run and timings are labelled
  as contended where they were). No Pawsey.
* U-Net, ResNet-34 `tv_in1k` (default `repeat` channel adapter for 10 bands), recipe
  `fastai_like`, **30 epochs**, batch 16, last-epoch model, no tuning. Augmentation: the
  dataset loader default, random resized crop (area 0.5–1) back to 128², horizontal and
  vertical flips (the doy vector is not augmented). Seeds 0, 1, 2. Test is evaluated once per
  run after training (`--evaluate-test`). Test never selects a model, an epoch or a setting; the decision rules use it only to confirm a val result ("test agrees"), as in every experiment here.
* Arms (`run.sh`; 6 per seed, 18 runs, ordered seed → arm):

  | tag | dataset / mode | question |
  | --- | --- | --- |
  | `e016-none` | `pastis`, stack, no metadata | **baseline** for Q1 and Q2 |
  | `e016-film` | `pastis_meta`, dict, doy (cyclic), FiLM | Q1a |
  | `e016-channels` | `pastis_meta`, dict, doy as 2 extra input channels | Q1d |
  | `e016-film-shuffled` | `pastis_meta_shuffled`, dict, FiLM | Q1b |
  | `e016-film-tileshuffled` | `pastis_meta_tileshuffled`, dict, FiLM | Q1c |
  | `e016-groups-shared` | `pastis`, stack split into 10 m / 20 m groups, shared encoder | Q2 |

  The metadata embedding uses the defaults (`dim` 64, field dropout 0).
* **Run time estimate: about 6 min per run uncontended (range 5–10 min), so 30 epochs.**
  Basis: past 4090 runs of the same model and recipe (E008: Oxford Pet, 3,304 images at 256²,
  20 epochs in 117–119 s, about 560 img/s at 256² including validation). A 128² sample costs
  a quarter of the FLOPs (4.17 GFLOPs forward, measured), but at batch 16 per-step overhead
  dominates, so we assume 1,000–1,500 img/s rather than 4×. An epoch is 11,640 training
  samples plus 3,856 validation samples, about 10–14 s, so 30 epochs take 5–7 min plus
  about 30 s for setup and the test pass. The memory-mapped loader is not the limit: on CPU it
  fetches, normalises and augments 1,860 img/s from the page-cached 3.8 GB training array
  (augmentation runs on the GPU in real runs). The CPU smoke (`--subset 512`, 1 epoch, 8
  threads, encoder frozen in epoch 1 by `fastai_like`) took 12–13 s per epoch (about 80
  img/s for train + val); it checks that the pipeline runs and is not a GPU estimate. The
  groups arm runs the shared encoder twice (10 m at 128², 20 m at 64²), about 1.25× the
  encoder FLOPs. **Check:** if seed 0 of `e016-none` takes more than 10 min uncontended, the
  epoch count is amended here before any further run or any result is read.
* Reported per arm: val and test mIoU (19 classes, void ignored), per-class IoU, boundary F1,
  params, FLOPs, throughput; for Q1, test per-class IoU deltas for the winter-cereal and
  summer-crop groups named in Q1a. (The draft's per-month val mIoU analysis is dropped at review: run records hold only
  aggregate metrics, and it would need per-sample evaluation. The per-class deltas cover the
  question.)

## Decision rules
Pooled SD = sqrt((SD_a² + SD_b²)/2) over the 3 seeds. "Test agrees" means test mIoU moves in
the same direction. Decisions use val mIoU; test is confirmatory.

* **Q1a:** FiLM helps if `film` − `none` > 2 pooled SD on val and test agrees.
* **Q1b:** the gain is attributed to the date only if Q1a passes **and** `film-shuffled` −
  `none` is **not** > 2 pooled SD on val. If the shuffled arm also gains, the Q1a gain is not
  evidence for date information (it would be a parameter or regularisation effect) and is
  reported that way.
* **Q1c:** if Q1a and Q1b pass, the gain is called phenological if `film` −
  `film-tileshuffled` > 2 pooled SD on val and test agrees. If `film-tileshuffled` keeps at
  least half of `film`'s gain over `none`, that part is attributed to tile (regional)
  identity, and the docs say so.
* **Q1d (secondary):** FiLM vs channels by the same 2-pooled-SD rule. Reported; no default
  changes unless it contradicts E009.
* **Q2:** band groups help if `groups-shared` − `none` > 2 pooled SD on val and test agrees.
  If so, E015's recommendation (group S2 bands by resolution, shared encoder) holds on two
  datasets and the docs say so. If not, it is narrowed to Sen1Floods11-like data. Q2 is
  ruled separately from Q1. It shares the baseline, so a lucky or unlucky baseline seed would
  move both questions; this is stated in the results.
* **Consequences for Q1.** If Q1a–Q1c pass, `docs/api_design.md` changes its "metadata has
  not shown a benefit" note to: helps when the metadata carries information the image lacks
  (date for single-date crop typing), with E016 as the example. If Q1a fails, PASTIS joins
  E009 and E013 as a negative result.
* **Expected outcome, stated in advance:** a FiLM gain of 1–3 mIoU points, absent in the
  shuffled control and mostly but not fully kept by the tile-shuffled control. The four tiles'
  crop mixes differ a lot, so tile identity alone may carry some of it. For Q2, a gain smaller
  than E015's, because only one group is split off and PASTIS's 20 m bands are bilinear
  upsamplings (E015 also had a 60 m group).
* Stopping: abort only for a systemic failure (more than 2 FAILED lines) or the time check
  above.

## Budget
18 runs × about 6 min ≈ 1.8 h of 4090 time uncontended (up to about 3 h contended), run
sequentially with `bash experiments/E016_pastis_date_metadata/run.sh` from the repo root. No
reruns except for crashes.
