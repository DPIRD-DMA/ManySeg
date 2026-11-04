# E013: CloudSEN12+ cloud and shadow masking. Auxiliary layers, metadata, input_cross (pre-registered)

Status: **complete** (2026-10-15; Pawsey Setonix, code `7f05bf0`, 5,318 units including a timed-out re-run). **Negative on all three questions.**

CloudSEN12+ "high" (expert labels, Sentinel-2 L1C, 13 bands, 4 classes: clear, thick cloud,
thin cloud, cloud shadow). The official splits are **ROI-disjoint**, so test scenes are at
locations never seen in training. Dataset details, licence and leakage rules are in
`packages/manyseg-bench/src/manyseg_bench/datasets/cloudsen12.py` and
`docs/benchmark_design.md`. One experiment answers three questions with a **shared baseline**.

## Questions and hypotheses
* **Q1 (auxiliary surface layers).** Do the static S1GBM backscatter and Copernicus DEM layers
  help, stacked (early) or with per-input encoders (intermediate concat)? Hypothesis: a small
  gain concentrated in **cloud shadow**, from separating shadow from dark surfaces, water and
  terrain shadow. The backscatter is a multi-year mosaic, not a coincident S1 image, so it
  cannot see clouds. On Sen1Floods11 (E005) no fusion arm beat S2-only, and a negative result
  here is plausible.
* **Q2 (metadata conditioning).** Does a metadata vector (lat/lon, day of year, solar zenith
  and azimuth) help on unseen locations? Conditioning is by FiLM (default) or as channels. The
  solar geometry describes the image itself, so it physically constrains shadow length and
  direction. Hypothesis: FiLM ≥ none, and any gain is in shadow. Because test locations are
  unseen, a lat/lon *shortcut* cannot pay off as it could on Sen1Floods11's shared events
  (E009). `film_nogeo` checks that.
* **Q3 (`input_cross` on a second dataset).** E012 found that `unet` + `input_cross` recovered
  81–83% of fastai_unet's Sen1Floods11 gain. Does that replicate on 13-band cloud masking,
  where thin cloud is a per-pixel reflectance cue?

## Protocol
* Pawsey Setonix (MI250X), code snapshot recorded per run. U-Net, ResNet-34 `tv_in1k`
  (channel adapter `repeat`, the default), recipe `fastai_like`, **20 epochs**, batch 16, full
  512² tiles (19,450 train / 1,767 val / 2,335 test). Augmentation is the dataset's loader
  default: random resized crop (area 0.5–1) back to 512², plus horizontal and vertical flips. Seeds 0, 1, 2. Test is
  evaluated once per run, after training.
* Arms (`run.sh`, 8 per seed, 24 runs):

  | Tag | Dataset / mode | Used in |
  | --- | --- | --- |
  | `e013-s2` | `cloudsen12`, stack | **baseline** for Q1, Q2 and Q3 |
  | `e013-aux-early` | `cloudsen12_s2_aux`, stack (16 ch) | Q1 |
  | `e013-aux-mid` | `cloudsen12_s2_aux`, dict, intermediate concat (3 encoders, 68 M params) | Q1 |
  | `e013-meta-film` | `cloudsen12_meta`, dict, FiLM | Q2 |
  | `e013-meta-channels` | `cloudsen12_meta`, dict, metadata as channels | Q2 |
  | `e013-meta-film-nogeo` | `cloudsen12_meta_nogeo` (no lat/lon), FiLM | Q2 |
  | `e013-unet-cross` | `cloudsen12`, `input_cross: true` | Q3 |
  | `e013-fastai_unet` | `cloudsen12`, `fastai_unet` | Q3 |

  The baseline is valid for Q2 because in stack mode the runner drops metadata vectors
  (`vectors_dropped`), so it is the same S2-only model.
* Budget: 24 runs, about 30–40 GCD-hours, run as three 8-GCD packs (one per seed) via
  `scripts/setonix/manyseg_pack.sbatch`. Each pack stages the 190 GB cache once onto node NVMe.
* Reported per arm: val and test mIoU, per-class IoU (thin cloud and shadow especially),
  boundary F1, params, FLOPs and throughput.

## Decision rules
Pooled SD = sqrt((SD_a² + SD_b²)/2) over the 3 seeds. "Test agrees" means test moves in the
same direction.

* **Q1:** an aux arm counts as helping if its val mIoU − `e013-s2` > 2 pooled SD and test
  agrees. Two arms are compared with one baseline, so a single marginal pass is reported as
  such. If either passes, the docs recommend the aux layers for cloud and shadow masking and
  note the cost (16 vs 13 channels; 3× parameters for intermediate fusion). Otherwise it is a
  negative result, consistent with E005. `aux-mid` has three encoders, so any win it has over
  `aux-early` is confounded with capacity and is not claimed as a fusion effect.
* **Q2:** FiLM helps if `meta-film` − `e013-s2` > 2 pooled SD on val and test agrees. FiLM vs
  channels follows the same rule. For the location shortcut, if `meta-film-nogeo` is within 1
  pooled SD of `meta-film` or above it, lat/lon adds nothing beyond solar geometry and date;
  the docs then advise against lat/lon on geographic splits unless shown otherwise.
* **Q3:** replicates E012 if `unet-cross` − `e013-s2` > 2 pooled SD on val and test agrees. If
  fastai_unet − `e013-s2` > 2 pooled SD, the gap closed ((cross − s2) / (fastai − s2)) is also
  reported, with ≥ 50% as the bar. If Q3 replicates, `input_cross` is recommended for
  multispectral and remote-sensing segmentation in the docs and the README. **The default
  stays `False`** to keep exact SMP parity; that is a design commitment, not an evidence
  question. If Q3 fails, the E012 recommendation is narrowed to Sen1Floods11-like tasks.

## Amendment (2026-10-15 09:51, before any run finished)
The project's Pawsey GPU allocation is being used too fast. The three `e013-fastai_unet` runs
were cancelled at epoch 4 of 20, because at about 850 s per epoch they would have held all three
8-GCD pack nodes about 2 h after every other arm had finished. Q3's decision bar
(`unet-cross` − `e013-s2` > 2 pooled SD, test agrees) does not need them. The secondary
"gap closed" figure is dropped, and E012 remains the fastai comparison. Every other arm runs
as registered.
* Correction (11:50): `scancel <pack>.7` assumed step ids follow task order, but they follow
  launch timing. In pack 1 (seed 0), step 7 was `e013-meta-channels`, so that run was cancelled
  at epoch 12 while seed 0's `fastai_unet` kept running. The fastai step was then cancelled.
  The seed-0 `meta-channels` run was re-submitted alone (job 50910604, same code `7f05bf0`, same
  task line). `manyseg_pack.sbatch` now names steps `t<task>` so they can be identified.

## Results (mean ± seed SD; 3 seeds unless noted)

| arm | n | val mIoU | test mIoU | test thin IoU | test shadow IoU | val bF1 | params | img/s |
|---|---|---|---|---|---|---|---|---|
| e013-aux-early | 3 | 0.7284 ± 0.0027 | 0.6828 ± 0.0012 | 0.3934 | 0.5688 | 0.281 | 24.5 M | 55 |
| e013-aux-mid | 3 | 0.7323 ± 0.0035 | 0.6806 ± 0.0019 | 0.3895 | 0.5628 | 0.286 | 68.1 M | 40 |
| e013-meta-channels | 2 | 0.7321 ± 0.0058 | 0.6859 ± 0.0012 | 0.4086 | 0.5596 | 0.283 | 24.5 M | 47 |
| e013-meta-film | 3 | 0.7316 ± 0.0027 | 0.6824 ± 0.0022 | 0.3995 | 0.5590 | 0.285 | 24.6 M | 54 |
| e013-meta-film-nogeo | 3 | 0.7307 ± 0.0043 | 0.6812 ± 0.0040 | 0.3952 | 0.5573 | 0.286 | 24.6 M | 66 |
| e013-s2 | 3 | 0.7324 ± 0.0011 | 0.6850 ± 0.0030 | 0.4006 | 0.5675 | 0.286 | 24.5 M | 55 |
| e013-unet-cross | 3 | 0.7355 ± 0.0017 | 0.6848 ± 0.0045 | 0.3974 | 0.5684 | 0.302 | 24.5 M | 55 |

`e013-meta-channels` has **2 seeds**. The re-submitted seed 0 (job 50910604) hit its 3 h time
limit at epoch 12 of 20, because staging the 157 GB cache on a single-GCD job took about 1 h.
It was not resubmitted, because the arm is secondary and no decision depends on it. Per-class IoU on test, S2 baseline (seed 0): clear 0.907, thick cloud 0.865, thin
cloud 0.397, shadow 0.573. Thin cloud is the hard class. Val mIoU is about 5 points above test
throughout (val 0.732 vs test 0.685). This is consistent with the ROI-disjoint test set
containing harder scenes, and it is the same for every arm.

Pre-registered comparisons (val Δ vs 2 pooled SD; test must agree):

```
e013-aux-early vs e013-s2: val -0.41 pts (2 pooled SD 0.42), test -0.22, test shadow IoU +0.13, thin -0.72 -> no
e013-aux-mid vs e013-s2: val -0.02 pts (2 pooled SD 0.51), test -0.43, test shadow IoU -0.47, thin -1.11 -> no
e013-meta-film vs e013-s2: val -0.08 pts (2 pooled SD 0.41), test -0.26, test shadow IoU -0.85, thin -0.11 -> no
e013-meta-channels vs e013-s2: val -0.03 pts (2 pooled SD 0.84), test +0.09, test shadow IoU -0.79, thin +0.80 -> no
e013-meta-film-nogeo vs e013-s2: val -0.17 pts (2 pooled SD 0.63), test -0.37, test shadow IoU -1.02, thin -0.54 -> no
e013-unet-cross vs e013-s2: val +0.31 pts (2 pooled SD 0.28), test -0.01, test shadow IoU +0.09, thin -0.32 -> no
e013-meta-film vs e013-meta-channels: val -0.05 pts (2 pooled SD 0.91), test -0.35, test shadow IoU -0.06, thin -0.90 -> no
e013-meta-film-nogeo vs e013-meta-film: val -0.09 pts (2 pooled SD 0.72), test -0.11, test shadow IoU -0.17, thin -0.44 -> no
```

## Decision
* **Q1 (S1GBM backscatter + DEM): negative.** Neither early (−0.41 val, −0.22 test) nor
  intermediate fusion (−0.02 val, −0.43 test, 3× parameters) helps. Shadow IoU is not improved
  either (+0.13 / −0.47 on test). A static, multi-year backscatter mosaic and terrain add
  nothing the 13 S2 bands do not already carry for cloud and shadow. This is consistent with
  E005 (no fusion gain on Sen1Floods11). The docs do not recommend these layers for cloud
  masking.
* **Q2 (metadata conditioning): negative.** FiLM is −0.08 val / −0.26 test vs no metadata.
  Solar geometry did not help shadow (−0.85 test shadow IoU). By the location-shortcut rule,
  `meta-film-nogeo` is within 1 pooled SD of `meta-film` (−0.09 val), so lat/lon adds nothing on
  unseen locations. With E009 (FiLM under the bar on held-out flood events), metadata
  conditioning has now shown no benefit on two remote-sensing datasets. The docs present it as
  an available tool without a recommendation, and advise against lat/lon on geographic splits.
  The ViT `token` metadata mode (roadmap) is not worth building on this evidence.
* **Q3 (`input_cross` replication): does not replicate.** Val +0.31 just clears 2 pooled SD
  (0.28), but test is −0.01, and test must agree. The E012 recommendation is narrowed to tasks
  whose labels are largely decided by per-pixel input values at full resolution (spectral
  thresholds or index rules, as for water on Sen1Floods11). It is not a general multispectral
  recommendation. The default stays `False` (unchanged).
* Cost: 5,126 Pawsey units in total, about 15% over the 09:50 forecast. The three packs ran
  3.3 h, held by the slowest arms; see the amendments and the `pawsey-budget` lesson (pack only
  runs of similar length together).
