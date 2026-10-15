# E013: CloudSEN12+ cloud and shadow masking. Auxiliary layers, metadata, input_cross (pre-registered)

Status: **planned** (registered 2026-10-15 before any training run; a 2-epoch throughput pilot
on 2,000 tiles, val mIoU not used, set the epoch budget).

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
  `scripts/setonix/manyseg_pack.sbatch`. Each pack stages the 191 GB cache once onto node NVMe.
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
