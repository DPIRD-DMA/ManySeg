# E002: ManySeg vs SMP vs fastai U-Net on Oxford Pet

Status: **complete** (2026-10-14). 18 runs, 0 failures. Code: commits `039ce3b`..`fad5523`. The
bench changes made during the runs were RNG- and arithmetic-equivalent for Pet (verified).
Later experiments run from a frozen worktree.

## Hypotheses
* H1: At an identical training recipe, ManySeg's `unet` matches SMP's `Unet`. Same
  architecture (parameters and FLOPs identical) and same encoder weights.
* H2: fastai's DynamicUnet beats both U-Nets because of its architecture and not only because
  of its training recipe.
* H3: The fastai training recipe accounts for a large part of fastai's reported advantage.

## Setup
* Oxford-IIIT Pet trimaps (3 classes), 256² fixed resize. Train 3304 / val 376 (10% of
  trainval by name hash) / test 3669 (official).
* Encoder resnet34. The identical torchvision-V1 weights were used in every run (fingerprint
  `b89adf45880df6fa`).
* 20 epochs, batch 16, bf16 autocast. GPU-resident data with seeded random-resized-crop and
  flip augmentation, identical across providers.
* Recipes: `standard` (AdamW 5e-4, wd 1e-4, warmup + cosine) and `fastai_like` (fastai
  `fine_tune` defaults: frozen first epoch, encoder LR ×0.1, one-cycle, β2 0.99, eps 1e-5, wd 1e-2
  with no decay on norm/bias).
* Seeds 0–2. Hardware: RTX 4090, **shared with other jobs**, so epoch times are indicative only.
* Selection: final epoch. Test was evaluated once per run; it is a fixed comparison, not a
  search.

## Results (mean ± SD over 3 seeds)

| recipe | provider | params (M) | GFLOPs @256² | val mIoU | val boundary F1 | test mIoU | epoch (s, contended) | Welch t vs ManySeg (val / test) |
|---|---|---|---|---|---|---|---|---|
| fastai_like | fastai | 41.22 | 62.8 | 0.8226 ± 0.0013 | 0.7899 ± 0.0013 | **0.8251 ± 0.0005** | 28.7 | +1.1 / **+9.6** |
| fastai_like | ManySeg | 24.44 | 15.7 | 0.8216 ± 0.0009 | 0.7877 ± 0.0040 | 0.8220 ± 0.0003 | 11.1 | |
| fastai_like | SMP | 24.44 | 15.7 | 0.8218 ± 0.0009 | 0.7866 ± 0.0018 | 0.8230 ± 0.0007 | 11.5 | +0.3 / +2.2 |
| standard | fastai | 41.22 | 62.8 | 0.8143 ± 0.0022 | 0.7766 ± 0.0013 | **0.8180 ± 0.0001** | 28.0 | +0.5 / **+6.8** |
| standard | ManySeg | 24.44 | 15.7 | 0.8135 ± 0.0009 | 0.7740 ± 0.0032 | 0.8149 ± 0.0008 | 10.9 | |
| standard | SMP | 24.44 | 15.7 | 0.8134 ± 0.0025 | 0.7734 ± 0.0029 | 0.8155 ± 0.0009 | 11.2 | −0.1 / +0.9 |

Recipe effect (`fastai_like` − `standard`, val): ManySeg +0.0080 (t 10.5), SMP +0.0083 (t 5.4),
fastai +0.0083 (t 5.6). On test: +0.0071, +0.0075 and +0.0071.

Raw records are in `runs/`; `analyse.py` regenerates `results.md`.

## Interpretation
* **H1 holds within this experiment's resolution.** ManySeg and SMP do not differ detectably
  on val. On test SMP is +0.0010 (t 2.2, n=3) with one recipe and +0.0006 (t 0.9) with the
  other. That is not significant, but we keep watching it. The models are identical in
  parameters and FLOPs and use the same encoder weights. The remaining differences are the
  random decoder initialisation stream and the encoder implementation (torchvision vs timm
  ResNet).
* **H3 holds strongly.** The training recipe is worth about +0.8 mIoU points for every
  implementation, about 2.5× the largest architecture effect seen here. Comparisons between
  SMP and fastai that use each library's typical training loop mostly measure the recipe.
* **H2 holds, but the effect is small.** On the larger test split (3,669 images), fastai's
  DynamicUnet is about +0.3 mIoU points ahead under both recipes. That is highly consistent
  across seeds. The validation split (376 images) is too small to resolve it. The gain costs
  1.7× the parameters, 4.0× the FLOPs and about 2.6× the training time per epoch.
* **Methodological lesson.** A 376-image validation split cannot resolve effects of about
  0.3 points on Pet. From E003 onward, every final configuration is also evaluated once on
  test. Selection rules still use val.

## Decisions
* Keep `unet` (SMP-equivalent) as the default reference decoder. Keep `fastai_unet` available
  as a higher-capacity option. It is now a measured Pareto point: +0.3 points for 4× FLOPs.
* Use the `fastai_like` recipe for subsequent Pet experiments (pre-registered rule: higher mean
  val mIoU, 0.8220 vs 0.8137).
* E003 tests whether fastai's advantage comes from capacity or from design, using ManySeg
  decoders with an identical encoder and recipe.
