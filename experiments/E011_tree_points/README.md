# E011: Tree localisation and counting heads on OAM-TCD (pre-registered)

Status: **complete** (2026-10-14; 9 runs on Pawsey Setonix, AMD MI250X; hm_mse/density code `e69f564`, hm_focal `cfb0548` after the amendment below). Dataset approved by the maintainer:
OAM-TCD CC BY 4.0 subset, stored at 1024² (20 cm). Trees are points (polygon centroids) and
canopy groups are ignore regions.

## Questions
1. Which centre-heatmap formulation localises trees better: Gaussian MSE regression or
   CenterNet focal loss on logits?
2. Is a density head competitive for counting (MAE) with counting heatmap peaks?

## Protocol
* ManySeg U-Net, resnet34 `tv_in1k`, recipe `fastai_like`, 30 epochs, batch 8, 512² native-
  resolution crops (random-resized, area 0.2–0.3 of the 1024² tile) with flips. Evaluation on
  full 1024² tiles. σ = 2 px, matching radius 5 px (1 m), peak threshold 0.5 (fixed in advance).
* Splits: train = official folds 1–4; val = fold 0 (tiles grouped by source image); test =
  official test, evaluated once per final run.
* Arms (3 seeds each):
  | id | heads | loss | primary metric |
  |---|---|---|---|
  | hm_mse | heatmap (regression) | MSE | point F1 @ 1 m |
  | hm_focal | heatmap (segmentation) | penalty-reduced focal | point F1 @ 1 m |
  | density | density | MSE ×100 | count MAE |
  Heatmap arms also report the count MAE from their number of peaks.
* Budget: 9 runs, about 3 h on the shared 4090.

## Decision rule
The heatmap preset default switches from `regression` to `segmentation` only if focal beats MSE
on val F1 by more than 2 pooled seed-SDs and test points the same way. For counting, we
recommend density only if its MAE beats the better heatmap arm's peak-count MAE by the same
rule.

## Amendment (2026-10-14 15:05, before any E011 result)
The first epoch of the `hm_focal` arm showed focal losses around 10³, versus about 0.2 for
`hm_mse`. The heatmap head started at bias 0 (p = 0.5 everywhere), whereas the CenterNet
focal-loss formulation assumes an initial prior of p = 0.1 (bias −2.19). Without it, the
negative pixels swamp the loss. That would compare loss formulations with a required
initialisation missing from one arm. Fix (`point_heads(heatmap="segmentation")` now sets
`bias_init = logit(0.1)`; tested). Only the 3 `hm_focal` runs were cancelled and resubmitted;
the `hm_mse` and `density` runs are unchanged (same protocol, same code apart from this preset).
Platform: Pawsey Setonix MI250X. Data are staged to node-local NVMe.

## Results (mean ± SD, 3 seeds; matching radius 1 m, peak threshold 0.5)

| arm | val point F1 | val precision | val recall | val count MAE | test F1 | test count MAE |
|---|---|---|---|---|---|---|
| hm_mse | **0.2995 ± 0.0230** | 0.8363 ± 0.0139 | 0.1827 ± 0.0179 | 38.26 ± 1.12 | **0.4006 ± 0.0244** | 40.97 ± 1.64 |
| hm_focal | 0.2776 ± 0.0166 | 0.8871 ± 0.0084 | 0.1647 ± 0.0119 | 40.01 ± 0.62 | 0.3078 ± 0.0160 | 46.90 ± 0.77 |
| density | (no peaks) | | | **20.02 ± 0.39** | | **19.40 ± 0.37** |

Mean trees per val tile: about 49.

* focal − MSE, val F1: −0.022 (−1.1 pooled SD); test −0.093.
* density MAE − hm_mse peak-count MAE, val: −18.2 (−21.8 pooled SD); test −21.6.

## Interpretation
* **Heatmap loss: MSE regression stays the default.** Focal is not better on val and clearly
  worse on test, even with the CenterNet prior initialisation (see the amendment).
* **Counting: use the density head.** Its count MAE is about half that of counting heatmap
  peaks, on val and test. Counting by integrating density is robust where peak-picking
  under-detects.
* **Recall is the weakness of both heatmap arms** (0.16–0.18 at precision 0.84–0.89). The
  pre-registered peak threshold of 0.5 is too strict, because MSE-regressed Gaussian peaks
  rarely reach 1. That is a calibration problem, not evidence against the heads. A
  threshold chosen on val (exploratory, reported separately) is the follow-up. These runs did
  not save models, so the sweep needs a rerun with `--save-model`.
* Absolute error remains large (density MAE about 20 of about 49 trees per tile). OAM-TCD's
  individual-tree labels are incomplete and canopy groups are ignored, which bounds what any
  method can show here.

## Decision
`point_heads()` keeps `heatmap="regression"`. The docs recommend `density_head()` for counting
and heatmap + offsets for localisation, with the peak threshold calibrated on validation data.
