# E011: Tree localisation and counting heads on OAM-TCD (pre-registered)

Status: **planned** (registered 2026-10-14 before any run). Dataset approved by the user:
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
