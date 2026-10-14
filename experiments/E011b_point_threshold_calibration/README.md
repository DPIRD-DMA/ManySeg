# E011b: Validation-calibrated peak threshold for tree localisation (pre-registered)

Status: **planned** (registered 2026-10-14 before any run). Follow-up to E011, whose fixed peak
threshold (0.5) gave precision 0.84 but recall 0.18 for MSE heatmaps.

## Question
If the peak threshold is chosen on validation data, do MSE-regressed centre heatmaps recover
recall and point F1 on the official test split?

## Protocol
* Identical to E011's `hm_mse` arm (OAM-TCD CC-BY, 1024² at 20 cm, U-Net resnet34 `tv_in1k`,
  `fastai_like`, 30 epochs, batch 8, σ = 2 px, radius 1 m). Seeds 0–2. Pawsey Setonix MI250X.
* New: `--task-kwargs '{"calibrate": true}'`. Every validation pass scores the threshold grid
  {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}. After training, the threshold with the best
  **validation** F1 at the final epoch is recorded (`calibrated_threshold`), and test is
  evaluated **once** at that threshold. Test never influences the choice.
* Comparison: test F1 at the calibrated threshold vs E011 `hm_mse` test F1 at 0.5
  (0.4006 ± 0.0244). Also reported: the calibrated threshold per seed, precision and recall.
* Budget: 3 runs, about 1 h of GCD time.

## Decision rule
If calibrated test F1 exceeds E011's fixed-threshold test F1 by more than 2 pooled seed SDs,
the docs and the bench default for point tasks switch to validation-calibrated thresholds
(`calibrate=True`). Otherwise we report it and keep the fixed threshold with the caveat.
