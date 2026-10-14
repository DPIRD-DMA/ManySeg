# E011b: Validation-calibrated peak threshold for tree localisation (pre-registered)

Status: **complete** (2026-10-14; 3 runs, Pawsey Setonix MI250X, code `b385989`). **Positive: test F1 0.40 → 0.59.** Follow-up to E011, whose fixed peak
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

## Results

| Seed | Calibrated threshold (val) | Test precision | Test recall | Test F1 | Test count MAE |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.15 | 0.587 | 0.564 | 0.5755 | 21.67 |
| 1 | 0.15 | 0.583 | 0.588 | 0.5852 | 19.97 |
| 2 | 0.20 | 0.662 | 0.555 | 0.6037 | 20.09 |
| **Mean ± SD** | | | | **0.5881 ± 0.0143** | **20.58 ± 0.95** |

E011 `hm_mse` at the fixed threshold of 0.5 scored test F1 0.4006 ± 0.0244 (precision about
0.84, recall about 0.18).

* Calibrated − fixed: **+0.188 test F1**. 2 pooled SD = 0.040, so the difference is 9.4 pooled
  SDs.
* Val F1 peaks at a threshold of 0.15–0.2 on every seed (0.50–0.51). Val F1 at 0.5 in these
  runs is 0.27–0.32, which reproduces E011's 0.30. Calibration on val and on test agree:
  test's own best grid threshold is 0.2 on every seed, and the val choice is 0.15 or 0.2.
  The curve is flat near the optimum, so this shows little sensitivity to the exact value.
* Precision and recall are now balanced (about 0.6 each) instead of 0.84 / 0.18.

**Exploratory, not pre-registered:** counting calibrated peaks gives test count MAE
20.58 ± 0.95, against 40.97 at the fixed threshold and 19.40 ± 0.37 for E011's density
head. Most of the density head's counting advantage in E011 came from the miscalibrated
threshold. Density is still slightly better, about 1.2 trees per tile, with no threshold to
tune.

## Decision
**Switch to calibration.** `PointTask(calibrate=True)` is now the bench default for point
tasks. Validation scores the threshold grid, and test is evaluated once at the val-best
threshold. Density tasks are unaffected because they have no peaks. The library's
`find_peaks` keeps an explicit threshold argument and no default calibration, because
calibration needs labelled validation data. The docs tell users to calibrate it. E011's
statement "use the density head for counting" is softened: density stays slightly better and
needs no threshold, but calibrated peaks are close.
