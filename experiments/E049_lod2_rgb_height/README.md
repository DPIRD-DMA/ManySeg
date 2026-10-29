# E049: LoD2 roofs from RGB only, with predicted heights (pre-registered 2026-10-29 09:30)

Why: E047b's roof model takes the AHN nDSM as an input channel and fits each face's plane to it,
so it needs a DSM at inference time. The user asked (09:15) for RGB as the only input, with the
model predicting the heights that the reconstruction then uses, normalised so that non-buildings
are 0.

## Data
E047's cache unchanged (3DBAG LoD2.2 faces, PDOK 2023 ortho 25 cm, AHN4 nDSM; train 502 tiles
from 4 towns, val 59 Haarlem, test 48 Leidsche Rijn + Staphorst, 512 px = 128 m). New target,
precomputed once (`train49.py targets`): **height** = the 3DBAG roof planes rendered per pixel
(E031 `height_map`: metres above each building part's mean ground; the same height the metric
scores), **0 off roofs**. Trees, cars and terrain are 0, so the target is the building-only
nDSM the user described. The AHN nDSM is not used by the H arms at all (input or planes).

## Model
E047's ResNet34 U-Net and B3 heads (CornerLineTask: face instances, corners, wall direction),
**3 input channels (RGB)**, plus:
* **height** head (1 channel, metres / 10): L1 over all pixels, weight 1.
* **slope** head (arm H1 only; 2 channels, dz/dx and dz/dy in m per pixel): L1 on roof pixels
  whose 4 neighbours lie in the same face, target = central difference of the augmented height
  target (so no extra transform is needed under the dihedral augmentation), weight 1.
fastai_like, 150 epochs (E047b), batch 8, dihedral augmentation, seed 0.

## Arms (test areas)
* **H0**: RGB -> heads + height. Faces F3 (`polygons_from_corners`) + `planar_faces` (2.5 px,
  E048b); plane per face = E043b's robust eroded fit **to the predicted height**; J2 mesh.
* **H1**: RGB -> heads + height + slope. Same faces; plane per face from the **predicted slopes**
  (median dz/dx, dz/dy inside the eroded face) with the offset from the predicted height
  (E043's `slope` arm). H1's robust-from-height planes are reported too (H1r).
* **R** (reference): E047b's RGB + nDSM model (`B3_s0_e150.pt`), same F3 + planar faces, robust
  planes from the real nDSM, E047b's selected settings (tuned on the same val by the same score).
* **D** (diagnostic, not in the rules): H0's faces with robust planes from the real nDSM,
  separating face errors from height errors.
Tuning (H0, H1 separately): instance thresholds then E047's F3 grid on val by E031's score face
F1 / (1 + height RMSE), with robust planes from the predicted height and without the planar join
(as in E047); the planar join is applied at test with the tuned settings.

## Metrics (E031 `metrics` against 3DBAG, per test area)
Face F1 (IoU > 0.5), height RMSE (m, all GT building pixels), normal-angle error, corner F1,
C-IoU; plus height-raster MAE on GT roof pixels and on non-roof pixels for the predicted height
and for the real nDSM.

## Decision rules (fixed now)
1. **RGB-only roofs viable** if the best of H0 / H1 / H1r (one arm, chosen by val score) has, on both
   test areas, face F1 >= R - 0.10 and height RMSE <= R + 0.75 m.
2. **Close to the DSM pipeline** if that arm also has face F1 >= R - 0.03 and RMSE <= R + 0.25 m
   on both areas.
3. **Slope head worth it** if H1 (slope planes) has normal error <= H0 - 1 deg on both areas with
   RMSE <= H0 + 0.05 m (face F1 reported; H0 and H1 are different models).
Seed 0 only; a second seed if any rule margin is within 0.02 (F1) / 0.05 m / 0.25 deg.
Test once per checkpoint (ledger keyed by SHA-256).

## Amendment 1 (09:35, during training, before any evaluation; Codex review notes/codex_e049.md)
* Height target: `train49.py targets` clips the E031 height to [0, 100] m. This touches 0.003% of
  train roof pixels (1051 below 0, min -5.6 m; 4096 above 100 m), 0.005% of val and 0.002% of
  test; kept (no retrain), the metric still scores the unclipped E031 height.
* Predicted heights are clamped at 0 before plane fitting (heights above ground cannot be
  negative); the height MAE is reported on the raw, unclamped prediction.
* "J2 mesh" in the arms: the metrics are E031's face / plane rasters as in E047 (J2 does not change
  them); J2 meshes are built only for the visual 3D comparison.
* Protocol: the test ledger keys each checkpoint (H0, H1) separately and is written before the test
  split opens; the R checkpoint's SHA-256 must equal E047b's; full runs require both test areas;
  `--limit` smoke runs (val only) are marked non-decision-grade. One smoke run was made with
  2-epoch smoke checkpoints; no smoke run uses the real checkpoints.

## Results, seed 0 (10:02; worktree e049-433109f with evaluate49 from ebdac49; `runs/eval_H0_s0_H1_s0.json`)
Training 150 epochs: H0 16.9 min, val PQ 0.092 (under-segmentation 0.72); H1 val PQ 0.100
(E047b with the nDSM: 0.252). Val: H0 face F1 0.21, RMSE 3.78 m, normal error 22.7 deg.
Test (48 tiles; Leidsche Rijn / Staphorst):

| arm | face F1 | height RMSE m | normal err deg | corner F1 | C-IoU |
|---|---|---|---|---|---|
| R (RGB + nDSM, F3 + planar) | 0.564 / 0.591 | 1.37 / 1.27 | 5.6 / 2.9 | 0.36 / 0.29 | 0.63 / 0.61 |
| D (H0 faces, nDSM planes) | 0.168 / 0.303 | 1.92 / 1.84 | 10.5 / 4.5 | 0.10 / 0.11 | 0.38 / 0.34 |
| H0 (RGB, predicted height) | 0.168 / 0.303 | 4.26 / 2.96 | 22.5 / 20.1 | 0.10 / 0.11 | 0.38 / 0.34 |
| H1 (RGB, predicted slopes) | 0.181 / 0.312 | 3.93 / 2.96 | 19.2 / 22.5 | 0.10 / 0.11 | 0.39 / 0.36 |
| H1r (H1, planes from its height) | 0.181 / 0.312 | 4.25 / 3.09 | 23.8 / 20.9 | 0.10 / 0.11 | 0.39 / 0.36 |

Height raster MAE on GT roof pixels: H0 3.20 / 2.26 m, H1 3.16 / 2.32 m, AHN nDSM 0.48 / 0.44 m;
off roofs: H0 0.36 / 0.24 m, H1 0.43 / 0.28 m, AHN 0.52 / 0.62 m (trees).

**Decision (seed 0):** best arm on val H1r. Rule 1 **fails** (face F1 -0.38 / -0.28 vs R, allowed
-0.10; RMSE +2.88 / +1.82 m, allowed +0.75); rule 2 fails; rule 3 (slope head) **fails** (normal
error -3.3 deg at Leidsche Rijn, +2.3 deg at Staphorst). The second-seed trigger fired on rule 3's
RMSE margin (Staphorst +0.005 m vs the 0.05 bar), so seeds 1 (H0_s1, H1_s1) run under the same
protocol (unit e049-s1); rule 1 cannot flip on a second seed (gap 0.28-0.38 vs 0.10).

Reading: without height input the face partition collapses (adjacent faces and terraced houses
merge); D shows those faces alone cost +0.55 m RMSE over R. Predicted heights then add
+1.0 to +2.3 m: the model finds buildings and keeps trees / ground at 0 (cleaner than AHN off
roofs) but guesses storey counts (Leidsche Rijn flats 4-5 m too tall) and blurs ridges, so plane
tilts are 20 deg off. Visual: `fig_e049_heights.png` (session scratchpad), status page section
"RGB-only roofs".
