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
