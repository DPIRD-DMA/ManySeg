# E047: trained LoD2 roof faces from aerial RGB + AHN heights (pre-registered 2026-10-28 18:20)

Why: the user's roof idea (graph faces + one plane per face from height, E043/E043b) has only
been tested as an oracle (E031, E043b). This is the first trained screen. User authorised
autonomous overnight work (16:13), local or remote compute as reasonable.

## Data (all open, Netherlands)
* Roof faces: 3DBAG v20250903 LoD2.2 RoofSurfaces ("© 3DBAG by tudelft3d and 3DGI", CC BY 4.0).
* Input RGB: PDOK Luchtfoto `2023_ortho25` (Beeldmateriaal Nederland, CC BY 4.0), WMS, 25 cm.
* Input height: AHN4 DSM and DTM 0.5 m (PDOK WCS, CC0); nDSM = DSM - DTM (DTM gaps under
  buildings filled from the nearest valid pixel), clipped to [0, 60] m, bilinear to 25 cm.
* Caveat: 3DBAG roofs are reconstructed from AHN point clouds, so the height input and the
  targets share a source; this matches how Dutch LoD2 is produced, but results transfer to other
  countries only where a comparable DSM exists.
Tiles 512 px at 0.25 m (128 m) as E031 (`e031.make_tiles`, >= 3 buildings). Areas (3DBAG tiles
in a 2 km box): train amersfoort, zwolle, eindhoven, groningen (up to 8 3DBAG tiles each, all
tile windows); val haarlem (6 3DBAG tiles, at most 150 windows); **test = E031's 40 + 40 tiles of
leidscherijn and staphorst** (same windows, seed 0).

## Model
One ResNet34 U-Net, 4 input channels (RGB + nDSM / 10 m), B3 heads on **roof faces** as instances
(extent / boundary / distance, corner heatmap of face vertices turning > 20°, wall-direction
field), E036's CornerLineTask, fastai_like, 40 epochs, batch 8, dihedral augmentation, seed 0.

## Arms (test areas; planes always from the input nDSM)
* **A0**: E031's plane segmentation of the input nDSM inside the building mask, no learning;
  mask = nDSM > 2.5 m (no learned mask for an untrained baseline); E031's grid, tuned on val.
* **F0**: trained face instances -> exact outline + Douglas-Peucker (`polygons_from_labels`).
* **F3**: trained face instances + corners + directions (`polygons_from_corners`, E036 grid).
F0/F3 planes: E043b's robust eroded fit on the input nDSM per face. Instance thresholds and each
arm's grid tuned on val by face F1 x (1 - min(1, height RMSE / 1 m)) (E031's score).

## Metrics (E031's `metrics`, against 3DBAG faces and planes)
Face F1 (IoU > 0.5), height RMSE (m, all building pixels; a missed roof counts as 0), corner F1
(0.5 m), C-IoU, normal-angle error; per test area.

## Decision rule (fixed now)
**Trained faces viable** if the better of F0/F3 beats A0 on face F1 by >= 0.10 on both test
areas with height RMSE no worse than A0 + 0.10 m. **F3 preferred over F0** if its face F1 is
>= F0 + 0.02 on both areas and its corner F1 is not lower. Single seed screen; a second seed only
if a decision lands within 0.02 of a threshold.

## Amendment 1 (18:25, before any training; Codex review notes/codex_e047.md)
* Decision: viability is judged per single trained arm (F0 or F3 must pass on both areas), and
  the output reports whether any margin is within 0.02 (second-seed trigger).
* Normal-angle error is pooled and reported (it was computed but dropped).
* A tile with no valid DTM in its window takes its terrain from a padded (+96 m) DTM window
  instead of the DSM median; `prepare47.py repair` fixes any tiles built before this change.
* Test once per checkpoint: a ledger keyed by the checkpoint's SHA-256.
* Diagnostic arm **A0m** (secondary, not in the rule): A0's plane segmentation inside the learned
  roof extent (F0's tuned extent threshold), separating mask quality from face partitioning.

## Amendment 2 (18:40, before any training; visual check of prepared tiles)
Rendering prepared tiles showed buildings in the imagery and nDSM without 3DBAG faces, from two
causes, so windows were filtered (`prepare47.py filter`, `consistency47.py`, `prepare47.py consistency`):
* Coverage: windows reaching outside the fetched 3DBAG tiles (outlines from the 3DBAG WFS) were
  dropped: train 714 -> 579, val 97 -> 60, test 80 -> 66.
* Time: 3DBAG (BAG 2025) vs the AHN survey: faces on flat ground (new builds) or smooth elevated
  roofs without faces (demolished). A tile is kept if >= 90% of its face pixels have nDSM > 1.5 m
  and smooth (5x5 std < 0.3 m) elevated (> 2.5 m) blobs >= 25 m^2 away from faces cover <= 10% of
  the face area: train 579 -> 502, val 60 -> 59, test 66 -> 48 (leidscherijn 21, staphorst 27).
The test set is therefore a subset of E031's windows; the decision rule is unchanged.

## Results
(pending)
