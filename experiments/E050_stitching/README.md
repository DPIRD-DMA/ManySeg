# E050: stitching tiled predictions for vector outputs: blend vs cutlines (pre-registered 2026-10-29 11:20)

Why: ManyPred blends overlapping patch predictions with gradient weights (averaging). The user
asked (10:47) whether a cutline (each pixel from one patch, the cut placed where neighbouring
predictions agree) suits vector decoding better. Averaging can blur boundary ridges, split corner
peaks and mix direction modes; a cut keeps every pixel's heads from one prediction.

## Reference and question
The seamless reference is a **single forward pass over the whole scene** (fully convolutional
models; 2048 x 2048 fits on the 4090). Each stitching method is applied to ManyPred-style patch
predictions of the same scene (`manypred.windows.make_patch_indexes`, duplicates dropped). The
question: which stitching reproduces the single-pass **decoded vectors** best where patches
overlap. Agreement with the single pass needs no ground truth; it isolates the stitching effect
from model error. Context differences (a patch sees less than the whole scene) also cause
disagreement; agreement outside the overlap zone (single coverage) is reported as that floor.

## Tasks, models, scenes
* **Roads**: E044 unified model `U_s0` (City-Scale, 1 m), decoded with E044's tuned R4 decoder.
  Scenes: City-Scale 2048 px regions; val 9 (tuning), test 27. Patches 1024, overlap 256.
* **Roof faces**: E047b model (RGB + nDSM, 0.25 m), F3 faces with E047b's tuned settings.
  Scenes: 2048 px (512 m) windows built from 4 x 4 E047 input windows (PDOK 2023 ortho, AHN4 nDSM)
  at the densest 3DBAG locations: val 2 in Haarlem (tuning), test 2 in Leidsche Rijn + 2 in
  Staphorst. Patches 512, overlap 128 (the model's training size; ManyPred's 0.3 overlap ratio
  rounded).

## Stitching arms (all heads of a pixel come from the same rule; activations before stitching)
* **blend**: ManyPred's `gradient_weights` weighted mean (today's behaviour).
* **midline**: each pixel from the patch where it is most central (argmax of the gradient weight).
* **seam**: patches placed in ManyPred's row-major order; in the overlap with already placed
  output, a min-cost cut (dynamic programming, image-quilting style: one vertical path through the
  left overlap strip, one horizontal path through the top strip) where cost = sum over heads of
  mean |new - current| (direction heads / 2) + mu x distance from the strip's middle (normalised
  0-1; keeps the cut away from patch edges).
* **seam_obj**: seam + lambda x max(support_new, support_current) (support = road mask / roof
  extent), so the cut avoids objects.
Tuned on val (by the primary metric): mu in {0.1, 0.5} for seam; (mu, lambda) in {0.1, 0.5} x {1, 4}
for seam_obj.

## Metrics (decoded output vs single-pass decoded output, pooled over a task's test scenes)
Overlap zone = pixels covered by >= 2 patches; the rest = single-coverage zone.
* Roofs: face agreement F1 (raster IoU > 0.5 between faces; a face is in the overlap zone if any
  of its pixels is) per zone; plus field MAE vs the single pass per zone.
* Roads: junction agreement F1 (junctions = degree >= 3, matched one-to-one within 5 px) and length
  agreement F1 (graph points within 3 px) per zone.
Primary: roofs face F1 (overlap zone); roads junction F1 (overlap zone).

## Decision rule (fixed now)
Per task, the best cut arm on val (midline, seam, seam_obj) is **preferred over blend** if on test
its primary metric is >= blend + 0.01, and its guard is no worse than blend - 0.005 (roofs: face F1
over all faces; roads: length F1 in the overlap zone). If both tasks prefer a cut, ManyPred gets a
`merge="seam"` option for vector outputs (blend stays the default for plain rasters).
Single run (decoding is deterministic; predictions are fixed by the checkpoints).
