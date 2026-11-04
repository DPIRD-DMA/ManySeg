# E033: distance-aware vertex loss and snap offsets for graph vertices (pre-registered 2026-10-27 11:35)

Status: **complete: screened out on seed 0** (2026-10-27; see Results). Asked for by the maintainer: a correct but slightly offset vertex should be
penalised less than a far one, the penalty should grow with distance, and training should push
predictions towards the right place, as general tools usable for points, lines and boundaries.
Codex reviewed a first version; its issues (scale mismatch, no real "pull", distorted soft targets,
quadratic EDT) were fixed before this pre-registration. Two general tools came out of it:

* `manyseg_bench.losses.distance_weighted_bce` + `distance_map`: inside the target blob plain BCE;
  outside it, false positives weighted `1 − exp(−D²/(2·2²))` (cheap next to the truth, full cost far
  away) plus `1.0 · p · D / 32` (more the further away, capped at 32 px); pixel means; distances
  from the exact geometry, precomputed in the loader (SciPy, linear time). It suppresses wrong
  predictions in proportion to their distance; it does not move them.
* `manyseg.vectors.offset_field` / `offset_field_from_mask` / `snap_points` (now in `experiments/E030_whu_polygons/graph_tensor.py`): the vector from each
  pixel within a radius to the nearest true element (points, or line / boundary pixels), trained
  as a 2-channel regression head; at decode time a detected point moves by its predicted vector.
  This is the part that moves predictions to the right place.

## Question
On WHU footprints (graph method, E030 setup), does either tool give more accurate, more complete
polygons than plain B2?

## Arms (3 seeds each; E030's setup, data, recipe, budget; decoder tuned on val; test once)
| arm | change from B2 |
|---|---|
| B2 (control) | E030's B2 runs, re-evaluated with this evaluation code |
| B2d | vertex loss = `distance_weighted_bce` (tolerance 2 px, far weight 1, cap 32 px; D to the exact vertex pixels) |
| B2s | extra 2-channel `snap` head: `offset_field` to the nearest true vertex within 8 px (/8), masked smooth-L1, weight 1; decoder places each vertex at its peak pixel's centre plus the predicted offset |

## Decision rule (fixed before any run)
Primary metric C-IoU × detection F1 on test, paired by seed, each arm against B2. An arm wins if
the mean paired difference is ≥ 0.02 and > 2 SDs of the paired differences, with C-IoU not more
than 0.02 lower. Secondary, reported: PoLiS (the direct measure of vertex position error, where
B2s should show), C-IoU, detection F1, recall, N-ratio. Training-failure check as E030. One
dataset: a win makes the tool an *option* for graph heads; a default needs a second dataset
(next: tree points on OAM-TCD, E011's setup, and field boundaries on AI4Boundaries, E010's).

## Budget
6 runs × about 60–75 min on the local 4090 (12 GB capped service), queued after E032.

## Amendment 1 (2026-10-27 12:10, before any E033 result): staged seeds
At the maintainer's request, to shorten iteration: every arm first runs **seed 0 only**. An arm goes on
to seeds 1–2 only if its seed-0 **validation** C-IoU × detection F1 (the tuned decoder's val score)
beats B2 seed 0's by more than 0.005; test is not looked at for this. The decision rule is
unchanged and is applied only to arms with all 3 seeds; arms stopped after seed 0 are reported as
"screened out on one seed", with no claim beyond that. Run order: B2e, B2w (E032), then B2d, B2s
(E033), then the advanced arms' seeds 1–2.

## Amendment 2 (2026-10-27 12:45, agreed with the maintainer after a Codex review of the options, `E032_graph_completeness/OPTIONS.md`; no result of the affected runs exists)
B2d is dropped (same reason as E032's B2w). B2s runs after E034's B2t. Controls for seeds 1–2, if needed, are trained then (E030 amendment 3).

## Results

Status: **complete: screened out on seed 0** (2026-10-27). Validation C-IoU × F1: B2 0.440,
**B2s 0.423** (B2d dropped, amendment 2). Test: B2s C-IoU 0.612, detection F1 0.732, C-IoU × F1
0.448, PoLiS 0.40 m (B2 0.613 / 0.750 / 0.460 / 0.39 m). The snap head did not improve vertex
positions (PoLiS) or closure. Single seed: "not promising", no stronger claim.
