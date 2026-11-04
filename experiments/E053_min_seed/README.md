# E053: dropping seed fragments (`min_seed`) on trained models (pre-registered 2026-11-02 09:00)

Why: the maintainer asked for the hip-roof tip fix before 0.1.0. In the end-to-end oracle
(`tests/integration/test_vector_pipeline.py`, perfect targets) the 45-degree tip of a hip-end
face gets a seed of its own and becomes an extra 17-25 px face (5 slow-tier cases, also in a
single pass), and with fields' absolute distance (E052, seed threshold 3 px) a 1 px seed on
the threshold's edge at a field corner becomes an extra 16 px field. Both are seed fragments:
over every e2e scene, seed components that are not an instance's largest are 1-6 px, the
largest are >= 60 px (roofs), >= 110 px (footprints), >= 594 px (fields). New option
`instances_from_outputs(min_seed=)`: seed components (4-connected) under `min_seed` px are
dropped; a foreground component left without a seed gets one at its distance maximum (as
before), so small objects stay. With `min_seed=8` all 5 tip cases and both field corner cases
pass in the oracle (checked before this registration; it is not trained evidence).

## Design
No training. Existing checkpoints, each task's shipped decoder settings (`block_decoder`),
`min_seed` 0 (today) vs 8, fixed (no tuning), val and test:
* roofs: E047b `B3_s0_e150` (3D BAG; E047's F3 metrics: face F1, height RMSE, corner F1),
  extent 0.4 / boundary 0.3 / seed 0.15, corners 0.5, big_dev 4, agree 0.6, merge 8;
* footprints: E036 `B3_s0`, `B3_s1` (WHU; E038's metrics: boundary F, detection F1),
  0.6 / 0.5 / 0.15, corners 0.5, big_dev 4, agree 0.4, merge 8;
* fields: E052 `A_s0`, `A_s1` (FTW NL; E046's metrics), 0.4 / 0.3 / 3 px, simplify 0.5.
These test sets were used before for these checkpoints (E038, E047b, E052); this is a new
comparison of a decoder option at fixed settings, not a re-selection.

## Decision rule (fixed now)
Per task, on test (mean over seeds where there are two), min_seed 8 - min_seed 0:
`min_seed=8` becomes the task's `block_decoder` default if the primary metric (roofs face F1;
footprints and fields boundary F) is >= -0.005 and detection F1 (roofs: face F1) >= -0.005.
Otherwise the task keeps 0, and its oracle failures stay documented limitations.

## Results (09:00; `runs/e053_*.json`; GPU shared with test runs, timings not reported)
Test, `min_seed` 0 -> 8 at the shipped settings. min_seed 0 reproduces the earlier results
(E047b F3 face F1 0.571 / 0.589; E052 A runs).

| task (checkpoints) | primary 0 -> 8 | detection F1 0 -> 8 | objects 0 -> 8 | other |
|---|---|---|---|---|
| roofs (E047b s0) | face F1 0.580 -> 0.621 (leidscherijn 0.571 -> 0.588, staphorst 0.589 -> 0.656) | (face F1) | 2481 -> 2113 | height RMSE 1.35 -> 1.28 m, corner F1 0.314 -> 0.337 |
| footprints (E036 s0, s1) | boundary F 0.654 -> 0.658, 0.659 -> 0.662 | 0.828 -> 0.854, 0.830 -> 0.852 | -5.8%, -4.9% | C-IoU +0.001-0.002 |
| fields (E052 A s0, s1) | boundary F 0.5957 -> 0.5956, 0.5985 -> 0.5983 | 0.592 -> 0.601, 0.599 -> 0.605 | -3.2%, -2.5% | C-IoU -0.001 |

Mean 8 - 0: roofs face F1 +0.041; footprints boundary F +0.004, detection F1 +0.024; fields
boundary F -0.0002, detection F1 +0.008. Val agrees in direction for every task.
**Decision (rule above): `min_seed=8` becomes the `block_decoder` default for footprints,
fields and roofs.** Trained models make many more seed fragments than the oracle: the dropped
seeds were mostly spurious objects, not lost ones: object precision 0 -> 8 rose (roofs 0.558 ->
0.648, footprints 0.792 -> 0.841 / 0.792 -> 0.833, fields 0.595 -> 0.615 / 0.629 -> 0.644) while recall
moved by at most 0.007 (roofs 0.604 -> 0.597; footprints and fields within 0.001).
`instances_from_outputs` keeps 0 (a low-level function with generic defaults). In the e2e
oracle all hip-tip and field-corner cases now pass; the holed field's narrow strip (3 px seed
threshold) still splits and stays a documented limitation.
