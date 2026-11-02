# E053: dropping seed fragments (`min_seed`) on trained models (pre-registered 2026-11-02 09:00)

Why: the user asked for the hip-roof tip fix before 0.1.0. In the end-to-end oracle
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
