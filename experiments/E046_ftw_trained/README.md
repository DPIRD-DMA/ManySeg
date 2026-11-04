# E046: B0 vs B3 trained on farm fields, FTW Netherlands (pre-registered 2026-10-28 18:00)

Why: the maintainer asked whether B3 (instances + corners + wall directions, the footprint default since
E038) extends to field boundaries. E045 tests this with perfect targets; this is the trained
test on real imagery, registered before E045's result and run regardless of it.

## Data
Fields of The World, Netherlands (source polygons CC0-1.0): official train / val / test chips
(256 px, 10 m Sentinel-2), input = both dates' B04 B03 B02 B08 (8 channels, standardised with
train statistics). Targets: the official instance mask (B0 losses); B3's corner and direction
targets from each instance's traced outline, holes filled, Douglas-Peucker 1 px (`e045.chip_polygons`).
`prepare46.py` builds the cache.

## Models
ResNet34 U-Net (ImageNet weights, first conv adapted to 8 channels by timm), fastai_like recipe,
40 epochs, batch 16, dihedral augmentation, seed 0. **B0**: instance heads, bench InstanceTask
(boundary width 2). **B3**: B0's heads + corner (focal, CenterNet prior) + direction (masked L1),
E036's CornerLineTask. Local 4090.

## Evaluation (`evaluate46.py`)
Per model on val: instance thresholds (extent {0.4, 0.6} x boundary {0.3, 0.5} x seed {0.15, 0.3}),
then the arm's grid (B0: simplify {0.5, 1, 1.5, 2}; B3: E036's 16 settings), by boundary F. Test once.
**Primary: boundary F** (E038: points every 0.5 px, 1 px tolerance = 10 m; points within 1 px of
the chip edge excluded) against the DP-1 px outlines. Secondary: detection F1, C-IoU, PoLiS (px).

## Decision rule (fixed now)
Seed 0, test: **B3 preferred for fields** if its boundary F beats B0 by >= 0.01 and its detection
F1 is no more than 0.03 below; **B3 adequate** (one representation for buildings and fields) if
its boundary F is no more than 0.01 below B0 with the same detection condition; otherwise fields
keep B0. If seed 0 lands within 0.01 of either threshold, seed 1 of both arms decides on the mean.

## Amendment 1 (18:05, before any training; Codex review notes/codex_e046.md)
* One accepted set of fields for both arms (`prepare46.instances`): each 4-connected piece of an
  FTW id is its own instance; pieces < 16 px dropped; background-only holes filled; polygon =
  exterior of the final pixels, DP 1 px; degenerate polygons removed from the mask too. On 300
  chips polygon-vs-mask IoU median 0.957, 1st percentile 0.72 (thin fields lose area to DP;
  the same GT for both arms).
* `prepare46` aborts on any incomplete split; `evaluate46` tunes both arms on val before any
  test, refuses to re-run a seed's test, records checkpoint hashes, and applies the rule above
  in code (`decide`).
* Labels: FTW Netherlands labels arable land only (`crop_types: Bouwland` in the data config);
  grassland parcels are background, as in the FTW benchmark. Both arms learn and are scored the
  same way; absolute boundary precision is lowered by visible but unlabelled fields.
* GPU nondeterminism (cuDNN) is not removed; seed 1 handles near-threshold cases as registered.

## Results (18:03; seed 0, NL test 388 chips)
Training: B0 5.0 min, val PQ 0.418; B3 ~13 min (CPU-bound corner targets), val PQ 0.421.

| arm | thresholds (extent/boundary/seed) | decoder setting | val boundary F | test boundary F | test P / R | test det F1 | C-IoU | PoLiS px |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.6/0.3/0.3 | {'simplify': 0.5} | 0.592 | 0.592 | 0.617 / 0.569 | 0.593 | 0.262 | 1.62 |
| B3 | 0.6/0.3/0.3 | {'corner_threshold': 0.5, 'big_dev': 4.0, 'agree': 0.6, 'merge_deg': 8.0} | 0.568 | 0.565 | 0.608 / 0.528 | 0.604 | 0.530 | 1.73 |

B3 - B0: boundary F -0.027, detection F1 +0.011.
**Decision: fields keep B0** (B3's boundary F is more than 0.01 below; not within 0.01 of a
threshold, so no seed 1). This agrees with the E045 oracle. The B3-trained model's instance heads
are as good as B0's (val PQ 0.421 vs 0.418, test detection F1 0.604 vs 0.593), so one B3-style
model can serve fields as well, decoded with the mask outline (`polygons_from_labels`) instead
of corners and lines (not tested on test; would be a new comparison).

