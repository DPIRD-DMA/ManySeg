# E046: B0 vs B3 trained on farm fields, FTW Netherlands (pre-registered 2026-10-28 18:00)

Why: the user asked whether B3 (instances + corners + wall directions, the footprint default since
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

## Results
(pending)
