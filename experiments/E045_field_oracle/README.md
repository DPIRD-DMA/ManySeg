# E045: do the footprint representations transfer to farm fields? (pre-registered 2026-10-28 17:25)

Why: the user asked whether the B3 approach (instances + corners + wall directions) extends to
field boundaries. Fields differ from buildings: larger (median ~635 px at 10 m), more irregular
outlines (curved, ragged), and in some places they touch (shared edges). Oracle first, as for
footprints (E028/E035): perfect targets + E028's corruption levels L0-L3, no training.

## Data
Fields of The World (FTW) instance masks, Netherlands and Denmark (source polygons CC0-1.0), val
and test chips only (256 px, 10 m). `e045.field_tiles`: 1024 px oracle tiles = 3 x 3 chips at
offsets 0/384/768 px (128 px empty gutters, so no false contacts). GT = each instance's traced
outline, holes filled, Douglas-Peucker 1 px (vector-like, no staircase; the E038 lesson), >= 16 px.
Tiles: NL val 42 (tuning), NL test 43, DK test 36 (DK = transfer: never tuned on).

## Arms (same corrupted targets per tile/level/seed)
* **B0**: instance heads (extent/boundary/distance) -> contour -> Douglas-Peucker; grid thr
  {0.4, 0.6} x thr_b {0.3, 0.5} x thr_s {0.15, 0.3} x simplify {0.5, 1, 2} (24), tuned per level.
* **B3**: E035b's decoder (`decode_b3r`, the one the library's `polygons_from_corners` follows)
  on the tuned B0 instances; grid thr_c {0.3, 0.5} x big_dev {4, 8} x agree {0.4, 0.6} x
  merge_deg {8, 15} (16), tuned per level.
Tuning: NL val, seeds 0-1, by pooled boundary F. Test: NL test + DK test, seeds 0-4.

## Scores
**Primary: boundary F** (E038 definition: points every 0.5 px along GT and predicted outlines,
matched within 1 px; points within 1 px of a chip edge excluded as artificial cuts), pooled per
test set, level and seed. Secondary: detection F1 (IoU > 0.5), C-IoU, PoLiS, N-ratio; recall on
the irregular subset (GT area / min rotated rectangle < 0.8).

## Decision rule (fixed now)
B3 transfers to fields if, on both test sets at L1 and L2, its boundary F is no more than 0.01
below B0's and its detection F1 no more than 0.03 below (mean over seeds). B3 *preferred* for
fields if, in addition, its boundary F beats B0 by >= 0.01 at L1 and L2 on both sets. Otherwise
fields keep B0 (masks) and the "one representation" claim is limited to buildings.

## Notes
* E045 found `connected_components` slow on large CPU components (102 sweeps, ~9 s per tile);
  fixed in the library (scipy path, identical ids, commit dd0e06b) before any result.

## Amendment 1 (17:30, before any tuning or test result; Codex review notes/codex_e045.md)
* Corruption scale: E028's occlusion rate is per 100 m of outline with 4-12 px disks. Tiles now
  carry gsd 0.5 m for corruption, so fields get the same disks per pixel of outline as the building
  oracle (at the true 10 m, disks would cover most outlines; at gsd 1 there were ~10x fewer).
  PoLiS is reported in px.
* B0 now vectorises with the library's exact pixel-edge outline (`polygons_from_labels`) +
  Douglas-Peucker, as B3 does before its corner step; E028's marching-squares contour cut each
  pixel corner by half a pixel and favoured B3 for a reason unrelated to the representation.
* Analysis (`analyze45.py`): per test set, level and seed, pool counts over tiles; mean over seeds;
  apply the rule above.

## Results
(pending)
