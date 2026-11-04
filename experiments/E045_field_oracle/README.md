# E045: do the footprint representations transfer to farm fields? (pre-registered 2026-10-28 17:25)

Why: the maintainer asked whether the B3 approach (instances + corners + wall directions) extends to
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

## Results (17:45; Pawsey CPU job 51717785, 14 min on one 128-core node, ~30 core-h)
Tuned on NL val: B0 thr 0.6 (0.4 at L0), thr_b 0.3, thr_s 0.15, **simplify 0.5** at every level;
B3 thr_c 0.5 (0.3 at L0), big_dev 4, agree 0.6, merge_deg 8. Test, mean ± sd over 5 seeds:

| test set | metric | arm | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|---|
| netherlands test | bf | B0 | 0.995 ± 0.000 | 0.862 ± 0.007 | 0.538 ± 0.017 | 0.263 ± 0.003 |
| netherlands test | bf | B3 | 0.973 ± 0.000 | 0.818 ± 0.008 | 0.521 ± 0.018 | 0.249 ± 0.003 |
| netherlands test | det_f1 | B0 | 0.966 ± 0.000 | 0.827 ± 0.008 | 0.447 ± 0.020 | 0.087 ± 0.001 |
| netherlands test | det_f1 | B3 | 0.964 ± 0.000 | 0.821 ± 0.009 | 0.447 ± 0.020 | 0.087 ± 0.001 |
| netherlands test | ciou | B0 | 0.337 ± 0.000 | 0.276 ± 0.001 | 0.220 ± 0.001 | 0.171 ± 0.002 |
| netherlands test | ciou | B3 | 0.885 ± 0.000 | 0.677 ± 0.003 | 0.508 ± 0.004 | 0.331 ± 0.004 |
| netherlands test | polis | B0 | 0.301 ± 0.000 | 1.051 ± 0.008 | 2.171 ± 0.070 | 3.236 ± 0.068 |
| netherlands test | polis | B3 | 0.274 ± 0.000 | 1.225 ± 0.014 | 2.480 ± 0.086 | 3.586 ± 0.074 |
| netherlands test | irr_recall | B0 | 0.974 ± 0.000 | 0.863 ± 0.006 | 0.568 ± 0.023 | 0.316 ± 0.005 |
| netherlands test | irr_recall | B3 | 0.970 ± 0.000 | 0.855 ± 0.007 | 0.567 ± 0.023 | 0.317 ± 0.007 |
| denmark test | bf | B0 | 0.981 ± 0.020 | 0.782 ± 0.018 | 0.471 ± 0.017 | 0.260 ± 0.002 |
| denmark test | bf | B3 | 0.929 ± 0.020 | 0.739 ± 0.019 | 0.455 ± 0.017 | 0.249 ± 0.002 |
| denmark test | det_f1 | B0 | 0.935 ± 0.026 | 0.624 ± 0.029 | 0.312 ± 0.017 | 0.100 ± 0.002 |
| denmark test | det_f1 | B3 | 0.931 ± 0.026 | 0.620 ± 0.029 | 0.311 ± 0.017 | 0.100 ± 0.002 |
| denmark test | ciou | B0 | 0.342 ± 0.000 | 0.261 ± 0.001 | 0.189 ± 0.001 | 0.137 ± 0.001 |
| denmark test | ciou | B3 | 0.874 ± 0.000 | 0.681 ± 0.001 | 0.454 ± 0.003 | 0.262 ± 0.002 |
| denmark test | polis | B0 | 0.480 ± 0.074 | 1.636 ± 0.099 | 3.216 ± 0.143 | 4.928 ± 0.047 |
| denmark test | polis | B3 | 0.504 ± 0.076 | 1.915 ± 0.101 | 3.601 ± 0.142 | 5.326 ± 0.046 |
| denmark test | irr_recall | B0 | 0.977 ± 0.018 | 0.884 ± 0.018 | 0.593 ± 0.029 | 0.384 ± 0.012 |
| denmark test | irr_recall | B3 | 0.972 ± 0.018 | 0.878 ± 0.018 | 0.592 ± 0.028 | 0.385 ± 0.012 |

B3 transfers to fields: **False**; B3 preferred for fields: **False**

**Decision: B3 does not transfer to fields; fields keep B0 (masks).** Boundary F: B3 is 0.02-0.05
below B0 even with perfect targets (L0) and 0.04 below at L1 on both test sets (rule allows 0.01);
detection F1 and the irregular-field recall are equal. C-IoU is far higher for B3 (0.68 vs 0.28
at L1) because B0's tuned outline keeps the pixel staircase (simplify 0.5 hugs a 1 px tolerance);
PoLiS is slightly worse for B3.

Why (visual check, Denmark test tile 3 at L0, `e045_l0.png` / `e045_spike2.png` in the session
scratchpad): (1) curved and ragged field edges become straight walls between corners; (2) where
fields touch, a neighbour's corner peak lies within the 4 px snap radius of this field's ring, so
the outline snaps to it and makes an out-and-back spike. Buildings rarely touch, which fits B3
working there. Possible follow-up (not registered): corner ownership per instance (a peak may only
be used by the instance whose ring turns there). E046 tests the trained case.


### Exploratory follow-up (val only, not registered): corner ownership
`explore_own_corners.py`: each corner peak used only by the instance whose ring is nearest. NL val,
every third tile, seed 0, E045's tuned settings, pooled boundary F:

| level | B0 | B3 | B3 + ownership |
|---|---|---|---|
| L0 | 0.995 | 0.971 | 0.979 |
| L1 | 0.863 | 0.816 | 0.819 |
| L2 | 0.554 | 0.534 | 0.535 |

Ownership removes some spikes (+0.008 at L0) but almost none of the gap at realistic levels: the
deficit is mostly B3 straightening curved / ragged field edges. Not pursued; fields keep B0.
