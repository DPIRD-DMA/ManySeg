# E039: the unified head set on roads (R4), oracle (pre-registered 2026-10-28 12:40)

Why: the user wants one solution for all graph-shaped outputs. B3 (buildings) = region + lines +
keypoints + line direction, decoded as "lines from the mask, nodes from keypoints, geometry from
directions". E037 killed the dense pixel graph for roads (overpasses) and E028's R1 (mask +
junction heatmap, no pairing) gave no gain. This tests whether the B3 head set, extended with a
**crossing** keypoint class and a **second direction mode** (only where two roads overlap),
fixes the mask's road topology (junctions, overpasses) while keeping its robustness. The same
heads with the crossing class and second mode unused are B3.

## Representation R4 (roads)
Targets: road mask (E028 R0 widths; region = line support for roads), keypoint heatmaps
(Gaussian sigma 2 px) for **junctions** (GT degree >= 3), **ends** (degree 1, not at the tile
border) and **crossings** (E028 grade-separated crossings), and a direction field
(cos 2t, sin 2t) in two modes over each road's mask: mode 1 = first road at a pixel, mode 2 =
a second road of a different chain whose angle differs by > 20 deg (zero elsewhere).
Corruptions: E028 levels for mask and heatmaps (missed peaks per class); the direction field is
zeroed under the occlusion disks and gets N(0, 0.1 x vec) noise (as E035's B3).

## Decoder
1. R0 on the mask (its tuned parameters at that level, as B3 used B0's), giving the
   skeleton graph.
2. Keypoint peaks per class (threshold thr_k, 5 px kernel); zones = disks of radius zr around
   them; R0's graph is clipped outside the zones; each clipped line end on a zone boundary is an
   **arm**, with a heading from its last 6 px, refined by the direction mode closest to it.
3. Per connected zone: with a junction peak, all arms join one node at the junction peak (two
   or more junction peaks in a zone: nodes joined by a minimum spanning tree); with only a
   crossing peak, arms are **paired straight through** (greedy, most opposite first, if the
   headings are within pair_deg of opposite) and paired arms are joined without a node, so the
   two roads do not meet; leftover arms join a node at the crossing peak; with only an end
   peak, the arm runs to the end point.
4. Components shorter than 20 px dropped; E028 end bridging (gap).
Tuned on Innaloo per level (2 seeds) over thr_k {0.2, 0.4} x zr {8, 12} x pair_deg {140, 160} x
gap {0, 25, 50}; R0 and R2 use E028's tuned parameters per level. Test: Murdoch and Northam,
levels 0-3, seeds 0-4; R0, R2 and R4 decoded on the same corrupted draws.

## Decision rule (fixed now)
Pooled over the two test areas:
1. **Topology gain**: R4 junction F1 >= R0's + 0.10 at L0 and at L1, and R4 keeps >= 50% of
   GT crossings without a false junction (`cross_ok / cross_n`) at L0.
2. **Robustness kept**: R4 APLS >= R0's - 0.01 at L1 and at L2.
Both -> **the unified head set is viable for roads**; next step a trained City-Scale screen
(own pre-registration). Reported alongside: R2 (graph ceiling), per-area results, level 3.

## Compute
CPU only, MemoryMax 12 GB, 6 processes, nice 10.

## Results
(pending)
