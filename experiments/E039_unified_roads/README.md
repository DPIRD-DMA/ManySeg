# E039: the unified head set on roads (R4), oracle (pre-registered 2026-10-28 12:40)

Why: the maintainer wants one solution for all graph-shaped outputs. B3 (buildings) = region + lines +
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

## Amendment 1 (2026-10-28 13:10, after smoke tiles only; no tuning or test result read)
Rendered smoke tiles (Murdoch 1, 3, 5, 9 at L0-L2) changed the design before any full run:
* **Lines head = thin centreline band (3 px)**, separate from the road surface (region): with the
  surface as the line support, dual carriageways whose surfaces touch became ladders. This is the
  road analogue of B3's outline channel.
* Keypoint peaks are kept only on the road surface (noise peaks off-road flooded L1).
* The thin centreline is fragile under corruption (its skeleton grows spurs that defeat gap
  bridging: one tile, L1, APLS 0.07 -> 0.17 with bridging vs surface 0.17 -> 0.71), so the decoder
  has three **modes**, chosen by tuning like any parameter: `centre` (centreline skeleton, dead
  ends healed along the surface skeleton), `surface` (surface skeleton, as R0) and `hybrid`
  (surface skeleton, replaced by the centreline within 15 px of branch points that no junction or
  crossing keypoint explains). A tried "drop ladder rungs" rule was abandoned (it cut lanes).
* Grid now thr_k {0.2, 0.4} x zr {8, 12} x mode {centre, surface, hybrid}; pair_deg 150, line
  threshold 0.3, gap = R0's tuned gap per level (compute). Decision rule unchanged.

## Decision rule (fixed now)
Pooled over the two test areas:
1. **Topology gain**: R4 junction F1 >= R0's + 0.10 at L0 and at L1, and R4 keeps >= 50% of
   GT crossings without a false junction (`cross_ok / cross_n`) at L0.
2. **Robustness kept**: R4 APLS >= R0's - 0.01 at L1 and at L2.
Both -> **the unified head set is viable for roads**; next step a trained City-Scale screen
(own pre-registration). Reported alongside: R2 (graph ceiling), per-area results, level 3.

## Compute
CPU only, MemoryMax 12 GB, 6 processes, nice 10.

## Results (2026-10-28)
Tuned locally on Innaloo (52 min, worktree e039-4fd9f01): L0 chose mode `centre` (thr_k 0.2, zr 8),
L1-L3 chose `surface`; `hybrid` never won. Test (Murdoch + Northam, L0-L3, seeds 0-4) ran on Pawsey
Setonix (job 51714535, 120 CPU, 2 min 46 s; code vec-5365cc3, decoder identical to the tuned
version; all arms in one run). Pooled over both areas (`runs/decision.json`):

| arm | APLS L0 / L1 / L2 / L3 | junction F1 L0 / L1 / L2 | crossings kept L0 / L1 / L2 |
|---|---|---|---|
| R0 mask | 0.937 / **0.572** / **0.210** / 0.036 | 0.759 / 0.651 / **0.314** | 0% / 2% / 8% |
| R2 graph | **0.985** / 0.522 / 0.122 / 0.013 | **0.976** / **0.794** / 0.217 | 94% / 69% / **41%** |
| R4 unified | 0.983 / 0.569 / 0.201 / **0.077** | 0.882 / 0.683 / 0.157 | **100%** / **76%** / 27% |

### Decision (registered rule)
Topology gain: junction F1 +0.123 at L0 but +0.032 at L1 (< +0.10) -> **fails**; robustness:
APLS -0.003 at L1, -0.009 at L2 (>= -0.01) -> passes. **R4 is not viable by the rule.**

### Interpretation
R4 is the best compromise of the three: it keeps the most overpasses at L0-L1 (100% / 76%, more
than the graph's 94% / 69%) and matches the mask's robustness (L1 APLS 0.569 vs 0.572, the graph
0.522), but under corruption its junctions are no better than the mask's because tuning switches
to the surface skeleton from L1 on (the thin centreline is too fragile). Next: a single decoder
that keeps the centreline topology and heals its gaps by min-cost paths (R6; Codex's R5 diagnosis,
notes/codex_r5_diagnosis.md).
