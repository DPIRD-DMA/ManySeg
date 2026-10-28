# E041: edge-scoring road decoder on the unified heads (R5), oracle (pre-registered 2026-10-28 14:00)

Why: the literature review (ManySeg-e028/notes/literature_unified_graphs.md) found the leading road
methods (SAM-Road / SAM-Road++, LineGraph2Road, GraphMapper) predict dense maps, take sparse
candidate nodes and *score candidate edges*, rather than tracing a skeleton. E039 showed our thin
centreline skeleton is fragile under corruption (spurs defeat gap bridging). R5 keeps the unified
heads and replaces skeleton tracing with candidate nodes + edge scoring, and uses E040's frame
field (two unordered directions per pixel; at crossings, the two roads' directions) instead of
R4's two-mode direction field. Approved by the user 13:19; code reviewed by Codex before running
(notes/codex_e040_e041.md, all 9 findings fixed: label-free use of the frame field, point-to-
segment crossing test, crossing precedence over nearby junction/end peaks, across-crossing edges
placed first, unused direction head dropped, this harness).

## R5
Heads: road surface, 3 px centreline band, junction / end / crossing heatmaps (E039), frame field
(u = v = road direction; two roads' directions where different chains overlap at > 20 deg).
Decoder: nodes = junction and end peaks + greedy maxima of the smoothed centreline band >= spacing
apart, none within r_cross of a crossing peak (a junction/end peak near a crossing peak is kept
only if its heatmap value is higher); candidate edges = node pairs within 2.2 x spacing, or 2.6 x
spacing if the segment passes within r_cross of a crossing peak; score = mean road probability
along the segment (centreline or surface, tuned) x mean frame agreement (max over u, v of cos^2;
0.5 where weak); greedy: across-crossing edges first, then edges at keypoint nodes, then by score with degree caps (end 1, junction 6,
others 2) and 25 deg angular suppression; components < 20 px dropped; E028 end bridging with R0's
tuned gap. Grid: thr_k {0.2, 0.4} x spacing {10, 14} x thr_e {0.2, 0.35} x edge source {line, mask}.
Tuned on Innaloo per level (2 seeds, APLS); tested on Murdoch + Northam, L0-L3, seeds 0-4, with R0, R2 (E028 settings) and R4 (E039's tuned settings) on the same draws.

## Design note (smoke tile Murdoch 3 at L0, before any tuning or test)
The first version left 44 components (GT 2): band nodes beside a junction used their 2-edge cap
linking past it, and greedy maxima can sit up to ~2x spacing apart. Fixed by placing edges at
keypoint nodes before band-band edges, and a 2.2x (was 1.6x) spacing candidate radius.

## Decision rule (as E039's, for R5)
1. Topology gain: R5 junction F1 >= R0's + 0.10 at L0 and L1, and crossings kept >= 50% at L0.
2. Robustness kept: R5 APLS >= R0's - 0.01 at L1 and L2.
Both -> R5 is viable; reported against R4 and R2 as well.

## Results
(pending)
