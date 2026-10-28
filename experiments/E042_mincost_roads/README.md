# E042: R6 = R4 + min-cost-path gap healing on the centreline (pre-registered 2026-10-28 16:20)

Why: E039's R4 kept every overpass at L0 and matched the mask's robustness, but failed the junction
gain at L1 because tuning switched to the surface skeleton (thin centreline too fragile). Codex's
R5 diagnosis: build ordered paths first, then heal gaps by minimum-cost paths. R6 adds mode
`mincost` to E039's decoder (`e039.mincost_heal`): from each dead end of the centreline graph (not
at the border, not near an end keypoint), a geometric min-cost path over cost = min(-log(max(line,
0.3 x surface)), cost_cap) + 0.05 within heal_r px to a *different* part of the graph; added if
its cost < heal_cost. Smoke tiles (Murdoch 1/3/9, L0-L2) set the cap (uncapped, occlusion gaps
cost ~7/px and never healed).

## Design
E039 harness unchanged (Innaloo tuning per level, 2 seeds, APLS; Murdoch + Northam L0-L3 seeds 0-4;
R0 and R2 at E028 settings on the same draws). Grid: thr_k 0.2, zr 8, thr_line 0.3, pair_deg 150,
gap = R0's per level; mode {centre, surface} plus mincost x (cost_cap, heal_cost, heal_r) in
{(3, 40, 50), (2, 60, 60), (7, 25, 40)} -> 5 settings. Runs on Pawsey (CPU).

## Decision rule (E039's, unchanged)
Topology gain: junction F1 >= R0 + 0.10 at L0 and L1, crossings kept >= 50% at L0; robustness:
APLS >= R0 - 0.01 at L1 and L2. Also reported: which mode tuning picks per level.

## Results
(pending)
