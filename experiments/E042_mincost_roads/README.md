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

## Results (Pawsey job 51716083, 6 min 43 s, code vec-cc37971)
Tuning (Innaloo) chose `mincost` at L0 (cap 3, cost 40, r 50) and L1-L2 (cap 2, cost 60, r 60),
`surface` at L3. Test, pooled Murdoch + Northam (`runs/decision.json`):

| arm | APLS L0 / L1 / L2 | junction F1 L0 / L1 / L2 | crossings kept L0 / L1 / L2 |
|---|---|---|---|
| R0 | 0.936 / 0.572 / 0.210 | 0.759 / 0.651 / 0.314 | 0% / 2% / 8% |
| R2 | 0.985 / 0.514 / 0.118 | 0.976 / 0.795 / 0.218 | 94% / 70% / 38% |
| R6 (R4 + mincost) | 0.983 / 0.553 / 0.186 | 0.881 / 0.466 / 0.156 | 100% / 39% / 27% |

**Decision:** topology gain fails (junction F1 L1 -0.185 vs R0), robustness fails (APLS L1 -0.019).
**R6 is not viable**, and worse at L1 than E039's R4 (surface mode at L1: APLS 0.569, junction 0.683,
crossings 76%): min-cost healing won on the tuning area but did not transfer. Conclusion from the
oracles: under E028's synthetic corruption the thin-centreline topology does not survive, however
its gaps are healed; R4 (surface skeleton + keypoint pairing under corruption) stays the best road
decoder. Whether a trained network's errors are kinder to the thin centreline is E044's question.
