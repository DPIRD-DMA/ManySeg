# E034: wall-tolerant vertex loss for the graph method (pre-registered 2026-10-27 12:45)

Status: **pre-registered**. Motivated by an analysis of the trained B2 (E030 seed 0, 200 val tiles,
`OPTIONS.md`): the outline is densified to points every ≤ 20 px, whose positions along a wall are
arbitrary (they depend on the wall's whole length); 72% of B2's spurious vertex peaks (15,783 of
21,975) lie on a true wall, and densified wall points are recovered less often than corners (0.40
vs 0.50 within 2 px). The vertex loss punishes a vertex placed anywhere on the correct wall except
the target points. Reviewed by Codex as the only B2 loss variant worth a full run.

## Arm
**B2t:** as B2 (E030 setup), but in the vertex BCE, pixels with target 0 that lie within 1.5 px of
the true outline (the graph's edges, rasterized) get weight 0. Corners and the densified points
stay positive targets (the vectors still point at them); predictions on a wall are neither
rewarded nor penalised. Decoder unchanged (linking already tolerates 6–10 px).

## Screening and decision
Seed 0 first; screened on validation C-IoU × detection F1 against B2 seed 0 (advance if higher by
more than 0.005), as E032 amendment 2. Seeds 1–2 with B2 control seeds 1–2 if it advances; then
E032's paired rule (mean paired difference ≥ 0.02 and > 2 SDs, C-IoU not more than 0.02 lower).
Detection F1, matched IoU, PoLiS and recall are reported alongside.

## Results

Status: **complete: screened out on seed 0** (2026-10-27; first run OOM-killed by its own 12 GB
cgroup cap at epoch ~15 and rerun from scratch with 8 loader workers and a 16 GB cap). Validation
C-IoU × F1: B2 0.440, **B2t 0.411**. Test: B2t C-IoU 0.619, detection F1 0.708, C-IoU × F1 0.439,
PoLiS 0.38 m (B2 0.613 / 0.750 / 0.460 / 0.39 m). Removing the penalty for on-wall vertices
slightly sharpened geometry but lost more buildings. Single seed: "not promising".
