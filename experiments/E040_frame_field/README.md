# E040: Frame Field Learning's frame field as B3's direction head, oracle (pre-registered 2026-10-28 13:40)

Why: the literature review (ManySeg-e028/notes/literature_unified_graphs.md) and the paper
(Girard et al., CVPR 2021, arXiv 2004.14875) suggest a frame field: two unoriented directions per
pixel as the complex coefficients (c0, c2) of z^4 + c2 z^2 + c0. One head can then carry walls,
corners and (for roads) crossings. If the field also gives corners (FFL's rule: a corner is where
the outline switches between the two frame directions), the unified head set loses the corner map.
Approved by the maintainer (13:19: queue the most promising approaches, write the code, Codex review).

## Arms (E035b harness: OSM vector GT, E028 corruptions, B0's instances for closure)
* **heat**: B3r as E035b (corner heatmap + (cos 2t, sin 2t) line field), its 16-setting grid.
* **field**: no corner heatmap; corners where the majority-smoothed per-point frame label (u vs v
  nearest the outline tangent, central difference over +-3 points) switches; wall test with the
  frame direction nearest the outline. Grid: window {5, 9, 15} x big_dev {4, 8} x agree {0.4, 0.6}
  x merge_deg {8, 15}.
* **both**: heatmap corners plus field corners not within 3 outline points of one; wall test with
  the frame field. Grid: thr_c {0.3, 0.5} x window {9, 15} x big_dev {4, 8} x agree {0.4, 0.6}.
* **B0** reference (E028 settings).
Targets: frame field on a 1 px band around every GT edge, u = edge tangent, v = normal (as FFL's
buildings; c2 = 0, c0 = -e^{4i theta}); corrupted like the direction field (zeroed under occlusion
disks, N(0, 0.1 x vec) per channel). Decoder samples the strongest field pixel in each outline
point's 3x3 neighbourhood (smoke-tile fix: the traced outline sits ~0.5 px off the wall).
Tuned on Innaloo per level (2 seeds, C-IoU x F1); tested on Murdoch + Northam, L0-L3, seeds 0-4,
all arms on the same corrupted draws.

## Amendment 1 (Codex review, before any tuning or test result)
* Field corners are label-free: per outline point the frame axis nearest the tangent; corners where
  that axis, smoothed over `window` points, turns > 45 deg (local maxima). The first version labelled
  roots u/v, which flip arbitrarily between pixels (sqrt branch), creating false corners.
* `polygon_b3f` returns nothing when < 3 walls remain, exactly as `polygon_b3r` (it fell back to the
  outline, an unfair detection advantage).
* The heat arm's line field is read with the same 3x3 strongest-pixel policy as the frame field.
* Known limitation: at oblique corners the overlapping 1 px target bands keep the later edge's
  frame (edge order); exact 90 deg corners are unaffected.

## Decision rule (fixed now; C-IoU x F1 pooled over the two test areas)
* **field (or both) beats heat**: >= heat + 0.01 at L1 and at L2, and >= heat - 0.01 at L0.
* **heatmap redundant**: field >= heat - 0.005 at L0, L1 and L2 (the unified head set can drop the
  corner map).
If neither holds, the frame field is not adopted for footprints (its road use is tested in E041).

## Results (2026-10-28)
Tuned locally on Innaloo (worktree e040-b5691e8); test on Pawsey (job 51716087, 3 min; code
vec-cc37971, identical decoders). C-IoU x F1 pooled over Murdoch + Northam (`runs/decision.json`):

| arm | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| B0 | 0.900 | 0.570 | 0.251 | 0.036 |
| heat (B3r) | **0.949** | **0.623** | 0.301 | 0.047 |
| field | 0.909 | 0.601 | 0.305 | **0.050** |
| both | **0.949** | 0.617 | **0.311** | 0.045 |

**Decision:** field beats heat: no (-0.040 at L0, -0.022 at L1); both beats heat: no; heatmap
redundant: no. The corner heatmap stays in the unified head set; frame-field corners are only
marginally better under heavy corruption (L2-L3, < 0.01). Tuning picked the smallest smoothing
window at L0-L1 (5) and the largest at L2-L3 (15). A B3 retrain with a frame-field head is not
justified by this oracle.
