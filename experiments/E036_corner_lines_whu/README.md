# E036: corners + lines (B3), trained on WHU (pre-registered 2026-10-28 09:45; user approved)

Why: E035b (oracle) showed the corner-trusting B3 decoder keeps B0's closure and beats B0's
polygons at every corruption level, but its registered rule compared it with B2, which is the
weakest arm once trained (E030: B2 C-IoU × F1 0.46 vs B0 0.60). This experiment compares B3 with
the trained baseline that matters, B0.

## Setup (E030's: WHU aerial 0.3 m, resnet34 U-Net, `fastai_like`, 40 epochs, batch 16,
dihedral augmentation, 8 loader workers, decoder tuned on val, test once)
**B3:** B0's instance heads (extent, boundary, distance; bench `InstanceTask` loss) + a `corner`
head (CenterNet focal loss, positives = the target's local maxima above 0.5, on Gaussian blobs, sigma 2 px, at vertices turning > 20° after a 1.5 px simplification of
the GT polygon, so raster staircase jogs are not corners) + a 2-channel `direction` head (masked L1
on `(cos 2θ, sin 2θ)` of the simplified walls within 1 px); the three loss groups weighted 1. The
decoder uses the direction's orientation only (unit-normalised; norm < 0.1 = no evidence).
Pre-launch Codex review (notes/codex_e036.md) changed the corner loss from plain BCE (sparse
target, constant-low optimum) and added the normalisation; corners on pixel edges peak at 0.94,
so the default `>= 0.999` positive rule would have found none. At init on 16 WHU train tiles:
instance loss 3.6, corner focal 2.3 with the corner bias initialised to −2.19 (CenterNet prior, p = 0.1; without it the preflight epoch averaged loss 199). Code: `manyseg.vectors.corner_targets`,
`polygons_from_corners`; `E030_whu_polygons/train.py --arm B3`. Decoding: instances with B0 seed
0's tuned thresholds (extent 0.6, boundary 0.5, seed 0.15), then `polygons_from_corners`, tuned on
val over 16 settings (`corner_threshold` {0.3, 0.5}, `big_dev` {4, 8}, `agree` {0.4, 0.6},
`merge_deg` {8, 15}).

## Decision (fixed 2026-10-28 09:45, before launch)
Seed 0 first; compared with B0 seed 0 on val C-IoU × detection F1 (B0_s0: 0.6277, from
`e030-aadd0f2/.../runs/B0_s0_eval.json`). If B3 is higher by more than 0.005 (B3_s0 val > 0.6327), seeds 1–2 for B3 and B0 (B0 seed 1 is already trained; its evaluation is still to run), then the paired rule (mean paired
difference ≥ 0.02 and > 2 SDs; detection F1 not more than 0.02 below B0). Also reported: C-IoU on
GT outlines cleaned of staircase vertices (post hoc), detection F1, matched IoU, PoLiS, N-ratio.

## Budget
One run (~2–2.5 h) on the shared 4090 after agreeing a slot with the Cloud-Aug session.

## Results (seed 0, 2026-10-28)
Trained 09:45-10:54 (40 epochs, final val PQ 0.649); decoder tuned on val over the 16 settings
(best: corner_threshold 0.3, big_dev 4, agree 0.6, merge_deg 8), test once.

| split | arm | C-IoU x F1 | C-IoU | det F1 | matched IoU | PoLiS m | N-ratio | TP / pred |
|---|---|---|---|---|---|---|---|---|
| val | B0 s0 | **0.628** | 0.736 | 0.853 | 0.878 | 0.370 | 1.14 | 14,382 / 16,931 |
| val | B3 s0 | 0.482 | 0.581 | 0.829 | **0.884** | **0.344** | 0.44 | **14,606** / 18,436 |
| test | B0 s0 | **0.603** | 0.709 | 0.850 | 0.861 | 0.391 | 1.18 | 41,099 / 48,751 |
| test | B3 s0 | 0.505 | 0.609 | 0.828 | **0.867** | **0.364** | 0.48 | **41,623** / 52,576 |

### Decision (registered screen)
B3 val 0.482 is not above 0.6327 → **B3 does not advance; no seeds 1-2.**

### Interpretation
B3 loses on C-IoU's vertex-count term, not on overlap: its matched IoU and PoLiS are slightly
better than B0's and it recovers more buildings, but it outputs 0.44x the vertices of WHU's
raster-traced GT (B0: 1.14x), whose outlines keep many staircase vertices (E037). Its lower
detection F1 comes from extra predictions (18,436 vs 16,931), plausibly because the registered
decoder fixed B0's instance thresholds on B3's own, jointly trained heads.

### Post hoc: staircase-cleaned GT (registered as post hoc; does not change the decision)
300 evenly spaced val tiles (`posthoc.py`, `posthoc.json`, `posthoc_tiles.png`); GT outlines
regularised with buildingregulariser 0.2.5 (parallel_threshold 3 px, simplify_tolerance 1.5 px,
as E037). Added the obvious baseline **B0 + regulariser** (B0's polygons through the same tool).

| arm | raw GT: C-IoU x F1 | regularised GT: C-IoU x F1 | C-IoU | matched IoU | PoLiS m | N-ratio | det F1 |
|---|---|---|---|---|---|---|---|
| B0 | **0.626** | 0.460 | 0.541 | 0.862 | 0.444 | 2.20 | **0.850** |
| B0 + regulariser | 0.522 | **0.632** | 0.751 | 0.853 | 0.447 | 1.10 | 0.842 |
| B3 | 0.482 | 0.628 | **0.760** | **0.868** | **0.400** | 0.85 | 0.825 |

The winner depends on the GT's vertex convention: against raw (staircase) GT B0 wins; against
regularised GT B3 ties with B0 + regulariser (whose output comes from the same tool as the GT)
and has the best outline geometry (C-IoU, matched IoU, PoLiS), losing on detection F1. Renders:
B3's walls are straighter; B0 + regulariser sometimes adds square notches; B3 sometimes cuts a
corner on complex shapes; all three follow the mask where trees occlude a wall.
Next if pursued (new pre-registration): B3 with its own tuned instance thresholds, judged on
regularised GT against B0 + regulariser.
