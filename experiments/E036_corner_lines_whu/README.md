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
instance loss 3.6, corner focal 2.3. Code: `manyseg.vectors.corner_targets`,
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
