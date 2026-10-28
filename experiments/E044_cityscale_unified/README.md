# E044: unified road heads, trained on City-Scale (pre-registered 2026-10-28 16:20)

Why: oracles E039/E042 show the unified road heads (R4) keep overpasses with mask-level robustness
under synthetic corruption, but synthetic corruption misled us before (B2: strong oracle, weak
trained). This is the first trained test on real imagery. User authorised autonomous overnight
work (16:13) with remote/local compute as reasonable.

## Data and model
City-Scale (Sat2Graph's 20 cities, 1 m/px): 144 train / 9 val / 27 test regions of 2048 px, each cut
into four 1024 tiles (`data.py`). Targets (E039 `encode_r4` logic, `data.encode_fast`: per-edge updates restricted to the edge's bounding box, ~50x faster; on a test tile 28 of ~4M direction values differ from `encode_r4` (window rasterisation at buffer boundaries), all other channels identical): road surface (6 m wide; City-Scale
has no widths), 3 px centreline, junction / end / crossing heatmaps (crossings = City-Scale's
overpass annotations), two-mode (cos 2t, sin 2t) direction. One ResNet34 U-Net (`train44.py`),
heads mask, line, kp (3), direction (4); losses BCE+Dice (mask, line), focal with explicit
positives (keypoints, CenterNet prior bias), masked L1 (direction), weights 1; fastai_like recipe,
40 epochs, batch 4, dihedral augmentation with analytic direction transform; seed 0; local 4090.

## Arms (same predictions)
* **R0**: E028 mask skeleton decoder on the predicted surface; grid thr {0.3, 0.5} x spur {10, 30}
  x gap {0, 25, 50}, tuned on val by APLS.
* **R4**: E039 decoder (modes centre / surface / mincost; thr_k, thr_line) on all predicted heads,
  with R0's tuned settings inside, tuned on val by APLS.
Test once on the 27 test regions (tiles with >= 10 GT edges): APLS, junction F1, crossings kept.

## Amendment 1 (16:58, before any test result)
The first full run collapsed (val mask IoU 0.0000 at epoch 5). Five-epoch debug runs: mask loss
alone 0.474, + line 0.477, + direction 0.463, + keypoints at weight 0.1: 0.000. Cause: the focal
keypoint loss was normalised per channel by its positives (clamped at 1); crossings are absent from
most batches, so that channel's summed negatives were divided by 1 and swamped the decoder. Fixed:
one focal loss over the three keypoint channels normalised by their total positives (floor 16);
all four losses at weight 1 then reach val mask IoU 0.360 at epoch 5. Seeded augmentation and the
workers-0 fix (Codex review) are also in.

## Decision rule (fixed now)
Unified heads viable on real data if, on test: R4 APLS >= R0 - 0.01, R4 junction F1 >= R0 + 0.05,
and R4 crossings kept >= R0 + 0.20. (Single seed: a screen; seeds 1-2 only if it passes.)

## Results
(pending)
