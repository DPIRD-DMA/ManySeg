# E032: completeness losses for the graph method (pre-registered 2026-10-27 11:00)

Status: **pre-registered**. Plan in `PLAN.md`, reviewed by Codex (`notes/codex_plan_e032.md` in the
E028 worktree); changes from the review are applied below.

## Question
The graph-tensor loss is per pixel, so nothing in training reflects that one missed vertex loses
a whole building at decode time. Do loss terms aimed at completeness give more complete, still
clean polygons?

## Setup
Exactly as E030 (WHU aerial, resnet34 U-Net, `fastai_like`, 40 epochs, batch 16, dihedral
augmentation, decoder tuned on val by C-IoU × detection F1, test once), code in
`../E030_whu_polygons` (`train.py --loss`). Control: E030's B2 runs (seeds 0–2), re-evaluated
with the same evaluation code as the new arms.

| arm | loss |
|---|---|
| B2 (control) | E030: BCE vertex + BCE edge on support + smooth-L1 vector + dice-BCE extent |
| B2w | B2 + decode-cost weighting: vertex BCE weight 1 + 10·target; edge BCE pos_weight 5 |
| B2e | B2w + 0.1 × destination loss: at pixels within ~1.5 px of a target vertex centre with a true edge in sector k, sample the **target** vertex map at the predicted endpoint (bilinear, `align_corners=False`; endpoints within half a pixel of the tile edge masked), −log(sample + 1e-4) |

Cut after review: the degree loss (degenerate: spreading 1/3 over six sectors satisfies it, and
it fights the per-sector BCE); reciprocity (needs geometric, not opposite-sector, matching);
topology losses (need an edge-line target). The destination loss samples the fixed target, so it
cannot be lowered by raising the predicted vertex map.

## Decision rule (fixed before any run)
Primary comparison **B2e vs B2**, metric **C-IoU × detection F1** on test, paired by seed:
B2e wins if the mean paired difference is at least 0.02 and more than 2 SDs of the paired
differences, and its C-IoU is no more than 0.02 below B2's. Then the B2e loss becomes the default
for graph heads. B2w vs B2 is reported as an ablation (no decision). Also reported: detection
F1, C-IoU, PoLiS, N-ratio, and recall (the share of GT buildings matched by a closed predicted
polygon at IoU > 0.5). Three seeds
are few; a borderline result is reported as inconclusive.

## Budget
6 runs (B2w, B2e × 3 seeds), about 75 min each including evaluation, on the local 4090 in the
12 GB capped service, queued after E030.

## Results
(pending)
