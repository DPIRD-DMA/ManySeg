# E032: completeness losses for the graph method (pre-registered 2026-10-27 11:00)

Status: **complete: screened out on seed 0** (2026-10-27; see Results). Plan in `PLAN.md`, reviewed by Codex (`notes/codex_plan_e032.md` in the
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

## Amendment 1 (2026-10-27 11:35, before any completed run)
The first B2e run (seed 0) diverged: at epoch 10, training loss 20.3 and val extent IoU 0.14,
where E030's B2 had about 0.7 and 0.87. It was stopped and discarded. The destination term's
−log(sample + 1e-4) has gradients that grow as the sampled target value falls towards zero, which
happens at the vertex blob's 3 px cut-off (about 70× the bounded version's on a synthetic check,
more for values closer to zero). The term is now **1 − sample**: same optimum, bounded gradients.
On 600 tiles × 3 epochs the corrected B2e trains normally (loss 4.18 → 2.53, extent IoU 0.33 →
0.63). The cause is the most likely one, not proven (the run's per-term losses were not logged).
Added: E030's training-failure check applies (val extent IoU below 0.5 at the end = broken run,
fixed and rerun as an amendment).

## Amendment 2 (2026-10-27 12:10, before any E032 result): staged seeds
At the user's request, to shorten iteration: every arm first runs **seed 0 only**. An arm goes on
to seeds 1–2 only if its seed-0 **validation** C-IoU × detection F1 (the tuned decoder's val score)
beats B2 seed 0's by more than 0.005; test is not looked at for this. The decision rule is
unchanged and is applied only to arms with all 3 seeds; arms stopped after seed 0 are reported as
"screened out on one seed", with no claim beyond that. Run order: B2e, B2w (E032), then B2d, B2s
(E033), then the advanced arms' seeds 1–2.

## Amendment 3 (2026-10-27 12:45, agreed with the user after a Codex review of the options, `E032_graph_completeness/OPTIONS.md`; no result of the affected runs exists)
B2w is dropped (it optimises the wrongly defined target: arbitrary densified points; see the vertex analysis in OPTIONS.md). B2e seed 0 (running) finishes and is screened. Controls for seeds 1–2, if needed, are trained then (E030 amendment 3).

## Results

Status: **complete: screened out on seed 0** (2026-10-27). Validation C-IoU × F1 (tuned decoder),
the screening metric: B2 0.440, **B2e 0.411** (B2w dropped, amendment 3). Test, for the record:
B2e C-IoU 0.604, detection F1 0.706, C-IoU × F1 0.426 (B2 0.613 / 0.750 / 0.460), recall 0.81.
The completeness losses (decode-cost weighting + bounded endpoint term) made closure slightly
worse, not better; no seeds 1–2 (staged-seed rule). Single seed: "not promising", no stronger
claim.
