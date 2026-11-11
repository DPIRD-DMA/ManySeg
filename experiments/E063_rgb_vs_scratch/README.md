# E063: RGB-pretrained weights vs training from scratch on non-RGB inputs (pre-registered)

Numbered E055 while it ran and renumbered on 2026-11-11, because E055 was taken on `main` by
another experiment. Run tags (`e055a-`, `e055b-`, `e055c-`), scripts' unit names and the frozen
worktree name (`e055-7a3a544`) keep the old number.

Status: **done** (registered 2026-11-04, before any run; finished 2026-11-05). Local RTX 4090, no
Pawsey units.

**Answer:** not in general. Against the better scratch recipe, RGB weights gained 4.2 points on
CloudSEN12+ and 1.0 on PASTIS, and lost 0.6 on Sen1Floods11 and 1.0 on BigEarthNet. Scarce labels
did not change the Sen1Floods11 result.

## Question
Does starting a multispectral encoder from ImageNet RGB weights beat starting from random
weights? E004 found no difference at the end of training, but on one dataset (Sen1Floods11),
one encoder (ResNet-34) and with all labels. Its scratch arm also used `fastai_like`, so the
random encoder was frozen for the first epoch and then trained at 0.1x LR.

* **Part A:** do other datasets and a ResNet-50 change the answer?
* **Part B:** do RGB weights help when labels are scarce?

## Arms
| arm | weights | recipe | why |
|---|---|---|---|
| `imagenet` | `tv_in1k`, `repeat` stem | `fastai_like` | the default |
| `scratch` | none | `fastai_like` | changes the weights only (as in E004) |
| `scratch_fair` | none | `fastai_like`, `encoder_lr_mult` 1.0, no frozen epochs | a random encoder has no filters to protect |

All arms use per-band dataset statistics. Neither recipe was tuned for this comparison.

## Part A: three datasets, all labels
Same protocol as E024 (ResNet-50; epochs and batch sizes as there; seeds 0-2). 2 arms x 3 datasets
x 3 seeds = 18 new runs from the frozen commit E024 used (`9a2d519`), compared with E024's
`imagenet` runs, so the code does not change between arms. Sen1Floods11 and PASTIS are U-Nets,
BigEarthNet an image head. E024's runs took about 3, 19 and 29 min per arm of 3 seeds.

## Part B: scarce labels, Sen1Floods11 S2
ResNet-50 U-Net, 3 arms x 4 sizes x 5 seeds = 60 runs, all from one new frozen commit.

| train chips | share | batch | steps/epoch | epochs | `imagenet` frozen epochs |
|---|---|---|---|---|---|
| 12 | 5% | 8 | 1 | 900 | 15 |
| 25 | 10% | 16 | 1 | 900 | 15 |
| 63 | 25% | 16 | 3 | 300 | 5 |
| 252 | 100% | 16 | 15 | 60 | 1 |

* Every size trains for 900 optimiser steps, with the first 15 frozen for `imagenet` and
  `scratch`, so less data does not also mean less training.
* `--train-subset N` draws chips by seed: arms with a seed share chips, and sizes nest. Val, test
  and normalisation statistics stay those of the full split.
* The 12-chip size needs batch 8 because the loader drops short batches, so compare arms within
  a size, not across sizes.

## Part C: CloudSEN12+ (added 2026-11-04, before any CloudSEN12 run)
Added at the user's request, because Sen1Floods11 may be too easy to separate the arms. Part B was
complete at this point, part A had finished Sen1Floods11 and one BigEarthNet seed.

* `cloudsen12`: 13-band L1C, 4 classes, 19,450 train / 1,767 val / 2,335 test chips, with
  ROI-disjoint splits, so test locations are unseen. Metric: val mIoU.
* ResNet-50 U-Net, `fastai_like`, 20 epochs and batch 16 as in E013, all three arms, seeds 0-2.
  9 new runs from frozen commit `7a3a544`, one seed of all arms at a time, on the local 4090. All
  arms run here, because E013's baseline is a ResNet-34 on different hardware.
* Decision: the part A per-dataset rule, with this part's own `imagenet` runs. The docs rule above
  stays on its 3 registered datasets; CloudSEN12 is reported as further evidence beside it.

## Decision rules
Primary metric: val mIoU (BigEarthNet: macro mAP) at the final epoch. Test is a veto only. Bolivia
is reported but does not decide anything (n = 15). "Scratch" below means both scratch arms.

**Part A, per dataset.** RGB weights *help* if `imagenet` beats both scratch arms by more than 2
pooled SD on val, with test agreeing. They *hurt* if either scratch arm beats `imagenet` by more
than 2 pooled SD on val, with test agreeing. Otherwise *level*.

**Part B, per size.** Paired by seed, since arms share chips. RGB weights *help* if the 95% CI of
the mean difference (t, df 4) excludes 0 in `imagenet`'s favour against both scratch arms, with
test agreeing. They *hurt* if the CI excludes 0 in favour of either scratch arm, with test
agreeing. Otherwise *level*. No correction for the number of sizes, so one isolated result at one
size is weak evidence.

**Docs.** Say RGB weights help in general only if they help on at least 2 of 3 Part A datasets and
hurt on none. Say they help with scarce labels only if they help at 12 or 25 chips.

## Limits known in advance
One recipe family, one encoder, and only Sentinel-2 optical bands: SAR, height and other sensors
stay untested. Part B has one dataset. 900 steps at 12 chips is many passes over few images, which
may favour whichever arm overfits least.

## Run log
* 2026-11-04 14:48: system restart during part B. The 45 finished runs were kept; the 252-chip
  size and part A then ran from the same commits and seeds.
* 2026-11-04 15:32: job memory cap raised live from 16 GB to 40 GB. The cap counts page cache, so
  BigEarthNet was re-reading from disk (epochs about 95 s, against 29 s in E024). Speed only.

## Results
Tables: `uv run python experiments/E063_rgb_vs_scratch/analyse.py`. All 87 runs
completed, no non-finite losses. Arms sharing a seed started from the same random weights.

### Part A: three datasets, all labels
Final epoch, mean ± SD over seeds 0-2. Differences are `imagenet` minus the arm, in points (val,
with 2 pooled SD; test).

| dataset | `imagenet` val | `scratch` val | `scratch_fair` val | vs `scratch` | vs `scratch_fair` | RGB weights |
|---|---|---|---|---|---|---|
| Sen1Floods11 | 0.8882 ± 0.0038 | 0.8937 ± 0.0007 | 0.8923 ± 0.0016 | −0.55 (0.54); −0.68 | −0.41 (0.58); −0.93 | hurt |
| BigEarthNet | 0.7381 ± 0.0031 | 0.6911 ± 0.0043 | 0.7481 ± 0.0016 | +4.70 (0.75); +2.51 | −1.00 (0.50); −1.76 | hurt |
| PASTIS | 0.2097 ± 0.0022 | 0.1936 ± 0.0007 | 0.1999 ± 0.0040 | +1.61 (0.33); +1.37 | +0.98 (0.65); +0.79 | help |

* **Docs rule:** RGB weights help on 1 of 3 datasets and hurt on 2, so no general claim.
* **The recipe matters as much as the weights.** On BigEarthNet, full encoder LR moves scratch by
  5.7 points, from well below `imagenet` to above it. E004-style scratch arms understate scratch.
* The Sen1Floods11 "hurt" is marginal: it passes by 0.01 points against `scratch` only.
* `imagenet` was not given a second recipe, while scratch was, so scratch had one more chance.

### Part B: scarce labels, Sen1Floods11
Mean over 5 seeds. Differences are `imagenet` minus the arm, in val points, with the 95% CI.

| chips | `imagenet` | `scratch` | `scratch_fair` | vs `scratch` | vs `scratch_fair` | RGB weights |
|---|---|---|---|---|---|---|
| 12 | 0.8514 | 0.8524 | 0.8539 | −0.10 ± 2.62 | −0.25 ± 1.72 | level |
| 25 | 0.8462 | 0.8597 | 0.8615 | −1.35 ± 2.21 | −1.53 ± 3.11 | level |
| 63 | 0.8647 | 0.8652 | 0.8762 | −0.05 ± 1.38 | −1.14 ± 1.50 | level |
| 252 | 0.8898 | 0.8939 | 0.8920 | −0.41 ± 0.65 | −0.21 ± 0.51 | level |

* **Docs rule:** no help at 12 or 25 chips, so no scarce-label claim. Intervals of ±2-3 points at
  those sizes leave room for a small benefit.
* `imagenet` at 252 chips (0.8898) matches E024's 0.8882 on the older commit.
* Bolivia mIoU (not decisive, n = 15): `imagenet` is lowest at every size.

### Part C: CloudSEN12+
Final epoch, mean ± SD over seeds 0-2, in the same format as part A.

| dataset | `imagenet` val | `scratch` val | `scratch_fair` val | vs `scratch` | vs `scratch_fair` | RGB weights |
|---|---|---|---|---|---|---|
| CloudSEN12+ | 0.7382 ± 0.0006 | 0.6822 ± 0.0072 | 0.6960 ± 0.0049 | +5.60 (1.02); +5.67 | +4.22 (0.70); +3.82 | help |

Mean test IoU per class:

| arm | clear | thick cloud | thin cloud | shadow |
|---|---|---|---|---|
| `imagenet` | 0.909 | 0.872 | 0.408 | 0.584 |
| `scratch` | 0.887 | 0.818 | 0.346 | 0.496 |
| `scratch_fair` | 0.894 | 0.843 | 0.380 | 0.504 |

* **Largest gain in this study.** `imagenet` leads both scratch arms on every class, and full
  encoder LR does not close the gap as it did on BigEarthNet.
* **20 epochs may flatter `imagenet`.** Its val lead over `scratch_fair` narrows from 8.1 points at
  epoch 10 to 5.2 at epoch 15 and 4.2 at the end. Every run's best epoch was 17-20, so longer
  training might narrow it further; this was not tested.
* `imagenet` (val 0.738, test 0.693) is in line with E013's ResNet-34 baseline (val 0.732, test
  0.685). The encoder and hardware differ, so this is a sanity check only.

### Overall
| dataset | `imagenet` minus better scratch arm, val points | RGB weights |
|---|---|---|
| CloudSEN12+ (beside the docs rule) | +4.22 | help |
| PASTIS | +0.98 | help |
| Sen1Floods11 | −0.55 | hurt |
| BigEarthNet | −1.00 | hurt |

* **Docs rule:** still no general claim. On the 3 registered datasets RGB weights help on 1 and
  hurt on 2. CloudSEN12+ makes it 2 of 4 each way.
* The losses are small and the largest gain is not, which supports keeping `tv_in1k` with `repeat`
  as the default. It does not show that RGB weights always help.
* Four datasets are too few to say what decides the outcome, so this study offers no rule for
  predicting it on a new dataset.
* Not run: `imagenet` under a second recipe, which could change the two "hurt" verdicts, and a
  `projection` rerun.
