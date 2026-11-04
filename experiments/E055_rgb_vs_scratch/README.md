# E055: RGB-pretrained weights vs training from scratch on non-RGB inputs (pre-registered)

Status: **registered** 2026-11-04, before any run. Local RTX 4090, no Pawsey units.

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

## Results
Pending.
