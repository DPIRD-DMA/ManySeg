# E056: Do separate encoders help when the second input carries the task? (MFNet RGB + thermal)

Status: **done, negative at the screen** (2026-11-09, RTX 3090, torch 2.14.1+cu130, seed 0
only). Separate encoders beat stacking by +0.3 test mIoU, below the 1-mIoU screen, so no
further seeds were run.

## Question
E005 (Sen1Floods11, S1 + S2) and E017 (CloudSEN12+ band groups) found no gain from separate
encoders over stacking. On Sen1Floods11 the optical input alone matched every fusion arm, so the
second input added little. MFNet is the opposite case. About half the images are at night, where
RGB is close to useless and thermal carries the scene. Thermal statistics are also far from the
ImageNet RGB a shared pretrained stem was trained on. RTFNet and CMX report two-encoder fusion
beating early stacking here.

## Protocol
* Data: MFNet (Ha et al., IROS 2017), 480×640, 9 classes (0 = unlabelled, car, person, bike,
  curve, car stop, guardrail, colour cone, bump), from the `chenzeyang1/RGB-T_datasets` HF copy.
  Test = `test_day.txt` + `test_night.txt` (393 = 205 day + 188 night; `test.txt` lists one extra
  id, 01203N, that is in neither list and is dropped). `train_noflip.txt` (1,176 ids = official
  train + val; the file has no trailing newline, so `wc -l` shows 1,175) is split every third image into val (392) and train (784, the official train size).
  Val is for monitoring only.
* Model: ManySeg U-Net, ResNet-34 ImageNet weights per encoder, segmentation head, cross-entropy.
  Random scale 0.75–1.5, 384×512 crop, flip; full-resolution eval. AdamW 3e-4, one-cycle,
  80 epochs, batch 8, bf16. RGB uses ImageNet normalisation; thermal is standardised with
  train-set mean and SD.
* Arms: `rgb`, `thermal`, `early` (4-channel stack, one encoder; **baseline**), `mid_concat` and
  `mid_gated` (one encoder per input, intermediate fusion).
* Metric: test mIoU over the 9 classes (the literature convention, unlabelled included), also
  without unlabelled, and split by day and night. Classes with zero union are skipped.
* Code reviewed by Codex before launch. Fixes taken: label resizing used `nearest` against
  bilinear images (grid misalignment), changed to `nearest-exact`; mIoU skipped classes absent from
  the labels even when predicted, now uses union > 0.

## Decision rule
* Separate encoders help if `mid_*` − `early` test mIoU > 2 pooled SD over 3 seeds, and night
  mIoU moves the same way.
* Seed 0 screen: go to seeds 1–2 if the best `mid_*` arm beats `early` by more than 1 mIoU.

## Results (seed 0; 9-class mIoU)

| arm | test | test day | test night | val | params | train min |
|---|---|---|---|---|---|---|
| rgb | 0.508 | 0.466 | 0.493 | 0.715 | 24.4 M | 20 |
| thermal | 0.493 | 0.424 | 0.496 | 0.606 | 24.4 M | 20 |
| **early** (baseline) | **0.551** | 0.496 | 0.541 | 0.743 | 24.4 M | 19 |
| mid_concat | 0.554 | 0.502 | 0.543 | 0.751 | 46.4 M | 18 |
| mid_gated | 0.552 | 0.490 | 0.558 | 0.747 | 45.8 M | 29 |

Train minutes are with other jobs sharing the GPU; compare them loosely. `early` first failed at
start-up with a CUDA OOM (four jobs on the GPU) and was re-run alone after the thermal-statistics
step was made chunked; the statistics are numerically the same.

* **Decision:** best `mid_*` − `early` = +0.3 test mIoU (concat), below the 1-mIoU screen.
  Separate encoders did not clear the screen, so stop at one seed. One seed doesn't establish
  that they are equivalent to stacking, only that no gain worth chasing showed up.
* Fusion itself helps a lot: `early` is +4.2 over RGB alone and +5.8 over thermal alone. Unlike
  Sen1Floods11, both inputs carry the task (RGB is far from useless at night: MFNet's night
  scenes are lit streets), and at seed 0 a single 4-channel encoder did as well as two
  encoders with nearly twice the parameters.
* Gated fusion is +1.7 night / −0.6 day against `early`. With one seed this can't be told apart
  from noise; it's the only hint worth following up, if anything is.
* Val is inflated (≈0.74 vs ≈0.55 test) because the val split takes every third frame of video
  sequences, so val frames neighbour train frames. Only test is used for the decision.
* With E005, this is a second dataset where separate encoders showed no worthwhile gain over
  stacking, here on one seed. The README advice ("stack first") stands.
