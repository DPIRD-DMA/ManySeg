# E005: Input fusion strategies on Sen1Floods11 (pre-registered)

Status: **complete, negative result** (2026-10-14; 35 runs, 0 failures; RTX 4090, frozen worktree `9e5bba0`).

## Questions
1. Does combining Sentinel-1 SAR with Sentinel-2 optical imagery beat the best single
   modality for flood segmentation?
2. Do ManySeg's intermediate or late fusion beat simple early fusion (channel stacking, which
   any library can do: SMP is the external baseline)?
3. Do the conclusions hold on the held-out Bolivia flood event?

## Protocol
* Data: Sen1Floods11 hand-labelled. Official train (252), valid (89) for selection,
  test (90) and Bolivia (15) reported once per final run. Preprocessing is fixed by
  manyseg-bench (see `datasets/sen1floods11.py`).
* Encoder resnet34 `tv_in1k` for every input. Non-RGB inputs use the default `repeat`
  channel adapter. U-Net decoder. Recipe `fastai_like` (or the E002 winner), 60 epochs,
  batch 16, 256² random-resized crops with flips, full 512² evaluation.
* Configs (5 seeds each, because the dataset is small and variance high):
  | id | provider | data | fusion |
  |---|---|---|---|
  | s1 | manyseg | S1 only | none |
  | s2 | manyseg | S2 only | none |
  | smp-early | smp | S1+S2 stacked (15 ch) | early |
  | ms-early | manyseg | S1+S2 stacked | early |
  | ms-mid-concat | manyseg | dict | intermediate, concat, all strides |
  | ms-mid-gated | manyseg | dict | intermediate, gated |
  | ms-late | manyseg | dict | late, concat |
* Metrics: water IoU (class 1) and mIoU on val, test and Bolivia; params; FLOPs.
* Budget: 35 runs, about 2–3 h. No hyper-parameter tuning per config.

## Decision rule
A fusion strategy is "better" only if its mean water IoU beats both early fusion and the best
single modality by more than 2× the pooled seed SD on val **and** points the same way on test.
Bolivia is reported but, at n=15 chips, treated as indicative only.

## Results (water IoU = class 1; mean ± SD over 5 seeds)

| arm | val water IoU | test water IoU | Bolivia water IoU | val mIoU | Welch t vs ms-early (val / test) | t vs s2 (val / test) |
|---|---|---|---|---|---|---|
| s1 (SAR only) | 0.6148 ± 0.0158 | 0.6211 ± 0.0090 | 0.5220 ± 0.0798 | 0.7816 ± 0.0089 | −23.8 / −24.8 | −24.6 / −34.5 |
| s2 (optical only) | 0.7962 ± 0.0047 | **0.8092 ± 0.0082** | 0.7680 ± 0.0172 | 0.8847 ± 0.0026 | −2.5 / +0.8 | |
| smp-early | 0.8075 ± 0.0138 | 0.8034 ± 0.0125 | 0.7439 ± 0.0356 | 0.8911 ± 0.0079 | +0.1 / −0.0 | +1.7 / −0.9 |
| ms-early | 0.8071 ± 0.0087 | 0.8036 ± 0.0137 | 0.7487 ± 0.0240 | 0.8908 ± 0.0049 | | +2.5 / −0.8 |
| ms-mid-concat | 0.8111 ± 0.0110 | 0.8033 ± 0.0096 | 0.7408 ± 0.0417 | 0.8932 ± 0.0063 | +0.6 / −0.0 | +2.8 / −1.1 |
| ms-mid-gated | 0.8062 ± 0.0088 | 0.8022 ± 0.0081 | 0.7367 ± 0.0534 | 0.8905 ± 0.0051 | −0.2 / −0.2 | +2.2 / −1.4 |
| ms-late | 0.7982 ± 0.0098 | 0.7807 ± 0.0058 | **0.7726 ± 0.0153** | 0.8861 ± 0.0054 | −1.5 / −3.4 | +0.4 / −6.3 |

## Interpretation
* **Q1: fusion vs the best single modality. No detectable gain.** Every fusion arm is about
  +1 point water IoU over S2-only on val (t 1.7–2.8), but level or slightly lower on test. That
  fails the pre-registered rule, which requires >2 SD on val and the same direction on test.
  SAR alone is far weaker (−18 points). The hand-labelled chips are mostly clear-sky optical,
  where S2 already contains the signal. SAR would matter under cloud, which this split barely
  samples.
* **Q2: ManySeg's intermediate or late fusion vs early stacking. No.** Concat and gated
  intermediate fusion are indistinguishable from stacking. Late fusion is significantly worse
  on test (−2.3 points, t −3.4).
* **Q3: Bolivia (event holdout, n = 15).** Late fusion scores highest (0.773) and S2-only next
  (0.768), but seed SDs are 0.015–0.05, so this is indicative only, as pre-registered. It hints
  that per-modality decoders generalise better to an unseen event, but the main test split
  points the other way.
* **External parity:** SMP early fusion and ManySeg early fusion are indistinguishable
  (0.8075 vs 0.8071 val), which supports the multi-input bench path.

## Decision
No fusion default changes. Early fusion (stacking) stays the recommended baseline for
co-registered inputs. Intermediate and late fusion remain available for heterogeneous inputs
(different resolutions or backbones), where stacking is impossible; that is their actual use
case. A follow-up only makes sense on data where the modalities are complementary, e.g.
cloud-affected S2, or E009's leave-events-out protocol.
