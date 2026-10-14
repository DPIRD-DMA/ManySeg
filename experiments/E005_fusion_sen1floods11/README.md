# E005: Input fusion strategies on Sen1Floods11 (pre-registered)

Status: **planned** (registered 2026-10-14 before any E005 runs).

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
