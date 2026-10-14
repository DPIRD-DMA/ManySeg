# E010: Boundary + distance instance heads for field delineation (pre-registered)

Status: **complete, positive result** (2026-10-14; 9 runs on Pawsey Setonix, AMD MI250X, code `537848a`). Dataset approved by the user: AI4Boundaries orthophoto subset.
Proposed dataset: AI4Boundaries orthophoto subset (CC BY 4.0, 5.6 GB, 512×512 RGB at 1 m,
official train 5,319 / val 1,140 / test 1,139). The official split is **random, not spatially
blocked**, so we will also report a country-held-out evaluation. The user decides; see the
session summary.

## Questions
1. Do boundary + distance heads beat the simple extent-only + connected-components baseline on
   object-level metrics (PQ, object F1, over-/under-segmentation)?
2. Do per-head task decoders beat one shared decoder for extent/boundary/distance, at equal
   training budget, and what does it cost (params, FLOPs)?

## Protocol
* ManySeg U-Net, resnet34 `tv_in1k`, recipe `fastai_like`, 40 epochs, batch 16, 512² inputs
  with random-resized crops and flips. Instance targets are generated on the GPU from the
  dataset's field-enumeration band (`boundary_width=2`, inner, normalised distance).
* Arms (3 seeds each):
  | id | heads | decoders |
  |---|---|---|
  | extent_cc | extent only; instances = connected components | shared |
  | bd_shared | extent + boundary + distance | shared |
  | bd_separate | extent + boundary + distance | one U-Net per head |
* Post-processing: `instances_from_outputs` defaults (torch backend), fixed before any run.
  No threshold tuning on test.
* Metrics: val PQ (primary), object F1@0.5, over/under-segmentation, extent IoU. Test and the
  country-held-out split are evaluated once per final run.
* Budget: 9 runs, about 3–4 h on the shared 4090. If wall-clock exceeds 6 h, move to Setonix
  after confirming the account with the user.

## Decision rule
`bd_*` "beats" `extent_cc` if mean val PQ is higher by more than 2 pooled seed-SDs and test PQ
points the same way. `bd_separate` is adopted as the recommended preset only if it beats
`bd_shared` by the same rule. Otherwise the shared decoder stays the default (cheaper).

## Results (mean ± SD, 3 seeds; val = official val split, test = official test split)

| arm | params (M) | val PQ | val object F1 | val precision | val recall | val SQ | val over-seg | val under-seg | test PQ | test F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| extent_cc | 24.4 | 0.1453 ± 0.0039 | 0.1901 ± 0.0044 | 0.2903 ± 0.0244 | 0.1419 ± 0.0078 | 0.7643 ± 0.0027 | 0.1850 ± 0.0064 | 0.7777 ± 0.0106 | 0.1507 ± 0.0027 | 0.1945 ± 0.0028 |
| bd_shared | 24.4 | 0.2898 ± 0.0239 | 0.3555 ± 0.0285 | 0.2716 ± 0.0319 | 0.5171 ± 0.0046 | 0.8151 ± 0.0020 | 0.2227 ± 0.0080 | 0.4143 ± 0.0065 | 0.2982 ± 0.0244 | 0.3622 ± 0.0289 |
| bd_separate | 30.7 | **0.3521 ± 0.0079** | **0.4319 ± 0.0102** | 0.3691 ± 0.0192 | 0.5213 ± 0.0082 | 0.8153 ± 0.0010 | 0.2233 ± 0.0015 | 0.4162 ± 0.0104 | **0.3601 ± 0.0065** | **0.4371 ± 0.0081** |

Pre-registered comparisons (val PQ difference in pooled seed SDs; test difference):
* bd_shared − extent_cc: +0.145 (+8.4 SD); test +0.148.
* bd_separate − extent_cc: +0.207 (+33.1 SD); test +0.209.
* bd_separate − bd_shared: +0.062 (+3.5 SD); test +0.062.

## Interpretation
* **Boundary + distance heads beat extent + connected components** decisively: PQ roughly
  doubles. The mechanism is visible in the error decomposition. Under-segmentation (merged
  touching fields) drops from 0.78 to 0.41, and recall of individual fields goes from 0.14 to
  0.52. Segmentation quality of matched fields (SQ) also rises, 0.76 → 0.82.
* **Per-head decoders help further** (+0.06 PQ, mainly through precision 0.27 → 0.37) for +6.3 M
  parameters (+26%). Separate decoders stop the boundary and distance objectives from
  competing for one decoder's capacity.
* **Absolute PQ is modest (0.35).** AI4Boundaries' GSAA parcel labels are known to be incomplete
  (unlabelled fields count as background), which depresses precision. Post-processing
  thresholds were fixed in advance and not tuned. Threshold tuning on val is the obvious,
  separately labelled follow-up.
* Training curves were still rising at 40 epochs (bd_shared seed 0: val PQ 0.13 at epoch 10 →
  0.28 at 35), so all arms are under-trained by an equal budget.

## Decision
* Recommend `instance_heads(decoders="separate")` for field delineation in the docs, with the
  shared-decoder preset as the cheaper option.
* Keep extent + CC only as a baseline. It merges touching fields.
* Follow-ups: post-processing threshold calibration on val; a country-held-out evaluation (the
  official split is random); longer training.
