# E010: Boundary + distance instance heads for field delineation (pre-registered)

Status: **planned, blocked on dataset approval** (registered 2026-10-14 before any run).
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
