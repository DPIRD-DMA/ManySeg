# E023: Semantic change detection: fused vs per-input land-cover heads (pre-registered)

Status: **planned** (registered 2026-10-16 before any run; epochs set by the timing rule below).

## Question
Semantic change maps land cover at both dates plus where it changed. ManySeg can predict the
two land-cover maps from the fused features, or with per-input heads (`"input": "t1"` /
`"t2"`) that read each date's own encoder features, optionally sharing one decoder and head
(`"share"`). Does the per-input design help, and does sharing?

## Data
**SECOND** (`second`): 512×512 aerial RGB pairs, 6 land-cover classes labelled where change
happened, from the torchange copy (train 2,968 / test 1,694; val = the sha1 10% of train).
One dataset only: results are single-dataset and reported as such.

## Protocol
* ResNet-34 (`tv_in1k`) U-Net, recipe `fastai_like`, batch 8, seeds 0–2, local RTX 4090.
  Loss: CE(change) + CE(land cover t1) + CE(land cover t2) on changed pixels.
* Arms (`--tag e023-<arm>`):

| arm | encoder / fusion | land-cover heads |
|---|---|---|
| `stack` | both dates stacked (early fusion) | from the fused features |
| `siam-fused` | Siamese, `concat_diff` | from the fused features |
| `siam-perinput-shared` | Siamese, `concat_diff` | per input, one shared decoder + head |
| `siam-perinput-separate` | Siamese, `concat_diff` | per input, separate decoders + heads |

  4 arms × 3 seeds = 12 runs. Epochs: the largest of 30 / 20 / 15 for which the 12 runs fit in
  6 h at the smoke-timed rate (recorded here before any run). **Smoke:** frozen first epoch
  28 s (`stack`) to 53 s (`siam-perinput-separate`); 30 epochs ≈ 7.8 h, 20 epochs ≈ 5.2 h:
  **20 epochs**.
* Metrics (Yang et al. 2021): primary SECOND **Score** = 0.3 mIoU(change / no change) + 0.7
  SeK; SeK, mIoU and change F1 reported.

## Decision rule
* Per-input heads are recommended for semantic change if the better per-input arm beats
  `siam-fused` (same encoder and fusion, land-cover heads on the fused features) by more than
  2 pooled SD of val Score with test agreeing. `stack` is a separate system-level baseline
  (early fusion), reported but not part of the head decision.
* `share` becomes the documented default for per-input change heads if `siam-perinput-shared`
  is within 2 pooled SD of, or better than, `siam-perinput-separate` (fewer parameters, same
  mapping for both dates); otherwise separate heads are documented.
* Single dataset: any recommendation is marked as resting on SECOND only.

## Amendment (2026-10-16, before any E023 run; Codex review)
* The head decision originally compared per-input heads with the better of `stack` and
  `siam-fused`; a `stack` win would then reflect early fusion, not the head source. It now
  compares against `siam-fused` only (the arms differ only in where the land-cover heads read).
* `run.sh` defaulted to 30 epochs; it now defaults to the registered 20.
* "Test agreeing" (here and in every experiment since E015) is a veto only: arms are compared
  and chosen on validation; test can block an adoption but never select between arms.
