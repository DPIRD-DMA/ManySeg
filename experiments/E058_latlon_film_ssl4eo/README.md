# E058: Where should metadata enter an image model? E057 at scale (SSL4EO-S12, ~48k locations)

Status: **done, positive** (2026-11-09, RTX 3090, torch 2.14.1+cu130, seeds 0–2). Repeats
E057 with 85× more training locations. Every injection of latitude beats the baseline by more than 2 seed SDs (the shuffled control
missed its own bar by 0.0001), and the arms rank by
how much of the network latitude reaches. The decision rule is E057's, fixed before any E058
run; the screen to run seeds 1–2 was "seed 0 ranking as in E057".

## Why a redo
E057 had ~400 training tiles and ~104 test locations (about 39 chips per tile). Models memorised
the training places (train loss 0.15, test NLL 5–10), the latitude-only prior was below chance,
and seed SD understated the uncertainty from which tiles landed in test.

## Protocol
* Data: SSL4EO-S12 v1.1 (`embed2scale/SSL4EO-S12-v1.1` on Hugging Face), `S2RGB` modality,
  the first 100 of 477 training shards (51,200 locations). Each location has 4 seasonal
  Sentinel-2 RGB images (B04 B03 B02, 8-bit, 264 px at 10 m), a cloud mask and the centre
  lat/lon. `prep.py` takes the 256 px centre, 2×2-averages it to 128 px, and splits 70/15/15
  by 2° × 2° block. Sites within 5.5 km of a neighbouring block in a different split are dropped
  (2,635, ~5%), so no two sites closer than that straddle splits.
  Train 34,425 locations / 1,422 blocks; val 7,100 / 284; test 7,040 / 281. Latitude 53° S–66° N.
* Training draws one season per location per step, among those with cloud fraction < 0.3 (else
  the least cloudy). Evaluation uses each location's least cloudy season.
* Model, target, metrics, augmentation and arms as E057 (`train.py` is E057's with RGB input
  and the seasonal sampling): ResNet-34, image-scope head, 36 longitude bins of 10°, 30 epochs,
  batch 128, AdamW 3e-4 one-cycle, bf16, flips and 90° rotations. ~11–13 min per run, two at a
  time.
* Code reviewed by Codex before launch. Fix taken: block-hash splitting alone let sites just
  either side of a block edge land in different splits; the 5.5 km buffer was added.
  A numpy-bool bug in that buffer crashed the first prep run; fixed and re-run.

## Decision rule (from E057)
An arm helps if its test accuracy − `none` > 2 pooled SD over seeds 0–2, val agrees in sign, and
`film_shuffled` stays within 2 pooled SD of `none`. Placement comparisons use the same rule.

## Results (mean ± seed SD, seeds 0–2)

| arm | latitude reaches | test acc | per seed | val acc | test top-3 | median lon err | test NLL |
|---|---|---|---|---|---|---|---|
| film_shuffled | (random latitude) | 0.383 ± 0.004 | 0.387 / 0.384 / 0.379 | 0.438 ± 0.006 | 0.674 | 7.18° | 2.245 |
| none (baseline) | — | 0.393 ± 0.005 | 0.390 / 0.390 / 0.399 | 0.440 ± 0.005 | 0.685 | 6.90° | 2.049 |
| head | logits (additive) | 0.433 ± 0.004 | 0.434 / 0.428 / 0.436 | 0.480 ± 0.002 | 0.741 | 5.80° | 1.857 |
| film | pooled feature | 0.462 ± 0.008 | 0.458 / 0.456 / 0.471 | 0.502 ± 0.004 | 0.770 | 5.42° | 1.864 |
| channels | input, so the whole encoder | 0.488 ± 0.006 | 0.490 / 0.481 / 0.493 | 0.522 ± 0.004 | 0.791 | 5.14° | 1.701 |
| **film_encoder** | after every stage | **0.495 ± 0.005** | 0.501 / 0.495 / 0.490 | **0.539 ± 0.006** | **0.802** | **5.05°** | 1.840 |

Latitude-only prior (most common training longitude bin per latitude bin): 0.117 test accuracy.
Seed-0 probe (latitude bin from pixels, 18 × 10°): 0.60 accuracy, median error 3.7°.

| comparison | test diff | 2 pooled SD | val diff | 2 pooled SD | verdict |
|---|---|---|---|---|---|
| head − none | +0.040 | 0.009 | +0.040 | 0.007 | clears its bar |
| film − none | +0.069 | 0.014 | +0.061 | 0.009 | clears its bar |
| channels − none | +0.096 | 0.011 | +0.082 | 0.009 | clears its bar |
| film_encoder − none | +0.103 | 0.010 | +0.099 | 0.011 | clears its bar |
| film_shuffled − none | −0.009 | 0.009 | −0.002 | 0.011 | **control fails narrowly** (below `none`) |
| film − head | +0.029 | 0.013 | +0.021 | 0.006 | film better |
| channels − film | +0.026 | 0.015 | +0.021 | 0.008 | channels better |
| film_encoder − film | +0.033 | 0.014 | +0.038 | 0.011 | encoder placement better |
| film_encoder − channels | +0.007 | 0.012 | +0.017 | 0.011 | not shown on test |

* **Decision: the accuracy comparisons pass, but the full rule narrowly fails** on its control
  condition. Every placement beats `none` by more than 2 pooled SD on test, with val agreeing.
  `film_shuffled` is 0.0093 *below* `none` against a 0.0092 bar, so it is outside 2 pooled SD by
  0.0001. It sits below the baseline, not above, so it shows no gain without the true latitude;
  why a random latitude costs a little accuracy wasn't tested.
* **The arms rank by reach, with each step clearing the bar on test and val:**
  head (logits) < film (pooled feature) < channels (input) ≈ film_encoder (every stage).
  The same FiLM moved from the pooled feature into the encoder stages gains +3.3 points.
* **What changed from E057:** with enough data, `head` helps too (+4.0; it was null in E057), the
  FiLM arms no longer have worse NLL (that was overfitting), and the ranking is otherwise
  the same. Seed-to-seed spread is under a point.
* **The image already carries latitude** (probe median error 3.7°), and the true value still adds
  up to 10 points, so the vector is far from redundant here.

## Caveats
* Interpretation, not isolation: "reach" fits the ranking, but the placements also differ in
  parameter count and in how they adapt the pretrained encoder.
* The shuffled control is pooled FiLM only; the other placements have no control of their own.
* RGB only (no NIR); 100 of 477 shards; SSL4EO samples around cities, so the locations are not
  uniform over land.
* `film_encoder` is written in this experiment with forward hooks; it isn't a ManySeg option.
