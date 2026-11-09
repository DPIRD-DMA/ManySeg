# E057: Where should metadata enter an image model? (longitude from Sentinel-2 chips, given latitude)

Status: **done, positive** (2026-11-09, RTX 3090, torch 2.14.1+cu130). Latitude helps through
FiLM, channels and FiLM inside the encoder, not through `head`; where it enters matters more
than how. Seeds 0–2. **Not
pre-registered:** the decision rule below was stated after seed 0 and most of seed 1 (none,
film, head, channels) had been seen, before seed 2 ran. `film_encoder` was added after seed 0,
to test an explanation of the seed-0 ranking.

> **Superseded by [E058](../E058_latlon_film_ssl4eo/README.md)**, the same experiment on
> ~48k SSL4EO-S12 locations instead of ~400 tiles. E058 confirms the placement ranking; there
> `head` helps too, and the FiLM arms' worse NLL here looks like an overfitting artefact.

## Question
E055 found that FiLM helps when the vector changes the correct output for fixed pixels. Here
latitude does that for longitude: knowing a dry scrubland chip is at 31° S narrows where on Earth
it can be. Any way of giving the model latitude should help, so the question is **which
injection works**: ManySeg's FiLM, `head` or `channels`.

## Protocol
* Data: the existing global L1C chip set on the 3090 (`~/data_s2_l1c_chips`: 60k chips, 4
  bands, 256 px, 10–60 m GSD, 909 MGRS tiles, 56° S–72° N, all months). Chips with cloud
  fraction ≤ 0.2 kept (23k); each is located at the centre of its MGRS 100 km square (good to
  ~70 km, far below the bins); split 70/15/15 **by tile** (train 15,726 chips / 404 tiles, val
  3,416 / 92, test 4,134 / 104); 2×2-averaged to 128 px. `prep.py`.
* Target: longitude in 36 bins of 10° (cross-entropy). Metrics: top-1 and top-3 accuracy,
  median circular longitude error of the predicted bin centre, NLL.
* Model: ResNet-34 ImageNet weights, image-scope classification head (no decoder). AdamW 3e-4,
  one-cycle, 30 epochs, batch 128, bf16. Random flips and 90° rotations (these hide the shadow
  direction, which would otherwise give away the hemisphere). About 7 min per run.
* Arms (latitude as a `continuous` field, mean 0, SD 35°):
  * `none`: image only (**baseline**).
  * `head`: ManySeg `head` mode, an additive logit term `Linear([pool(f), e])`.
  * `film`: ManySeg FiLM. For image heads this modulates only the pooled deepest feature.
  * `channels`: latitude as an extra input band, so it reaches the whole encoder.
  * `film_shuffled`: FiLM fed a latitude drawn at random from the training latitudes (control).
  * `film_encoder`: FiLM after each of the four ResNet stages, written in this script with
    forward hooks (same metadata MLP shape as ManySeg's, zero-initialised projection). Not a
    ManySeg option.
  * `probe` (seed 0): predict the latitude bin (18 × 10°) from pixels alone.
* Code reviewed by Codex before launch. Fixes taken: CUDA can't index `uint16` tensors (data
  now int32 on the GPU); longitude bins clamped at the dateline instead of wrapping;
  `film_encoder`'s MLP reshaped to match ManySeg's.

## Caveats
* **Few independent locations.** About 39 chips per tile, so test is really ~104 places. Seed SD
  understates the uncertainty from which tiles landed in test.
* **Heavy overfitting.** Train loss reaches ~0.15 while test NLL is 5.4–9.8; val accuracy peaks
  around epoch 20. Models largely memorise training locations, so absolute accuracies are low
  (12–19% against 2.8% chance); the comparison between arms is what matters.
* **The latitude-only prior** (most common training longitude bin per latitude bin) scores
  0.5% test accuracy, below chance: the clustered training scenes put each latitude band's mode
  where test has few chips. The training table gives the true bin 5.1% probability on average
  (uniform: 2.8%), so latitude alone carries little.

## Decision rule
* An arm helps if its test accuracy − `none` > 2 pooled SD over seeds 0–2, val agrees in sign,
  and `film_shuffled` stays within 2 pooled SD of `none`.
* Placement: `film_encoder` − `film` under the same 2-SD rule. If it passes, the explanation is
  "reach, not mechanism", and FiLM inside the encoder becomes a candidate library option.

## Results (mean ± seed SD, seeds 0–2)

| arm | latitude reaches | test acc | per seed | val acc | test top-3 | median lon err | test NLL |
|---|---|---|---|---|---|---|---|
| none (baseline) | — | 0.120 ± 0.004 | 0.120 / 0.124 / 0.117 | 0.135 ± 0.011 | 0.303 | 37.3° | 5.74 |
| head | logits (additive) | 0.119 ± 0.013 | 0.125 / 0.128 / 0.104 | 0.151 ± 0.008 | 0.306 | 36.0° | 6.03 |
| film_shuffled | (random latitude) | 0.116 ± 0.013 | 0.130 / 0.106 / 0.112 | 0.144 ± 0.003 | 0.305 | 35.8° | 9.48 |
| film | pooled feature | 0.146 ± 0.010 | 0.147 / 0.155 / 0.136 | 0.168 ± 0.016 | 0.329 | 32.0° | 8.90 |
| channels | input, so the whole encoder | 0.166 ± 0.007 | 0.170 / 0.170 / 0.157 | 0.190 ± 0.007 | 0.364 | 25.1° | 5.60 |
| **film_encoder** | after every stage | **0.180 ± 0.004** | 0.185 / 0.178 / 0.179 | **0.202 ± 0.015** | **0.367** | **24.4°** | 8.04 |

Seed-0 probe (latitude from pixels alone): 36% accuracy over 18 bins (chance ≈ 6%), median
error 8°. The image carries latitude only coarsely, so the true value has room to help.

| comparison | test diff | 2 pooled SD | val diff | 2 pooled SD | verdict |
|---|---|---|---|---|---|
| head − none | −0.001 | 0.019 | +0.016 | 0.019 | no gain |
| film − none | +0.026 | 0.015 | +0.034 | 0.027 | **helps** |
| channels − none | +0.046 | 0.011 | +0.056 | 0.018 | **helps** |
| film_encoder − none | +0.060 | 0.007 | +0.068 | 0.026 | **helps** |
| film_shuffled − none | −0.004 | 0.019 | +0.009 | 0.016 | control OK |
| film_encoder − film | +0.034 | 0.015 | +0.034 | 0.031 | **placement matters** |
| film_encoder − channels | +0.015 | 0.012 | +0.012 | 0.024 | passes the rule (val agrees in sign, below its own bar) |

* **Decision: FiLM, channels and FiLM-in-the-encoder all help; `head` does not.** The shuffled
  control (pooled FiLM fed random latitudes) stays at the baseline, so FiLM's gain needs the
  true latitude; the other placements weren't given their own shuffled control.
* **Reach, not mechanism (an interpretation).** The arms rank by how much of the network latitude reaches:
  head ≈ none < film (pooled feature) < channels ≈ film_encoder (whole encoder). The same FiLM
  moved from the pooled feature into the encoder stages more than doubles its gain over the
  baseline (+0.026 → +0.060). This fits "reach matters", but doesn't isolate it as the cause:
  the placements also differ in parameter count and in how the encoder adapts.
* **Agrees with E055.** There ManySeg's FiLM reached every encoder level entering the decoder
  plus the output, while `channels` only touched the input, and FiLM won. In both experiments the
  injection that reaches more of the network wins.
* FiLM arms have higher test NLL than `channels` (8–9 vs 5.6), so worse log loss despite
  similar or better accuracy. Over-confidence on misses is a likely cause, but it wasn't
  measured.
* **Library implication:** ManySeg's FiLM for image heads (pooled feature only) is the weakest
  helpful placement here. A FiLM option inside the encoder stages (as `film_encoder` does with
  hooks) is the candidate change; untested on dense tasks.
