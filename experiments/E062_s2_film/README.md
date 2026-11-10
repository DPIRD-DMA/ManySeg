# E062: Does Sentinel-2 help as FiLM metadata instead of an image? (FLAIR #2, after E059/E061)

Status: **done, screen not passed** (2026-11-10, RTX 3090, seed 0 only). S2 summary
statistics as FiLM show no worthwhile gain over aerial alone at seed 0, so the shuffled control
and seeds 1–2 were not run. **Pre-registered:** the protocol and decision rule below were
written and committed before any E062 run.

## Question
In E059 a separate S2 encoder added little over aerial alone (+0.006 test mIoU), and in E061 a
wider S2 window as an image did not help. Each aerial patch covers only 10 × 10 S2 pixels, so
S2's spatial detail is small; its value is more likely spectral and seasonal. This tests
passing S2 as a vector of summary statistics through ManySeg's FiLM conditioning (a
patch-level hint, the same at every pixel) instead of as a second image encoder.

## Protocol
* Data: E061's patches (identical to E059's: same labels, domains and order, and the centre of
  E061's 40 × 40 median equals E059's 10 × 10 median; checked). Train 7,417, val 1,932, test
  1,463 patches from 8 / 2 / 2 départements.
* Model and recipe as E059: ResNet-34 U-Net on aerial (5 bands, 256 px at 0.4 m),
  cross-entropy, AdamW 3e-4 one-cycle, 30 epochs, batch 16, bf16, shared flips and 90°
  rotations. mIoU over the 12 scored classes from the full confusion matrix.
* Arms:
  * `aerial`: aerial only (baseline).
  * `s2enc`: + separate ResNet-18 S2 encoder on the 10 × 10 annual median, `same_extent`,
    concat fusion (E059's `mid_median`; reproduction check).
  * `film_med`: + FiLM from per-band mean and std of the annual median over the patch
    footprint (20 values).
  * `film_ts`: + FiLM from monthly per-band means over the footprint (12 × 10 = 120 values;
    months without a clear date, 17% of values, are passed as NaN, which ManySeg replaces with
    a learned "unknown" vector).
  * `film_ctx`: + FiLM from the `film_med` statistics plus the same statistics over the 40 × 40
    px (400 m) window (40 values).
  * `<arm>_shuf`: the picked FiLM arm with each patch's vector taken from another patch of the
    same split (fixed permutation). Control: a gain that survives shuffling doesn't come from
    the true S2 values.
* FiLM is ManySeg's default for pixel heads (`conditioning: "film"`: the encoder levels entering
  the decoder and the decoder output), MLP 64 wide. Vector features are z-scored with training
  statistics. FiLM arms add ~0.14 M parameters; `s2enc` adds 11.9 M.

* Code reviewed by Codex before launch. Fixes taken: `run.sh` could leave a stale
  `result.json` that the pipeline would accept after a failed run; patch alignment between the
  E059 and E061 files is now asserted at load.

## Decision rule
* Screen (seed 0), on **val** only: pick the FiLM arm (`film_med`, `film_ts`, `film_ctx`) with
  the highest val mIoU. Continue if it beats `aerial` by more than 0.01 val mIoU. Otherwise stop
  after seed 0: S2 as FiLM is reported as not promising, untested beyond one seed.
* If the screen passes, run the picked arm's `_shuf` control at seed 0, then `aerial`, `s2enc`,
  the picked arm and its `_shuf` at seeds 1–2.
* S2-as-FiLM helps if the picked arm's test mIoU − `aerial` > 2 pooled SD over seeds 0–2 (pooled
  SD = sqrt of the mean of the two seed variances), val agrees in sign, **and** the picked arm
  beats its `_shuf` control by more than 2 pooled SD on test. Two pooled SD over 3 seeds is a
  screening heuristic, not a calibrated test.
* Every run also computes test mIoU (it is in the logs), but the screen reads val only.
* Secondary, same rule: the picked arm against `s2enc` (FiLM vs a second encoder).

## Results (seed 0)

| arm | S2 as | test mIoU | val mIoU | params | final train loss |
|---|---|---|---|---|---|
| aerial (baseline) | — | 0.459 | 0.486 | 24.44 M | 0.471 |
| s2enc | image, second encoder | 0.451 | **0.494** | 36.35 M | 0.463 |
| film_med | FiLM, 20 footprint statistics | 0.457 | 0.490 | 24.59 M | 0.441 |
| film_ts | FiLM, 120 monthly means | 0.437 | 0.467 | 24.59 M | 0.407 |
| film_ctx | FiLM, 40 footprint + 400 m statistics | **0.471** | 0.477 | 24.59 M | 0.430 |

Test IoU per class:

| arm | bld | perv | imp | bare | water | conif | decid | brush | vine | herb | agri | plow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerial | 0.79 | 0.43 | 0.63 | 0.16 | 0.57 | 0.21 | 0.63 | 0.26 | 0.43 | 0.45 | 0.46 | 0.50 |
| s2enc | 0.79 | 0.42 | 0.66 | 0.17 | 0.71 | 0.27 | 0.64 | 0.28 | 0.21 | 0.44 | 0.42 | 0.41 |
| film_med | 0.78 | 0.46 | 0.65 | 0.18 | 0.69 | 0.27 | 0.63 | 0.26 | 0.16 | 0.45 | 0.42 | 0.52 |
| film_ts | 0.79 | 0.44 | 0.64 | 0.16 | 0.73 | 0.26 | 0.64 | 0.32 | 0.01 | 0.45 | 0.35 | 0.47 |
| film_ctx | 0.79 | 0.48 | 0.65 | 0.20 | 0.62 | 0.30 | 0.64 | 0.28 | 0.35 | 0.44 | 0.42 | 0.50 |

* **Screen: not passed.** The FiLM arm picked on val is `film_med`, +0.004 val mIoU over
  `aerial` against a required +0.010. Per the rule, E062 stopped after seed 0: S2 as FiLM is
  reported as not promising, untested beyond one seed. The shuffled control was not run.
* **Val and test disagree on `film_ctx`:** below `aerial` on val (−0.009), highest of all
  arms on test (+0.012). It was not picked by the pre-registered val screen. With two
  départements per split, one class can move the mean by several points (vineyard does here),
  so one seed can't separate this from noise.
* **The monthly series hurts** (`film_ts`, −0.018 val, −0.022 test) and has the lowest
  training loss of all arms (0.407 against 0.471). Its vineyard IoU collapses to 0.01.
* **The same direction in E059, E061 and E062:** in every S2 arm of the three experiments,
  water and coniferous IoU are above aerial alone and vineyard IoU is below it. Here (E062,
  seed 0): water 0.57 → 0.62–0.73, coniferous 0.21 → 0.26–0.30, vineyard 0.43 → 0.01–0.35.
  In E062 the S2 arms also reach a lower final training loss than `aerial` (0.407–0.463
  against 0.471). A plausible reading, not tested: S2 values for vineyard (or in general)
  differ between départements, so the model learns département-specific S2 cues that don't
  transfer to held-out ones.
* **`s2enc` does not reproduce E059's `mid_median` exactly** (0.451 / 0.494 test / val against
  0.465 / 0.498) although its data are asserted identical, while `aerial` reproduces E059 and
  E061 exactly. A likely cause, not confirmed: CUDA's bilinear-resize backward is
  nondeterministic, and only the S2-encoder arms resize features bilinearly. The difference is
  inside E059's seed SD for that arm (0.014 test).
* **Decision:** no recommendation to pass coarse imagery as FiLM statistics from this setup.

## Caveats
* One seed, two test and two val départements; the vineyard swing alone moves mean mIoU by up
  to 3.5 points (`film_ts` against `aerial`).
* Only simple statistics (band mean and std, monthly means) were tried, through ManySeg's
  default FiLM MLP (64 wide). Indices such as NDVI, other summaries, or FiLM dropout as a
  regulariser were not tried.
