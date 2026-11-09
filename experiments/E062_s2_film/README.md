# E062: Does Sentinel-2 help as FiLM metadata instead of an image? (FLAIR #2, after E059/E061)

Status: **pre-registered, not run.** This README, the arms and the decision rule were written
before any E062 run.

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
