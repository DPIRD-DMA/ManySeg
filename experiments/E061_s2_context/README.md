# E061: Does a wider Sentinel-2 context help a separate S2 encoder? (FLAIR #2, E059 follow-up)

Status: **pre-registered, not run.** This README, the arms and the decision rule were written
before any E061 run.

## Question
In E059 a separate S2 encoder beat stacking S2 onto the aerial grid, but added little over aerial
alone (+0.006 test mIoU, inside the noise). Its S2 input was only the 10 × 10 px (100 m) under
each aerial patch, because ManySeg's `same_extent` alignment requires both inputs to cover the
same ground. The published FLAIR #2 gains used a 40 × 40 px S2 window around each patch. If a
wider window helps clearly, ManySeg needs a "context" alignment, where a coarse input covers more
ground than the primary one and only its centre is fused.

## Protocol
* Data: E059's domains, split and patch sampling (same rng). The S2 annual median is cut as a
  40 × 40 px (400 m) window centred on the patch centroid; the 20 and 10 px contexts are its
  centre. Dates are chosen as in E059 (clear when max cloud and snow probability over the
  central 10 × 10 is below 20%; ≥ 3 needed). Each window pixel takes the median over the chosen
  dates on which it is itself clear, or over all of them if it never is, so the centre 10 × 10
  equals E059's median exactly. Requiring the whole window to be clear dropped ~70% of patches,
  so it was not used. Patches whose window leaves the zone are dropped. Every arm uses the same
  patches and medians, so `ctx10` is E059's `mid_median` on a slightly smaller patch set, with S2
  normalisation statistics taken over the 40 × 40 window.
* Model and recipe as E059: ResNet-34 U-Net on aerial (5 bands, 256 px at 0.4 m), ResNet-18 S2
  encoder, intermediate concat fusion, `same_extent`, cross-entropy, AdamW 3e-4 one-cycle,
  30 epochs, batch 16, bf16, shared flips and 90° rotations.
* S2 crops are upsampled by 6.4 (10 → 64 px, 20 → 128, 40 → 256), so ground per input pixel is
  equal across arms. For `ctx20` and `ctx40`, a forward hook on the S2 encoder keeps the centre
  1/k of each feature map (k = 2 or 4: the patch footprint, approximately, since 10 S2 px are
  100 m and the patch is 102.4 m, as in E059), so ManySeg fuses only the part over the patch.
  ManySeg is told the S2 input is 64 px, so the cropped features go through the same single
  bilinear resize onto the aerial grid as `ctx10` (checked: max difference 0). The deeper
  features in that centre have seen the surrounding context through their receptive field. This is a prototype of a context
  alignment, written in the experiment, not a ManySeg option.
* Arms: `aerial` (baseline), `ctx10`, `ctx20`, `ctx40`.
* Code reviewed by Codex before launch. Fixes taken: `ctx20` first went through two bilinear
  resizes against one for the other arms; the screen first picked the context arm on test.
* Metric: mIoU over the 12 scored classes from the full confusion matrix, as E059.

## Decision rule
* Screen (seed 0), on **val** only: pick the context arm, `ctx20` or `ctx40`, with the higher
  val mIoU. Run seeds 1–2 if it beats `ctx10` by more than 0.01 val mIoU (one point). Otherwise
  stop after seed 0: context is reported as not promising, untested beyond one seed.
* The picked arm is then fixed. Context helps if its test mIoU − `ctx10` > 2 pooled SD over
  seeds 0–2 (pooled SD = sqrt of the mean of the two seed variances) and val agrees in sign.
  Two pooled SD over 3 seeds is a screening heuristic, not a calibrated test. A pass justifies
  designing a context alignment in ManySeg (design only, for review).
* Secondary, same rule: the picked arm − `aerial`, i.e. whether S2 is worth adding at all.
