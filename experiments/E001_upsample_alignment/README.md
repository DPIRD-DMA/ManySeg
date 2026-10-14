# E001: Upsampling alignment in U-Net decoders

Status: **complete, negative result** (E001b, 2026-10-14). The geometric offset is real at
initialisation, but removing it does not change trained accuracy.

## Hypothesis

Timm CNN encoders use odd-kernel, "same"-padded strided convolutions, which centre feature
`j` at stride `s` on input pixel `s·j`. Nearest upsampling, and bilinear upsampling with
`align_corners=False`, both assume the feature is centred at `s·j + (s−1)/2`. As a result each
×2 decoder stage shifts features by about half a pixel up and to the left, and the shift
compounds across levels. If this is right, pixel predictions are systematically offset from
their inputs at initialisation, and every SMP-style U-Net carries the same bias.

## Method

For an untrained model in eval mode with a random 256×256 input, take the gradient of
output pixel `(i, j)` (summed over classes) with respect to the input. Compute the centroid of
`|grad|` and subtract `(i, j)`. Average this over 60 interior points
(`measure_offset.py`). Two seeds, resnet18 encoder, U-Net with SMP-default widths. CPU.
The measurement does not depend on hardware.

## Results (mean offset in px, (dy, dx); std over positions in brackets)

| Model | seed 0 | seed 1 |
| --- | --- | --- |
| ManySeg U-Net, nearest (SMP-equivalent default) | (−5.96, −5.83) [0.4] | (−5.78, −5.27) [0.5] |
| ManySeg U-Net, bilinear `align_corners=False` | (−5.91, −5.45) [0.4] | n/a |
| ManySeg U-Net, `upsample="aligned"` | (−0.69, −0.25) [0.4] | (−0.89, +0.20) [0.4] |
| SMP 0.5.0 `Unet("resnet18")` | (−11.9, −9.4) [5.7] | (−10.5, −8.4) [5.7] |

SMP's larger offset, and its strong dependence on position, are not explained yet.
Candidates are torchvision's resnet stem as opposed to timm's, and the different init.

## Interpretation

The bias is real, structural, and shared with SMP. Grid-consistent bilinear upsampling
(`aligned`: replicate-pad by one, then `align_corners=True` to `2n+1`, then crop to `2n`)
removes most of it. Trained networks can learn to compensate, because convs after
upsampling see unshifted skips. So the effect on trained accuracy, especially boundary
accuracy and thin structures, is unknown.

## Decision

* Keep `upsample="nearest"` as the default so the reference U-Net stays SMP-equivalent.
* Ship `upsample="aligned"` as an option, with tests:
  `test_spatial_alignment_aligned_upsampling_is_centred` and
  `test_spatial_alignment_nearest_bias_is_bounded`.
* Follow-up (Phase 5): train nearest, bilinear and aligned on Pet and one remote-sensing dataset
  with 3 seeds, and compare mIoU and boundary F1. Change the default only if `aligned` is
  better on both and no slower.
* Open question: ViT patch embeddings (k=16, s=16, no pad) **do** centre at `s·j + (s−1)/2`.
  The correct upsampling therefore depends on the backbone. A `FeatureSpec.offset` field,
  found by probing, may be the general solution.

## E001b: trained effect (Pet, fastai_like recipe, 20 epochs, 3 seeds; nearest = E003 unet runs)

| upsample | val mIoU | val boundary F1 | test mIoU | test boundary F1 | t vs nearest (val / test / test BF1) |
|---|---|---|---|---|---|
| nearest (default) | 0.8224 ± 0.0010 | 0.7877 ± 0.0032 | 0.8220 ± 0.0001 | 0.7858 ± 0.0019 | |
| aligned | 0.8213 ± 0.0006 | 0.7859 ± 0.0014 | 0.8220 ± 0.0005 | 0.7865 ± 0.0012 | −1.8 / +0.1 / +0.5 |
| bilinear | 0.8208 ± 0.0009 | 0.7863 ± 0.0024 | 0.8224 ± 0.0001 | 0.7860 ± 0.0009 | −2.2 / +3.7 / +0.1 |

**Interpretation.** Removing the ~6 px init-time offset (`aligned`) leaves mIoU and boundary F1
unchanged after training. The convolutions after each upsampling step learn to compensate
using the unshifted skips. The +0.0004 test edge for `bilinear` contradicts its val
direction and fails the pre-registered rule.

**Decision.** `nearest` stays the default (SMP-equivalent and the cheapest). `aligned` stays
available and tested, because it is the geometrically correct choice for analyses that use
untrained or frozen decoders, e.g. feature probing. Not yet tested: remote-sensing tasks with
thin structures (planned with AI4Boundaries boundary heads, E010).
