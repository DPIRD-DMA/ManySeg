# E025: training-step speed of the default U-Net against SMP

Status: **done** (registered and run 2026-10-21; "before" at code `5c079d8` plus uncommitted
work, "final" with the fix below). Result: the gap was ManySeg overhead, now fixed. Compiled
U-Net steps are at parity with SMP; eager steps are 18–25% faster, with no more memory.

A downstream project (OmniCloudMask
cloud / shadow training, fastai, bf16, 509² inputs, batch 40) measured whole-run wall times
5–8% longer with ManySeg's default U-Net than with SMP's `Unet` on the same timm encoder
(`edgenext_small`: 3.56 h vs 3.40 h; `convnextv2_femto`: 3.81 h vs 3.54 h, 35 epochs, RTX 4090).

## Question
Is a ManySeg training step slower than SMP's for the same architecture, and if so, where does
the time go? Is the cost ManySeg overhead that SMP does not pay (to be fixed), or inherent?

## Protocol
* `bench.py`: one configuration per process, `create_model(backbone=..., in_channels=3,
  num_classes=4, decoder="unet")` against `smp.Unet(encoder_name="tu-...", in_channels=3,
  classes=4)`, random init (weights do not change the cost).
* Step = forward + cross-entropy + backward + `zero_grad` under bf16 autocast (fp32 weights),
  batch 40, 3 × {509², 512²}, 4 classes. Timing: CUDA events per step, median of 40 steps after
  10 warm-up steps (20 compiled). Peak memory: `max_memory_allocated` over the timed steps.
* Grid: {manyseg, smp} × {`edgenext_small`, `convnextv2_femto`} × {509, 512} ×
  {eager, `torch.compile` (default mode)} = 16 runs, all in one session.
* Where it goes: `torch.profiler` over 5 eager steps at 509² (`profiles/`).
* Environment: torch 2.14.1+cu130, timm 1.0.30, segmentation-models-pytorch 0.5.0, RTX 4090.
  The GPU is shared: utilisation and memory of other work are recorded after each run.

## Decision rule
* A config pair differs if the medians differ by more than 2%. The step-to-step p10–p90
  spread at a fixed configuration is about ±2% on this machine.
* ManySeg overhead that SMP does not pay is fixed if the fix keeps outputs identical; the
  grid is rerun after the fix.
* A remaining gap that comes from a design commitment (e.g. padding inputs to the stride) is
  documented in the README next to the speed presets, with these numbers.

## Results

Median step time (ms; forward + backward, bf16 autocast, batch 40), ManySeg against SMP in the
same grid. SMP was rerun in each grid as the control: its medians moved by up to 6% between
windows (shared GPU; other work's utilisation before each run, 0–100%, is in `runs/`). Grids:
`before` (code as reported downstream), `after` (first fix; superseded) and `final`.

| backbone | size | mode | SMP before / final | ManySeg before | gap | ManySeg final | gap | peak GiB: MS before / MS final / SMP |
|---|---|---|---|---|---|---|---|---|
| edgenext_small | 509 | eager | 230 / 217 | 236 | +2.3% | 166 | −23.5% | 11.04 / 10.81 / 11.07 |
| edgenext_small | 509 | compile | 138 / 134 | 254 | +83% | 132 | −1.6% | 10.02 / 10.02 / 9.87 |
| edgenext_small | 512 | eager | 226 / 219 | 233 | +3.5% | 165 | −25.0% | 11.04 / 10.81 / 11.36 |
| edgenext_small | 512 | compile | 136 / 136 | 256 | +89% | 134 | −1.2% | 10.10 / 10.02 / 10.10 |
| convnextv2_femto | 509 | eager | 287 / 283 | 301 | +5.1% | 232 | −18.2% | 12.10 / 11.87 / 12.11 |
| convnextv2_femto | 509 | compile | 124 / 125 | 246 | +98% | 125 | −0.7% | 7.78 / 7.78 / 7.67 |
| convnextv2_femto | 512 | eager | 303 / 288 | 311 | +2.7% | 230 | −20.0% | 12.10 / 11.87 / 12.46 |
| convnextv2_femto | 512 | compile | 133 / 125 | 264 | +98% | 125 | −0.5% | 7.85 / 7.78 / 7.85 |

Parameters are identical (edgenext_small 7.668M, convnextv2_femto 7.493M).

The other decoders that used the gather, ManySeg before / final (`edgenext_small`, 509²,
batch 16; SMP's UNet++ and FPN refuse 509² inputs, and its UNet++ cannot build on this
4-level encoder, so there is no SMP reference):

| decoder | mode | before | final | change | peak GiB before / final |
|---|---|---|---|---|---|
| unetplusplus | eager | 84.4 | 82.2 | −2.6% | 5.25 / 4.98 |
| unetplusplus | compile | 70.7 | 69.4 | −1.9% | 4.59 / 4.59 |
| fpn | eager | 63.0 | 58.9 | −6.5% | 3.86 / 3.85 |
| fpn | compile | 44.9 | 38.5 | −14.2% | 3.07 / 3.07 |
| fastai_unet | eager | 194.6 | 193.3 | −0.7% | 9.79 / 9.79 |
| fastai_unet | compile | 159.8 | 150.7 | −5.7% | 8.36 / 8.36 |

(UNet++'s "before" already had the eager U-Net fix, which it shares.)

**Where the time went (profiler, 5 eager steps, edgenext_small 509², `profiles/`).** The
decoders' nearest-neighbour upsampling was an integer-index gather (`index_select`), the
workaround for an Inductor miscompile of `F.interpolate(mode="nearest", size=...)` with
`dynamic=True` (`docs/upstream/torch-inductor-dynamic-nearest.md`). Its backward is
`index_add_`, an atomic scatter-add: 74 ms per 5 steps against 22 ms for SMP's whole
`upsample_nearest2d`. Compiled, Inductor turns it into a bf16 atomic scatter, which made compiled
ManySeg slower than eager. The gather also returned contiguous (NCHW) tensors from channels-last
timm features, so cuDNN converted layouts around the decoder convs.

**Fix.** `upsample2x_nearest`: `F.interpolate(scale_factor=2, mode="nearest")` cropped to the
skip's size; `fit_nearest`: a plain crop where the map is at most one row larger (the fastai
U-Net's pixel-shuffled maps). Both give the gather's exact values for the sizes they are used
at: an output of 2n or 2n − 1 (a ceil-rounded or stride-padded pyramid), and `scale_factor=2`
compiles correctly with `dynamic=True`. In eager mode they are used whenever the sizes qualify.
Under `torch.compile` / `torch.export` they are used when the model pads inputs to a multiple
of the deepest stride (the default `pad_to_stride=True`), as a static flag, so no size check is
traced; otherwise the gather stays. Two details mattered:

* Autocast runs `upsample_nearest2d` in float32, so the first version carried float32 maps
  through bf16 decoders (+0.4–0.7 GiB peak). Nearest upsampling only copies values, so the
  result is cast back to the input dtype, losslessly.
* The fastai U-Net on encoders without a stride-2 level (ConvNeXt) ends at stride 2, where the
  final resize is a 2x upsample, not a crop; a float64 equivalence test against the gather
  caught a first version that cropped there.
* Whether a step is exactly 2x is a property of each pair of levels, not of the padding: with
  sparse `out_indices` (strides 4, 16, 32) FPN, UNet++ and the fastai U-Net have a 4x step. A
  Codex review caught a version that set one flag per decoder and so padded those steps with
  zeros (eager and compiled); flags are now per step, and the equivalence tests cover sparse
  levels and the flags-only (compiled) path. The timing grids used dense pyramids, unaffected.

Forward values are bit-identical to the gather (float64 tests for all four decoders, padded and
not, with and without a stride-2 level); gradients agree up to summation order, and in bf16 they
are closer to an fp32 reference than the gather's (max error 0.24–0.35% vs 0.31–0.71% of the
largest gradient). With channels-last features the decoder now stays channels-last, so later
convs can pick other cuDNN kernels: ConvNeXtV2 outputs then differ by about 1e-3 with TF32 on
and 6e-6 with TF32 off, the usual kernel-choice noise.

**Why eager is now faster than SMP.** SMP's encoder list for `tu-` encoders starts with the
input image and a stride-2 feature in NCHW layout; concatenating them puts SMP's two finest
decoder blocks (the most pixels) in NCHW, where bf16 BatchNorm is slow (159 ms vs 36 ms of BN
backward per 5 steps). ManySeg's decoder stays channels-last throughout. Compiled, both libraries
reach the same speed.

**Padding (509 → 512)** costs nothing measurable here: ManySeg's 509² and 512² steps are within
noise of each other, and SMP (which does not pad at 509) is no faster. It is not documented as
a cost.

These are single-session timings on a shared GPU, so treat ±3% as noise; the U-Net conclusions
rest on gaps of 18–98%, the other decoders' compiled gains of 6–14% (UNet++'s ±2% is noise).

**Downstream check (whole runs).** The OmniCloudMask project reran its full training with the
fix (fastai, bf16, 509², batch 40, 35 epochs, RTX 4090; one run each, data loading included).
ManySeg went from 3.56 h to 3.23 h on `edgenext_small` and from 3.81 h to 3.30 h on
`convnextv2_femto`. SMP's `Unet` took 3.40 h and 3.54 h, so ManySeg is now 5–7% faster end to
end, with accuracy unchanged within run-to-run noise. Whole-run times also include data
loading and evaluation, so the gap is smaller than the eager step gap.
