# E003: Decoder comparison on Oxford Pet (pre-registered)

Status: **planned** (registered 2026-10-14 before any E003 runs).

## Questions
1. Does decoder capacity explain fastai's reputation? E002 stage A already shows no
   difference at equal recipe. This experiment checks the decoders inside one library with an
   identical encoder, data and recipe.
2. Where do ManySeg's decoders sit on the accuracy vs FLOPs Pareto front?

## Protocol
* Encoder resnet34 (`tv_in1k`), Pet 256², 20 epochs, recipe = the better E002 recipe
  (chosen by E002 val mIoU before E003 starts), seeds 0–2.
* Decoders: unet, fastai_unet, fpn, deeplabv3plus, upernet, segformer (default widths).
* Metrics: final val mIoU, val boundary F1, params, FLOPs at 256² (exact; hardware
  independent). Latency is measured separately and labelled contended.
* Budget: 18 runs, about 2 h on the shared 4090. No extra runs unless a difference larger than
  2 seed-SDs appears.

## Decision rule
A decoder is "better" only if its mean exceeds the reference `unet` by more than 2× the pooled
seed SD. Otherwise we report no detectable difference and prefer the cheaper decoder.

## Amendment (2026-10-14 10:40, before any E003 run)
E002 showed that the Pet validation split (376 images) cannot resolve differences of about 0.3
mIoU points, while the test split (3,669 images) can. Each E003 configuration is therefore
evaluated **once** on test as well (`--evaluate-test`). The decision rule still uses val. Test
results are reported alongside and are not used to choose anything. The same amendment applies
to E001b. Recipe chosen by the pre-registered rule: `fastai_like`.
