# E012: Does fastai's full-resolution "input cross" stage alone close the gap? (pre-registered)

Status: **planned** (registered 2026-10-14 before any run). Follow-up to E006.

## Background
In E006, `fastai_unet` at out_stride 1 beat `unet` on Sen1Floods11 by +2.4 val / +1.7 test mIoU
(ResNet-34) and +1.1 / +1.2 (PVTv2-b2). The gain disappeared at out_stride 2, which removes only
fastai's full-resolution stages: the concatenation of the raw input image at stride 1 plus a
residual block. That points to this stage, not fastai's 4× wider decoder, as the source.
`unet` now has `decoder_kwargs={"input_cross": True}`, which adds exactly that stage (input
concat + fastai's residual block; about +7k parameters) and is otherwise SMP-identical.

## Hypothesis
`unet` + `input_cross` recovers most of fastai_unet's Sen1Floods11 gain over `unet` at a small
fraction of fastai_unet's latency and memory. On Pet (E002/E003: fastai +0.3) the effect is
small.

## Protocol
* Identical to E006 (Pawsey Setonix MI250X, recipe `fastai_like`; Pet 20 epochs; Sen1Floods11
  stacked S1+S2, 60 epochs, batch 16; test evaluated once per run).
* Settings: Sen1Floods11 × {resnet34 `tv_in1k`, pvt_v2_b2 `in1k`}; Pet × resnet34.
* Arms: `unet` (out_stride 1), `unet` + `input_cross`, `fastai_unet` (out_stride 1). Seeds 0–2.
  All arms are rerun on the same code, not reused from E006.
* Profiling sweep (`STAGE=profile`) for the `unet` + `input_cross` configs, as in E006.
* Budget: 27 training runs + 4 profiling configs, about 2 h of GCD time.

## Decision rule
* Gap closed = (cross − unet) / (fastai − unet) in mean val mIoU on Sen1Floods11, per encoder.
* `input_cross` "captures the gain" if cross − unet > 2 pooled seed SDs on val for **both**
  encoders, test points the same way, and the gap closed is ≥ 50%. Then the docs recommend it
  for fine-structure remote-sensing tasks. Whether it becomes a default is decided separately;
  defaults stay SMP-equivalent.
* Otherwise: a negative result. The fastai gain then needs width plus full resolution together.
