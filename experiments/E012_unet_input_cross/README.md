# E012: Does fastai's full-resolution "input cross" stage alone close the gap? (pre-registered)

Status: **complete** (2026-10-14; 27 training runs + 4 profiling configs; Pawsey Setonix, AMD MI250X, code `b385989`). Follow-up to E006. **Positive: `input_cross` captures 81–83% of fastai_unet's Sen1Floods11 gain at ~¼ of its FLOPs.**

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

## Results

Mean ± seed SD over seeds 0–2. All arms were run in the same job on the same code (`b385989`).
"Img/s" is steady-state training throughput from the training runs. Those ran on shared nodes,
so treat it as indicative only.

| Dataset | Encoder | Arm | Val mIoU | Test mIoU | Val bF1 | Test bF1 | Params | GFLOPs/img | Train img/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sen1Floods11 | resnet34 | unet | 0.8930 ± 0.0057 | 0.8840 ± 0.0101 | 0.558 | 0.588 | 24.47 M | 67.5 | 244 |
| Sen1Floods11 | resnet34 | unet + input_cross | 0.9125 ± 0.0037 | 0.9051 ± 0.0010 | 0.598 | 0.642 | 24.49 M | 76.7 | 196 |
| Sen1Floods11 | resnet34 | fastai_unet | 0.9166 ± 0.0010 | 0.9052 ± 0.0015 | 0.609 | 0.654 | 41.31 M | 280.9 | 89 |
| Sen1Floods11 | pvt_v2_b2 | unet | 0.9005 ± 0.0047 | 0.8912 ± 0.0054 | 0.574 | 0.618 | 28.17 M | 71.8 | 89 |
| Sen1Floods11 | pvt_v2_b2 | unet + input_cross | 0.9135 ± 0.0021 | 0.9088 ± 0.0031 | 0.604 | 0.651 | 28.19 M | 81.0 | 82 |
| Sen1Floods11 | pvt_v2_b2 | fastai_unet | 0.9165 ± 0.0016 | 0.9057 ± 0.0007 | 0.619 | 0.666 | 45.94 M | 332.0 | 32 |
| Pet | resnet34 | unet | 0.8222 ± 0.0013 | 0.8219 ± 0.0001 | 0.788 | 0.786 | 24.44 M | 15.7 | 272 |
| Pet | resnet34 | unet + input_cross | 0.8210 ± 0.0011 | 0.8222 ± 0.0005 | 0.786 | 0.786 | 24.44 M | 16.5 | 237 |
| Pet | resnet34 | fastai_unet | 0.8228 ± 0.0008 | 0.8259 ± 0.0003 | 0.790 | 0.794 | 41.22 M | 63.1 | 100 |

Decision quantities on Sen1Floods11 val:

| Encoder | cross − unet | 2 pooled SD | Gap closed | Test cross − unet |
| --- | --- | --- | --- | --- |
| resnet34 | +1.95 | 0.96 | 83% | +2.11 |
| pvt_v2_b2 | +1.30 | 0.73 | 81% | +1.76 |

On test, `unet + input_cross` equals fastai_unet with ResNet-34 (0.9051 vs 0.9052) and is
slightly above it with PVTv2-b2 (0.9088 vs 0.9057). Boundary F1 recovers by the same share.
Seed variance also drops: unet's test SD with ResNet-34 falls from 1.0 to 0.1 points. On Pet,
`input_cross` has no effect (val −0.1, test +0.0), as predicted. fastai_unet's small Pet lead
(+0.4 test) is therefore not the input stage.

### Cost
Profiled at 512², bf16, eager, on a dedicated MI250X GCD. The unet and fastai_unet rows are from
the E006 profiling job (`5d6b2f4`). The input_cross rows are from this job on a different node,
so compare them to within about ±10%.

| Encoder | Arm | Inference b=8 (ms) | Inference peak (MiB) | Train step b=8 (ms) | Train peak (MiB) |
| --- | --- | --- | --- | --- | --- |
| resnet34 | unet | 24.6 | 1279 | 106 | 2520 |
| resnet34 | unet + input_cross | 34.3 | 1537 | 138 | 3192 |
| resnet34 | fastai_unet | 121.2 | 4515 | 342 | 6972 |
| pvt_v2_b2 | unet | 64.7 | 1265 | 309 | 5506 |
| pvt_v2_b2 | unet + input_cross | 74.8 | 1521 | 342 | 6174 |
| pvt_v2_b2 | fastai_unet | 258.1 | 6487 | 952 | 11579 |

The stage costs +14% FLOPs, +16–40% batch-8 latency and +12–27% training memory over `unet`.
fastai_unet costs 4–4.6× FLOPs, 4× latency and 2.1–2.8× training memory.

## Decision
**The gain is captured.** On both encoders cross − unet exceeds 2 pooled SDs on val, test points
the same way, and the gap closed is ≥ 50% (83%, 81%). E006's explanation holds: fastai_unet's
Sen1Floods11 lead comes from reinjecting the raw input at full resolution, not from its 4×
wider decoder.

* The docs recommend `decoder_kwargs={"input_cross": True}` for tasks whose labels depend on
  per-pixel input values at full resolution (water, multispectral and SAR thresholds, thin
  structures). That holds even with strong encoders.
* The default stays SMP-equivalent (`input_cross=False`), as pre-registered. Changing it would
  need a separate experiment on a second remote-sensing dataset (CloudSEN12 is a candidate)
  and on natural images, where it does nothing here.
* Caveats: it only works at `out_stride=1` and single-tensor input (stacked early fusion).
  With `inputs={...}` the factory raises an error. Extending it to multi-input models would
  need a choice of which raster to reinject.
