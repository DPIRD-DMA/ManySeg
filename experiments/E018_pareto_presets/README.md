# E018: Pareto-front presets: encoder × decoder width (pre-registered)

Status: **planned** (registered 2026-10-15, before any run).

## Question
Which encoder and decoder combinations sit on the accuracy–latency Pareto front? Can a few
evidence-backed presets replace "pick SMP's U-Net + ResNet-34"? FLOP breakdown at 512² (15 ch,
ResNet-34 U-Net): encoder 43.2 GFLOPs, decoder and heads 24.3 (36%). The SMP-default decoder widths
(256…16) have never been tested here. E006 showed `out_stride` 2 is free and that PVTv2-b2 beat
ResNet-34; E014 showed bf16 + `torch.compile` is the deployed speed setting.

## Protocol
* Local RTX 4090 (no Pawsey units). Recipe `fastai_like`. Pet (20 epochs) and Sen1Floods11 S1+S2
  stacked (60 epochs, batch 16).
* Grid: encoders {ResNet-18, ResNet-34 (`tv_in1k`), EfficientNet-B0 (`ra_in1k`), PVTv2-b0
  (`in1k`)} × decoders {`unet` (default widths 256…16), `unet_w50` (128, 64, 32, 16, 8),
  `unet_w25` (64, 32, 16, 8, 8), `segformer`} = 16 configs × 2 datasets.
* **Stage 1 (screen):** seed 0 for all 32 runs. **Stage 2 (confirm):** seeds 1–2 for every
  config on, or within 0.5 val mIoU of, the stage-1 front of either dataset, plus the reference
  `resnet34-unet`.
* **Cost axis:** latency from `STAGE=profile` (compiled, bf16, batch 8, 512², idle 4090, E014
  protocol), plus FLOPs and parameters. Accuracy axis: val mIoU (test reported).
* Budget: about 32 + ≤ 24 runs at ~2–3 min, plus 16 profiles, about 3 h of 4090 time.

## Decision rule
* A config is on the **front** of a dataset if no other config has higher mean val mIoU by more
  than 1 pooled SD **and** lower latency.
* **Presets:** configs on the front of **both** datasets become named presets (e.g. `fast`,
  `balanced`, `accurate`), added to the library (`manyseg.presets`) and docs with their measured
  numbers. A preset that changes a decoder default (e.g. narrower widths) is opt-in. `unet`'s
  default widths keep SMP parity.
* **"Own architecture":** claimed only if a non-default combination dominates the SMP default
  (ResNet-34 U-Net): at least as accurate (within 1 pooled SD, or better) at ≥ 25% lower latency on
  both datasets. Otherwise the result is "the default is already on the front", and is reported
  that way.
