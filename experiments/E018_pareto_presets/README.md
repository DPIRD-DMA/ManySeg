# E018: Pareto-front presets: encoder × decoder width (pre-registered)

Status: **done** (registered 2026-10-15 before any run; run 2026-10-15 at code `586716d`). Result: four PVTv2-b0 configs are on the front of both datasets and became `manyseg.presets`; the ResNet-34 U-Net default is dominated.

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

## Results

All 32 screen runs, 36 confirm runs and 16 profiles completed; no failures. Latency is the
median of 30 compiled bf16 forwards (batch 8, 512², 15 channels) on the 4090, idle at the time
but a shared machine. Val mIoU is mean ± SD over seeds (n = 3 for confirmed configs, else seed 0
only). Full tables: `uv run python experiments/E018_pareto_presets/analyse.py`.

| config | latency b8 (ms) | GFLOPs | params | Pet val | Sen1Floods11 val | front |
|---|---|---|---|---|---|---|
| pvt_v2_b0-segformer | 4.5 | 17.5 | 3.8 M | 0.8197 ± 0.0008 | 0.9012 ± 0.0019 | both |
| pvt_v2_b0-unet_w25 | 4.9 | 10.4 | 3.8 M | 0.8229 ± 0.0010 | 0.9009 ± 0.0019 | both |
| pvt_v2_b0-unet_w50 | 5.3 | 13.8 | 4.2 M | 0.8254 ± 0.0004 | 0.9024 ± 0.0009 | both |
| pvt_v2_b0-unet | 7.3 | 27.2 | 5.7 M | 0.8268 ± 0.0003 | 0.9015 ± 0.0042 | both |
| efficientnet_b0-unet | 8.3 | 23.6 | 5.8 M | 0.8203 ± 0.0002 | 0.8813 ± 0.0095 | — |
| resnet34-unet_w25 | 8.6 | 46.8 | 21.9 M | 0.8186 ± 0.0010 | 0.8916 ± 0.0016 | — |
| resnet34-unet_w50 | 9.0 | 51.4 | 22.6 M | 0.8209 ± 0.0021 | 0.8909 ± 0.0058 | — |
| resnet34-unet (default) | 10.9 | 67.5 | 24.4 M | 0.8231 ± 0.0005 | 0.8901 ± 0.0062 | — |

Not confirmed (seed 0 only, more than 0.5 val mIoU behind a faster front member on both
datasets): all four ResNet-18 configs and EfficientNet-B0 `unet_w50` / `unet_w25` / `segformer`.

* **Presets (decision rule):** the four PVTv2-b0 configs are on the front of both datasets, so
  they become `manyseg.presets`: `fast` (SegFormer head), `compact` (quarter-width U-Net),
  `balanced` (half-width U-Net), `accurate` (full U-Net). All opt-in; `create_model` defaults
  are unchanged (SMP parity).
* **"Own architecture":** `pvt_v2_b0-unet_w50` dominates the ResNet-34 U-Net on both datasets:
  51% lower latency, +0.23 val on Pet (pooled SD 0.0005) and +1.23 on Sen1Floods11 (pooled SD
  0.0044); test agrees (+0.28, +1.03). The criterion is met, but by an off-the-shelf timm
  encoder plus a narrower decoder, not by a new architecture. We do not build one.
* **Decoder width:** for both 3-seed encoders, halving the U-Net widths changed val mIoU by
  −0.22 to +0.09 (ResNet-34 Pet −0.22, PVTv2-b0 Pet −0.14, both +0.08/+0.09 on Sen1Floods11) and
  saved 17–27% latency; the default widths are wider than these tasks need.
* **FLOPs are a poor latency proxy:** EfficientNet-B0 has the fewest FLOPs (6.6 G with
  `unet_w25`) but is slower than PVTv2-b0 at the same width on the GPU (depthwise convolutions).
* **`fast` vs the default:** `fast` is 2.4× faster but 0.34 val lower on Pet (pooled SD 0.0007)
  and +1.11 on Sen1Floods11, so it trades accuracy rather than dominating. The other three
  presets dominate the default on both datasets.
* **Deviation:** `analyse.py` uses a 0.5 val mIoU dominance margin when either config has only
  one seed, which the front rule above does not define (it matches the stage-2 "within 0.5"
  rule). It affects only the seven unconfirmed configs; all four presets and the reference have
  three seeds and are judged by 1 pooled SD.
* **CPU (supplementary, not pre-registered):** eager fp32, batch 1, 512², 15 channels, 8 threads,
  on the shared workstation under other load (load average ~9; results in `runs/profile_cpu`).
  Median ms (p10–p90): `fast` 137 (134–147), `compact` 175 (147–207), `balanced` 110 (106–113),
  `accurate` 142 (128–149), ResNet-34 U-Net 162 (157–167). Every preset is at least as fast as the
  default on CPU, but the margin is smaller than on the GPU and the spread is too wide to rank the
  presets against each other (`compact` slower than `balanced` is not plausible); re-measure on
  an idle machine before quoting CPU numbers.
* **Caveats:** two datasets, both binary-or-few-class at moderate resolution; one GPU type. The
  differences among the four presets on Sen1Floods11 are within noise. CPU latency is below.
