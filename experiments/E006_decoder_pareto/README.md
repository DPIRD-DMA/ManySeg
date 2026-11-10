# E006: Decoder Pareto front, accuracy vs cost at reduced output resolution (pre-registered)

Status: **complete** (2026-10-14; stage 1: 36 runs, stage 2: 32 runs, profiling: 36 configs; Pawsey Setonix, AMD MI250X, code `5d6b2f4`).
Pawsey Setonix as Slurm arrays (`scripts/setonix/`). Stage 1 is emitted with
`EMIT_TASKS=<file> STAGE=1 bash run.sh`. Stage 2 is emitted with `EMIT_TASKS=<file> STAGE=2
FRONTIER=<frontier.txt> bash run.sh` after `analyse.py --write-frontier` has applied the
selection rule below to the stage-1 results. `experiments/queues/queue6_pareto.sh` is a local
alternative and will not be launched.

Amendment (2026-10-14, before any run): the transformer encoder is `pvt_v2_b2` instead of
`pvt_v2_b1`. Its size (about 25 M encoder parameters) is closer to ResNet-34's.

## Questions
1. How much accuracy does each U-Net-family decoder lose when it stops at stride 2 or 4
   (`out_stride`, branch `feature/reduced-resolution`), and how much compute, memory and
   latency does that save?
2. Where do the reduced U-Nets sit against the decoders that are natively stride 4
   (SegFormer-MLP, FPN, DeepLabV3+) on the accuracy vs FLOPs and accuracy vs latency fronts?
3. Does the answer change with a hierarchical transformer encoder or a remote-sensing dataset?

## Hypothesis (pre-registered)
U-Net at `out_stride=2` loses **less than 1 pooled seed-SD of val mIoU** against
`out_stride=1`. It loses **measurably more boundary F1** (drop > 2 pooled seed-SDs). In
return it saves meaningful resources: **≥ 15% peak training memory or ≥ 10% batch-8
inference latency**. Mechanism: the stride-1 block costs few FLOPs (5.9% of U-Net at 512²
with ResNet-34; the stride-2 and stride-1 blocks together cost 15.6%) but is the most memory- and bandwidth-heavy stage. Its job is to sharpen edges, and
mIoU is dominated by object interiors. The hypothesis is checked separately in each of the
4 cells (dataset × encoder) by `analyse.py`.

## Protocol
* Code: the `feature/reduced-resolution` merge, at one recorded commit for every run.
* Encoders: `resnet34` (`tv_in1k`) and `pvt_v2_b2` (timm `pvt_v2_b2.in1k`; hierarchical
  transformer, strides 4–32). Its family (`pvt_v2_b1`) builds at any size and exports
  dynamically per `docs/backbone_compatibility.md`, and every E006 arm builds with `b2` at 3
  and 15 channels.
* Datasets and recipes (unchanged from earlier experiments, `fastai_like`):
  * Oxford Pet 256², 20 epochs (as E003).
  * Sen1Floods11 stacked S1+S2 (15 channels, `--dataset sen1floods11`), 60 epochs, batch 16,
    256² crops, evaluated on 512² tiles (as E005 `ms-early`).
* Arms (9 per cell): `unet` and `fastai_unet` at `out_stride` 1, 2 and 4; `segformer`, `fpn`
  and `deeplabv3plus` at their native stride 4. Default widths. Every head has `scale=1`, so
  every arm predicts and is evaluated at input resolution.
* What `fastai_unet` drops at reduced strides is defined in its module docstring and in
  `docs/api_design.md`. With ResNet, stride 2 drops the final pixel shuffle, the input cross
  and the full-resolution residual block, about 39% of its FLOPs. Stride 4 also drops the
  stride-2 up-block. With PVT (no stride-2 skip), stride 2 keeps the shuffle from stride 4 to
  stride 2.
* Seeds: **stage 1** runs all 36 configs (9 arms × 2 encoders × 2 datasets) with seed 0.
  **Stage 2** adds seeds 1 and 2 for the frontier configs. The selection rule is fixed here
  and implemented in `analyse.py --write-frontier`. For each cell it takes `unet:1` and
  `unet:2` (the hypothesis pair), then adds the seed-0 Pareto-optimal configs (val mIoU vs
  FLOPs per image) in decreasing val mIoU, up to 4 configs per cell.
* Metrics: final-epoch val mIoU and val boundary F1 (tolerance 2 px) at input resolution;
  params; forward FLOPs per evaluation image (exact, `flops_per_image` in each result);
  training throughput and peak training memory from the training run (contended GPU). Test
  mIoU/BF1 (and Sen1Floods11 Bolivia) is computed once at the end of each run
  (`--evaluate-test`). `analyse.py` prints it only with `--final`, after stage 2, and never
  uses it to select anything.
* Latency and memory sweep (`STAGE=profile`): `manyseg-bench profile` per encoder × arm at
  256² × 3 channels and 512² × 15 channels; bf16, eager; inference batch 1 and 8; training
  step at batch 8 (peak memory). Each record carries GPU contention. Latency claims will only
  be quoted from a re-run on an idle GPU (`STAGE=profile bash run.sh`, no training needed).
* Plots (`analyse.py --plot`): val mIoU (± seed SD) against FLOPs and against batch-8
  latency, one panel per cell.

## Budget
* Stage 1: 36 runs. Stage 2: at most 16 configs × 2 seeds = 32 runs. Profile: 36 calls.
* Estimate from E003/E005 throughput: about 4 h + 3.5 h + 1 h, so about 9 GPU-hours on the
  shared 4090, less on Setonix (PVT and `fastai_unet` arms are the slow ones). Hard cap: 68 training runs and
  10 GPU-hours. No additional arms, epochs or hyper-parameter tuning.

## Stopping rule
* A smoke run of each new ingredient fails (reduced `fastai_unet`, PVT weights,
  Sen1Floods11; see the preflight in `queues/queue6_pareto.sh`): stop and fix. No partial grid is
  started.
* A run that fails is retried once. If it fails again it is reported as failed and its
  config is excluded from the frontier for that cell (no substitution).
* Stage 2 runs only the configs listed by the pre-registered selection. The experiment ends
  when stage 2 and the profile sweep are done, or at the cap.

## Decision rule
* **Hypothesis**: supported if it holds in at least 3 of the 4 cells.
* **Recommendation for `out_stride`**: if U-Net `out_stride=2` (or `fastai_unet`
  `out_stride=2`) is within 1 pooled seed-SD of its `out_stride=1` val mIoU in at least 3
  cells, the docs recommend it as the speed or memory option for mIoU-driven tasks. The
  boundary F1 cost will be stated next to it. Defaults do not change: `out_stride=1` stays
  the default and stays SMP- and fastai-identical.
* **Stride-4 decoders vs reduced U-Nets**: a reduced U-Net "beats" a stride-4 decoder only
  if its val mIoU is higher by more than 2 pooled seed-SDs at no more FLOPs, as in the E003
  rule. Otherwise we report no detectable difference and prefer the cheaper one.

## Results
Pending.

## Results

Full tables: `results.md` (val mIoU, boundary F1, GFLOPs, params, latency per batch of 8 in
bf16, peak training memory, test mIoU). Plot: `pareto.png`. Latency and memory are from the
`STAGE=profile` sweep on a dedicated MI250X GCD, not contended. Compare them only within this
experiment, never against 4090 numbers.

Key configurations (mean of 3 seeds where n = 3; test evaluated once per run):

| dataset / encoder | config | val mIoU | val boundary F1 | test mIoU | ms / batch 8 | train MiB |
|---|---|---|---|---|---|---|
| Pet / resnet34 | unet:1 | 0.8219 ± 0.0007 | 0.7890 ± 0.0030 | 0.8222 | 9.2 | 652 |
| Pet / resnet34 | unet:2 | 0.8229 ± 0.0015 | 0.7887 ± 0.0025 | 0.8227 | 8.0 | 550 |
| Pet / pvt_v2_b2 | unet:1 | 0.8396 ± 0.0012 | 0.8212 ± 0.0015 | 0.8385 | 19.4 | 1417 |
| Pet / pvt_v2_b2 | unet:2 | 0.8391 ± 0.0005 | 0.8210 ± 0.0007 | 0.8381 | 18.2 | 1264 |
| S1F11 / resnet34 | unet:1 | 0.8911 ± 0.0031 | 0.5614 ± 0.0100 | 0.8873 | 24.6 | 2520 |
| S1F11 / resnet34 | unet:2 | 0.8889 ± 0.0049 | 0.5598 ± 0.0080 | 0.8867 | 19.8 | 2072 |
| S1F11 / resnet34 | fastai_unet:1 | **0.9153 ± 0.0020** | **0.6056 ± 0.0036** | **0.9047** | 121.2 | 6972 |
| S1F11 / resnet34 | fastai_unet:2 | 0.8949 ± 0.0031 | 0.5580 ± 0.0041 | 0.8857 | 36.6 | 2853 |
| S1F11 / pvt_v2_b2 | unet:1 | 0.9023 ± 0.0024 | 0.5738 ± 0.0018 | 0.8915 | 64.6 | 5506 |
| S1F11 / pvt_v2_b2 | unet:2 | 0.9039 ± 0.0015 | 0.5824 ± 0.0058 | 0.8957 | 59.8 | 4898 |
| S1F11 / pvt_v2_b2 | fastai_unet:1 | **0.9137 ± 0.0007** | **0.6171 ± 0.0027** | **0.9038** | 258.1 | 11579 |

## Hypothesis (pre-registered): not supported, in a useful direction
"U-Net at out_stride 2 loses < 1 seed SD of mIoU but measurably more boundary F1."
In all four dataset/encoder combinations, `unet:2` differs from `unet:1` by less than 1 seed SD
in mIoU **and** in boundary F1 (BF1 change −0.009 to +0.0003 val; test mIoU within ±0.004). It
saves 11–18% training memory and 6–20% latency. The predicted boundary cost did not
appear. Convolutions at stride 2 plus bilinear head upsampling preserve boundaries as well as
the full-resolution stage does on these datasets. `out_stride=4` *does* cost on Sen1Floods11
(−1 mIoU, −5 to −7 BF1; 1 seed).

## Other findings
* **fastai_unet at full resolution is the accuracy leader on Sen1Floods11**: +2.4 val / +1.7
  test mIoU over `unet` with ResNet-34, +1.1 / +1.2 with PVTv2-b2, and about +4 points boundary
  F1. It costs 4.9×/4.0× latency and 2.8×/2.1× training memory. **The gain disappears at
  out_stride 2** (fastai_unet:2 ≈ unet). So it comes from the full-resolution stages that
  stride 2 removes (concatenation of the raw input image + residual block at stride 1), not
  from decoder width. On Pet the same stages are worth only about +0.3 (E002/E003).
* **PVTv2-b2 beats ResNet-34** at similar parameters on both datasets (+1.7 Pet, +0.4–1.1
  S1F11 test) at about 2–2.6× latency on the MI250X.
* **Pareto front:** `unet:2`/`unet:4` dominate `unet:1` on cost at equal accuracy everywhere.
  `segformer` and `deeplabv3plus` are the cheapest but lose 0.5–1 mIoU. `fastai_unet:1` is the
  sole high-accuracy point on Sen1Floods11.

## Decisions
* Document `decoder_kwargs={"out_stride": 2}` for `unet` as the recommended speed/memory
  setting. It is not made the default, which stays SMP-equivalent.
* New candidate experiment (not yet registered): add fastai's full-resolution input-cross +
  residual stage to `unet` as an option, and test whether it captures fastai_unet's
  Sen1Floods11 gain at a fraction of the cost (most of fastai_unet's cost is the wide
  decoder, which this result suggests is not the source of the gain).
