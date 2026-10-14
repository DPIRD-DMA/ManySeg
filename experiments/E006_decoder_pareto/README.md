# E006: Decoder Pareto front, accuracy vs cost at reduced output resolution (pre-registered)

Status: **planned** (registered 2026-10-14 before any run; not launched). It will run on
Pawsey Setonix as Slurm arrays (`scripts/setonix/`). Stage 1 is emitted with
`EMIT_TASKS=<file> STAGE=1 bash run.sh`. Stage 2 is emitted with `EMIT_TASKS=<file> STAGE=2
FRONTIER=<frontier.txt> bash run.sh` after `analyse.py --write-frontier` has applied the
selection rule below to the stage-1 results. `experiments/queue6_pareto.sh` is a local
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
  Sen1Floods11; see the preflight in `queue6_pareto.sh`): stop and fix. No partial grid is
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
