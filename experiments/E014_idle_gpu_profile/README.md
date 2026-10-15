# E014: Eager vs torch.compile × fp32/bf16/fp16 on an idle GPU (pre-registered)

Status: **complete** (2026-10-15; RTX 4090, torch 2.14.1, code `2da5f72`). **Both bars pass in bf16 and fp16: `torch.compile` speeds inference up 1.46× (median), and bf16 is 1.61× faster than fp32 without compiling. Training gains are small.**

## Question
What do `torch.compile` and reduced precision buy for ManySeg models, and at what compile cost?
Earlier timings (E002–E006) were eager only. They were contended on the shared 4090, or taken
on MI250X. This sweep is the roadmap's "profiling sweep on an idle GPU".

## Protocol
* Local RTX 4090, launched while the GPU was idle (0% utilisation, about 1 GB used by the
  desktop). Every record carries `gpu_contention_at_start` / `_at_end`. Any config whose
  contention shows other processes is re-run or excluded, and exclusions are listed.
* `manyseg-bench profile`, random weights, 512², 3 → 2 classes. Inference median at batch 1
  and 8 (10 warmup + 30 timed calls). Training step at batch 8 (SGD; `first_call_ms` = first
  call incl. compilation).
* Grid: encoders {resnet34, convnext_tiny} × decoders {unet, fastai_unet, fpn, segformer,
  upernet} × precision {fp32, bf16, fp16} × {eager, `torch.compile` (default mode)} = 60
  configs, run sequentially (`run.sh`). Budget: about 2 h of GPU time.

## Decision rule
* The docs recommend `torch.compile` for inference if its median speedup over eager, at the
  same precision and batch 8, is ≥ 1.2× for most configs (≥ 6 of 10 encoder × decoder pairs,
  in bf16). They state the compile cost (`first_call_ms`) alongside.
* The docs name bf16 as the recommended inference and training precision on Ampere-or-newer
  GPUs if it is ≥ 1.3× faster than fp32 at batch 8 for most pairs. fp16 is reported for
  completeness. Accuracy effects of precision are E008's question, not this one.
* Rankings across decoders (latency, memory) are reported as measured. No decoder default
  changes from timing alone.

## Contention
The first pass overlapped with another agent's test suite, which ran on the GPU from about
08:00 to 08:17. Every config whose record showed other compute processes at its start or end
(24 of 60) was re-run on the idle GPU (`rerun_contended.sh`, same snapshot). `analyse.py` uses
the latest record per config and refuses contended ones, and none remain.

**Compile cost caveat:** the re-runs found Inductor's on-disk cache warm, so their
`first_call_ms` is about 2 s. Cold compile cost therefore comes only from the 36 first-pass
records (the rows with first-call values above 2 s): **8–15 s for the first inference call and
28–54 s for the first training step**, per new shape or mode. Each new batch size or the switch
to `train()` triggers a recompile.

## Results (512², latest record per config; ms are medians)

| encoder | decoder | prec | eager b1 ms | compiled b1 ms | eager b8 ms | compiled b8 ms | compile ×(b8) | train eager ms | train compiled ms | compile first call s (inf b1 / train) | train peak MiB eager / compiled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny | fastai_unet | fp32 | 18.8 | 16.1 | 145.0 | 124.2 | 1.17 | 442 | 396 | 9 / 36 | 15357 / 14909 |
| convnext_tiny | fastai_unet | bf16 | 15.9 | 10.5 | 92.3 | 70.1 | 1.32 | 273 | 233 | 2 / 2 | 11649 / 9165 |
| convnext_tiny | fastai_unet | fp16 | 16.0 | 10.4 | 89.7 | 67.3 | 1.33 | 265 | 225 | 2 / 2 | 11649 / 9165 |
| convnext_tiny | fpn | fp32 | 4.0 | 3.1 | 25.2 | 18.4 | 1.37 | 98 | 73 | 2 / 2 | 5308 / 5105 |
| convnext_tiny | fpn | bf16 | 6.0 | 2.2 | 14.0 | 8.0 | 1.76 | 41 | 38 | 2 / 2 | 3573 / 2821 |
| convnext_tiny | fpn | fp16 | 6.3 | 2.2 | 13.9 | 8.2 | 1.70 | 41 | 37 | 2 / 2 | 3573 / 2821 |
| convnext_tiny | segformer | fp32 | 3.6 | 3.0 | 25.4 | 21.7 | 1.17 | 97 | 81 | 2 / 2 | 5938 / 5914 |
| convnext_tiny | segformer | bf16 | 5.0 | 2.2 | 13.1 | 8.9 | 1.48 | 39 | 44 | 2 / 2 | 4076 / 3595 |
| convnext_tiny | segformer | fp16 | 5.3 | 2.1 | 13.1 | 8.6 | 1.52 | 39 | 44 | 9 / 34 | 4078 / 3660 |
| convnext_tiny | unet | fp32 | 4.5 | 3.3 | 29.0 | 24.4 | 1.19 | 112 | 91 | 9 / 40 | 6375 / 6166 |
| convnext_tiny | unet | bf16 | 5.7 | 2.5 | 14.6 | 9.6 | 1.52 | 46 | 48 | 12 / 46 | 3754 / 3407 |
| convnext_tiny | unet | fp16 | 5.6 | 2.4 | 14.0 | 9.0 | 1.57 | 44 | 46 | 12 / 44 | 3761 / 3407 |
| convnext_tiny | upernet | fp32 | 5.9 | 4.5 | 39.7 | 30.6 | 1.29 | 141 | 116 | 13 / 49 | 7297 / 6935 |
| convnext_tiny | upernet | bf16 | 6.9 | 3.1 | 22.0 | 15.0 | 1.46 | 66 | 63 | 15 / 54 | 4922 / 3740 |
| convnext_tiny | upernet | fp16 | 6.9 | 3.0 | 22.0 | 14.9 | 1.47 | 66 | 63 | 14 / 50 | 4922 / 3740 |
| resnet34 | fastai_unet | fp32 | 11.5 | 11.6 | 78.9 | 61.8 | 1.28 | 224 | 203 | 9 / 28 | 8804 / 7742 |
| resnet34 | fastai_unet | bf16 | 9.3 | 5.0 | 47.6 | 36.0 | 1.32 | 141 | 120 | 10 / 32 | 5809 / 5028 |
| resnet34 | fastai_unet | fp16 | 8.7 | 5.0 | 47.4 | 36.0 | 1.32 | 133 | 113 | 10 / 31 | 5808 / 5028 |
| resnet34 | fpn | fp32 | 3.6 | 2.4 | 12.4 | 9.9 | 1.25 | 40 | 36 | 10 / 29 | 2191 / 1806 |
| resnet34 | fpn | bf16 | 5.0 | 1.6 | 9.1 | 6.3 | 1.46 | 27 | 22 | 2 / 2 | 1727 / 1058 |
| resnet34 | fpn | fp16 | 4.6 | 1.6 | 9.0 | 6.4 | 1.40 | 26 | 22 | 2 / 2 | 1727 / 1058 |
| resnet34 | segformer | fp32 | 3.2 | 2.5 | 13.1 | 11.9 | 1.09 | 43 | 40 | 1 / 1 | 2825 / 2617 |
| resnet34 | segformer | bf16 | 4.1 | 1.5 | 9.1 | 5.6 | 1.64 | 30 | 25 | 2 / 1 | 2229 / 1828 |
| resnet34 | segformer | fp16 | 3.8 | 1.5 | 9.0 | 5.5 | 1.64 | 29 | 24 | 2 / 1 | 2228 / 1828 |
| resnet34 | unet | fp32 | 3.8 | 2.8 | 15.2 | 13.0 | 1.17 | 60 | 53 | 8 / 29 | 3355 / 2984 |
| resnet34 | unet | bf16 | 5.0 | 1.7 | 11.7 | 8.4 | 1.40 | 44 | 32 | 8 / 30 | 2056 / 1741 |
| resnet34 | unet | fp16 | 4.9 | 1.5 | 11.3 | 7.8 | 1.45 | 41 | 31 | 8 / 29 | 2055 / 1733 |
| resnet34 | upernet | fp32 | 5.3 | 3.9 | 26.2 | 19.6 | 1.34 | 83 | 74 | 11 / 34 | 4178 / 3632 |
| resnet34 | upernet | bf16 | 6.0 | 2.5 | 18.0 | 12.7 | 1.42 | 54 | 44 | 2 / 2 | 3067 / 1959 |
| resnet34 | upernet | fp16 | 5.4 | 2.5 | 17.7 | 12.6 | 1.41 | 53 | 46 | 12 / 36 | 3066 / 1960 |

compile speedup, inference b8, fp32: median 1.22x, >=1.2x in 5/10; train step median 1.13x
compile speedup, inference b8, bf16: median 1.46x, >=1.2x in 10/10; train step median 1.17x
compile speedup, inference b8, fp16: median 1.46x, >=1.2x in 10/10; train step median 1.17x
bf16 vs fp32 (eager, inference b8): median 1.61x, >=1.3x in 9/10
fp16 vs fp32 (eager, inference b8): median 1.64x, >=1.3x in 10/10

## Observations
* **Inference:** compile gives ≥ 1.2× at batch 8 in all 10 encoder × decoder pairs in bf16 and
  fp16 (1.32–1.76×), but in only 5 of 10 in fp32 (median 1.22×). At batch 1 it does better
  still: in bf16, eager is launch-bound (e.g. ResNet-34 U-Net 5.0 → 1.7 ms).
* **Precision:** eager bf16 is ≥ 1.3× faster than fp32 at batch 8 in 9 of 10 pairs (median
  1.61×). fp16 matches bf16. At batch 1, **eager bf16 is slower than eager fp32** for the light
  decoders (autocast cast overhead dominates). Compiled bf16 is the fastest option everywhere.
* **Training step:** compile helps by only 1.17× (median, bf16), and is slower in 2 ConvNeXt
  bf16 configs (unet 46 → 48 ms, segformer 39 → 44 ms). Peak training memory drops 3–39% with
  compile. With 28–54 s cold compile per shape, compile is not worth it by default for short
  training runs.
* **Decoder ranking** (eager bf16, batch 8, ResNet-34): segformer 9.1 ≈ fpn 9.1 < unet 11.7 <
  upernet 18.0 < fastai_unet 47.6 ms. This matches E006's FLOP ordering. Timing alone changes no
  default.

## Decision
* The docs recommend **`torch.compile` for inference** (≥ 1.2× in 10/10 pairs in bf16/fp16),
  with the cold compile cost (8–15 s per input shape; use `dynamic=True` for varying sizes, see
  Phase 2's single-graph test) stated next to it. For training, compile is optional: a median
  1.17× gain, sometimes slower, and 28–54 s per shape.
* The docs recommend **bf16 autocast** for inference and training on Ampere-or-newer GPUs
  (≥ 1.3× in 9/10 pairs). The note: at batch 1 in eager mode, fp32 can be faster; compile
  removes that. Accuracy effects belong to E008.
