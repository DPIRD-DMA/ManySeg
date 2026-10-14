# E014: Eager vs torch.compile × fp32/bf16/fp16 on an idle GPU (pre-registered)

Status: **planned** (registered 2026-10-15, before any run).

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
