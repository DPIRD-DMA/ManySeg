# Roadmap

Status as of 2026-10-14. ✅ done · 🔄 in progress · ⏳ planned.

## Phase 1: research and architecture ✅

* ✅ `architecture_review.md` covers SMP vs fastai (capacity, recipe, alignment), timm
  capabilities, MMSeg, TerraTorch, TorchGeo and HF.
* ✅ `api_design.md`, `testing_strategy.md`, `benchmark_design.md`, this roadmap.

## Phase 2: minimal vertical slice ✅

* ✅ `TimmBackbone` (any `features_only` timm CNN), `FeatureSpec`, five channel adapters.
* ✅ `UNetDecoder` (SMP-equivalent by default; `upsample` = nearest | bilinear | aligned).
* ✅ Pixel and image heads, `DenseModel`, `create_model` (Tensor or named-dict outputs).
* ✅ Dynamic, odd and non-square sizes with an explicit pad-and-crop policy.
* ✅ CPU / CUDA (MPS is tested where available), AMP, channels_last, state_dict.
* ✅ `torch.compile` fullgraph+dynamic (1 graph for 10 sizes on CUDA), `torch.export`
  dynamic H/W, ONNX with onnxruntime parity.

## Phase 3: independent benchmark 🔄

* ✅ Provider interface; ManySeg, SMP and fastai providers; weight fingerprints.
* ✅ Oxford Pet preparation with integrity checks; GPU-resident seeded loader and augmentation.
* ✅ Recipes (`standard`, `fastai_like`), trainer, metrics (mIoU, Dice, boundary F1),
  profiler, provenance with GPU contention, JSON results, report table.
* 🔄 E002: fair three-way comparison on Pet (2 recipes × 3 seeds).
* ⏳ Profiling sweep (eager vs compile; fp32, bf16, fp16; batch sizes), on an idle GPU.
* ⏳ CI workflow (CPU unit tests; slow tier nightly).

## Phase 4: expanded architecture support ⏳

Priority order. Each item comes with tests first.

1. **Capability probe and registry** for timm models (feature API, layout, size class,
   rounding, divisor, compile/export in a subprocess with a timeout). Generated table in docs.
2. **Transformer backbones**: hierarchical models (Swin with `strict_img_size=False`, NHWC→NCHW;
   PVTv2; Hiera at a fixed size) and plain ViT/EVA (`dynamic_img_size` + `dynamic_img_pad`,
   with the token grid computed from the input) plus a **ViT pyramid neck**
   (ViTDet/DPT-style, configurable scales).
3. **Decoders**: FPN, DeepLabV3+, UPerNet, SegFormer-MLP (lightweight), U-Net++, and a
   fastai-style Dynamic U-Net (ICNR shuffle, skip BN, input-image cross, middle block),
   all against the common decoder interface.
4. **Multi-input fusion**: per-input backbones (independent or shared), early, intermediate
   and late fusion; concat, add, gated, attention and cross-attention ops; explicit
   alignment policies (`same_grid`, `same_extent`).
5. **Multi-task**: per-head task-specific decoders; uncertainty head (mean + log-variance).
6. **Export helpers**: `manyseg.export.dynamic_shapes(model)` returning valid `Dim` ranges.
   Quantisation entry points (torchao) kept separate from model construction.

## Phase 5: research and optimisation ⏳

Candidate experiments. A budget is set before each one starts.

| Id | Question | Budget |
| --- | --- | --- |
| E003 | Is fastai's advantage (if any) decoder capacity? Width-matched U-Nets. | 3 seeds × 3 configs, Pet |
| E001b | Does `aligned` upsampling improve trained mIoU or boundary F1? | 3 seeds × 3 modes, Pet + WHU |
| E004 | Channel adapters on multispectral data (MADOS, Sen1Floods11 S2) | 5 adapters × 3 seeds |
| E005 | Early vs intermediate vs late fusion on Sen1Floods11 (S1+S2) | 3 × 5 seeds |
| E006 | Decoder Pareto front (accuracy vs FLOPs and latency) across decoders and backbones | core tier, 1 seed, then 3 seeds on the frontier |
| E007 | Normalisation in the decoder (BN / GN / none + skip BN) at small batch sizes | 3 seeds |
| E008 | bf16 / fp16 / int8 (torchao) accuracy loss vs speed | post-training, no training cost |

Stopping rule: an idea is kept only if it beats its baseline by more than 2 seed standard
deviations on at least two datasets, at no more than a small, stated cost in latency or memory.
Negative results are written up in `experiments/` like positive ones.

## Phase 6: release preparation ⏳

* Docs site, examples (single input, multispectral, multi-modal, multi-task, export).
* PyPI packaging for `manyseg` and `manyseg-bench` independently. Supported torch/timm range
  tested in CI.
* Clear error messages for unsupported backbone/size combinations, using the capability table.

## Known risks

| Risk | Mitigation |
| --- | --- |
| Swin dynamic export hangs inside timm | Probe in a subprocess with a timeout. Document it. Offer fixed-size export. |
| The configuration surface grows with fusion × decoder × head combinations | Keep composition over configuration. `create_model` only parses into modules, and new options need an experiment or a test. |
| Benchmark results on a shared GPU | Contention is recorded with each result. Timing claims are re-measured on an idle GPU. |
| Leakage in remote-sensing splits | Official geographic splits, or grid-cell assignment. Document the protocol for each dataset. |
| Upstream API drift (timm, SMP, fastai) | Pin tested versions in results; network and slow tests run nightly. |
