# Roadmap

Status as of 2026-10-14 (afternoon update). ✅ done · 🔄 in progress · ⏳ planned.

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
* ✅ E002: fair three-way comparison on Pet (2 recipes × 3 seeds). The recipe is worth +0.8 mIoU; fastai's decoder +0.3 on test at 4× FLOPs.
* ✅ Sen1Floods11 (S1 + S2 + DEM) multi-input dataset; `stack` vs `dict` input modes.
* ✅ E003 (decoders on Pet): unet stays default; fastai_unet +0.4 test mIoU at 4× FLOPs; stride-4 decoders −0.4 to −1.1.
* ✅ E001b (aligned upsampling, trained): no effect, a negative result; nearest stays default.
* 🔄 GPU queue chain: E005 (fusion, running) → E004 (channel adapters) → E010 (instance heads, AI4Boundaries) → E011 (tree points, OAM-TCD) → E009 (metadata conditioning).
* ✅ Datasets prepared: AI4Boundaries ortho (13 GB cache), OAM-TCD (CC-BY, 1024², ~22 GB cache).
* ⏳ Profiling sweep (eager vs compile; fp32, bf16, fp16; batch sizes), on an idle GPU.
* ⏳ CI workflow (CPU unit tests; slow tier nightly).

## Phase 4: expanded architecture support 🔄

1. ✅ **Capability probe** (`manyseg.backbones.capabilities`) and generated
   `docs/backbone_compatibility.md` (46 timm models; 41 any-size; 38 export dynamically).
2. ✅ **Transformer backbones** via meta-device introspection: hierarchical models (NHWC
   normalised), plain ViT/DeiT/EVA/DINOv2 through `ViTPyramidNeck` (any patch size, incl. 14),
   and fixed-size models with `img_size` plus clear errors.
3. ✅ **Decoders**: U-Net, U-Net++, FPN, DeepLabV3+, UPerNet, SegFormer-MLP, fastai-style
   Dynamic U-Net (parameter- and FLOP-identical to fastai).
4. ✅ **Multi-input fusion**: early, intermediate and late; five ops; shared encoders;
   `same_grid` / `same_extent`.
5. ✅ **Multi-task**: named heads, uncertainty heads, per-head task-specific decoders (`"decoder"` key in a head config).
6. ✅ **Export helper** `manyseg.export.export_model`. ⏳ Quantisation entry points (torchao / PT2E).
7. ✅ DDP verified (CPU gloo). ⏳ FSDP2.
8. ✅ **Instance heads** (`manyseg.instances`): boundary + distance presets, torch targets
   (exact EDT), seeded growth or watershed post-processing, with round-trip tests.
9. ✅ **Point and density heads** (`manyseg.points`): heatmap + offset and density presets,
   targets, NMS peaks, counting, offset grouping and panoptic labels, with round-trip tests.
10. ✅ **Bench tasks**: losses, object/point/count metrics, and task-aware training for
    multi-head outputs. ✅ AI4Boundaries and OAM-TCD datasets.
11. ✅ **Metadata conditioning** (`manyseg.conditioning`; branch `feature/metadata-conditioning`):
    `"kind": "vector"` inputs with cyclic / continuous / coords / categorical fields, missing
    values (mask, NaN, learned unknowns, field dropout), and `film` / `head` / `channels`
    injection with exact identity init, export and ONNX. Bench: vectors in dict mode,
    Sen1Floods11 metadata, leave-events-out splits. ⏳ `token` mode for ViTs; E009.

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
| E009 | Does metadata conditioning (FiLM / channels) help flood mapping on held-out events, or does it only memorise events? Pre-registered; `queue3_metadata.sh` | 4 configs × 5 event folds × 3 seeds, plus 6 official-split runs |

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
