# Roadmap

Status as of 2026-10-15, 08:35 AWST. ✅ done · 🔄 in progress · ⏳ planned.

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
* ✅ E005 (fusion, Sen1Floods11): negative. No fusion arm beats S2-only or early stacking; SMP early fusion equals ManySeg early fusion.
* ✅ E004 (stem adapters, 13-band S2; Setonix): `repeat` stays default; scratch matches ImageNet init; `projection` is harmful.
* ✅ E009 (metadata conditioning, leave-events-out; Setonix): FiLM beats metadata-as-channels (2.9 SE); FiLM vs none +0.9 water IoU, under the 2 SE bar; the lat/lon shortcut criterion was met.
* ✅ E010 (instance heads, AI4Boundaries; Setonix): positive. Boundary + distance doubles PQ vs extent + CC; per-head decoders add +0.06 PQ.
* ✅ E011 (tree points, OAM-TCD; Setonix): MSE heatmap stays default; the density head halves count MAE; the fixed peak threshold cost recall.
* ✅ E011b (val-calibrated peak threshold): positive. Test F1 0.40 → 0.59; now the bench default (`calibrate=True`). Calibrated peak counting comes close to the density head.
* ✅ E006 (decoder Pareto; Setonix): unet `out_stride` 2 is free (no mIoU or boundary-F1 loss, 11–18% less memory); fastai_unet's Sen1Floods11 lead comes from its full-resolution stages.
* ✅ E012 (unet `input_cross`; Setonix): positive. It recovers 81–83% of fastai_unet's Sen1Floods11 gain at +14% FLOPs (fastai: 4×), with no effect on Pet. Opt-in; the default stays SMP-identical.
* ✅ Execution on Pawsey Setonix (MI250X) via Slurm arrays: `scripts/setonix/` (see its README for the container, offline weights, MIOpen and Lustre gotchas).
* ✅ Datasets prepared: AI4Boundaries ortho (13 GB cache), OAM-TCD (CC-BY, 1024², ~22 GB cache).
* ✅ CloudSEN12+ high (clouds/shadows, S2 L1C; CC0 labels + Copernicus Sentinel notice): `cloudsen12` (S2), `cloudsen12_s2_aux` (+ static S1GBM backscatter + DEM) and `cloudsen12_meta` (+ lat/lon, date, sun angles) on ROI-disjoint splits; 2048² samples tiled to 512².
* ✅ E013 (CloudSEN12+; Setonix): negative on all three questions. S1GBM/DEM layers, metadata (FiLM, lat/lon on unseen locations) and `input_cross` give no gain; `input_cross` stays a targeted option for tasks whose labels follow per-pixel input values. 5,126 Pawsey units.
* ✅ E014 (profiling sweep on an idle 4090): bf16 1.61× over fp32; compile 1.46× more for inference in bf16 (10/10 pairs), only 1.17× for training; cold compile 8–15 s (inference) / 28–54 s (train) per shape.
* ✅ CI workflow (Linux unit tests on every push; macOS/MPS on PRs and nightly; slow tier nightly).

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
6. ✅ **Export helper** `manyseg.export.export_model`. ✅ Quantisation entry points `manyseg.quantize` (`manyseg[quant]`, torchao): PT2E static int8 with dynamic H/W, int8 weight-only / dynamic, size and agreement reports.
7. ✅ DDP verified (CPU gloo). ✅ FSDP2: `manyseg.distributed.shard_model` (per-stage `fully_shard`), matching DDP after a step on CPU (gloo).
8. ✅ **Instance heads** (`manyseg.instances`): boundary + distance presets, torch targets
   (exact EDT), seeded growth or watershed post-processing, with round-trip tests.
9. ✅ **Point and density heads** (`manyseg.points`): heatmap + offset and density presets,
   targets, NMS peaks, counting, offset grouping and panoptic labels, with round-trip tests.
10. ✅ **Bench tasks**: losses, object/point/count metrics, and task-aware training for
    multi-head outputs. ✅ AI4Boundaries and OAM-TCD datasets.
11. ✅ **Metadata conditioning** (`manyseg.conditioning`):
    `"kind": "vector"` inputs with cyclic / continuous / coords / categorical fields, missing
    values (mask, NaN, learned unknowns, field dropout), and `film` / `head` / `channels`
    injection with exact identity init, export and ONNX. Bench: vectors in dict mode,
    Sen1Floods11 and CloudSEN12+ metadata, leave-events-out splits (E009 and E013 done).
    `token` mode for ViTs: not planned, since metadata showed no benefit in E009 or E013.
12. ✅ **Reduced output resolution**: decoder `out_stride`
    for `unet` / `unetplusplus` / `fastai_unet`, where reduced decoders are strict prefixes of
    the default and the defaults keep SMP / fastai parity. Per-head `scale` (1 to 1/8; ceil
    sizes; no upsampling when the stride matches). Target downsampling and coordinate helpers,
    alignment, export and ONNX tests. Bench: scaled heads are evaluated at input resolution, and
    runs record FLOPs, throughput and peak memory (E006 done).
13. ✅ **`input_cross`** for `unet` (fastai's full-resolution input stage, opt-in; E012).

## Phase 5: research and optimisation 🔄

Done (see Phase 3 for results and `experiments/` for write-ups): E001b, E002–E006, E009–E012,
E011b, E014, E007 (mixed: frozen encoder BN +2.6 val on Sen1Floods11 at batch 2, GroupNorm decoder +0.5 on Pet; neither passes on both). E013 (negative), E008 (all variants safe; found and fixed an int8 bug on CPUs without VNNI). E007b (frozen encoder BN is a real small-batch fix on both datasets; accumulation alone is not). E015 (native resolution adds nothing but halves grouped-model FLOPs; splitting bands into groups helps +0.9 to +1.3; found the shared-encoder BN bug, fixed). Running: E016 (PASTIS, being prepared). Remaining candidates; a budget is set before each one starts:

| Id | Question | Budget |
| --- | --- | --- |
| — | `input_cross` default: only if E013 Q3 replicates *and* SMP parity is given up (currently a design commitment, so not planned) | — |

Stopping rule: an idea is kept only if it beats its baseline by more than 2 seed standard
deviations on at least two datasets, at no more than a small, stated cost in latency or memory.
Negative results are written up in `experiments/` like positive ones.

## Phase 6: release preparation 🔄

* ✅ Runnable examples (`examples/`): single input, multispectral, multi-modal, multi-task,
  instances/points, export, quantisation. Each runs on CPU in seconds; the slow tier runs them all.
* ✅ Docs site (MkDocs Material, `scripts/build_docs.py` → `build/site`): home, design docs, examples and every experiment write-up. Built locally and ready for **Read the Docs** (`.readthedocs.yaml`; the RTD-style pip build is tested). It goes live when the project is imported on RTD at release: RTD Community needs a public repo.
* ⏳ README status line (currently "pre-alpha").
* ✅ PyPI packaging for `manyseg` and `manyseg-bench` independently: metadata, licence and
  `py.typed` in each wheel, PyPI READMEs, `twine check` clean, and the `manyseg` wheel installs and
  runs alone (CI `package` job on PRs and nightly). Both names are free on PyPI. **Not
  published**: that needs a decision on version and timing. ✅ Tested version range: the floor is **torch 2.9 + timm 1.0.20**, bisected locally. On torch 2.4–2.8 most dynamic export and compile tests fail, because `torch.export` lacks deferred runtime asserts before 2.9. On timm 1.0.12, CAFormer dynamic export fails. Swin dynamic fullgraph compile needs torch ≥ 2.11 (xfail below). A nightly CI leg on the floor runs the unit tier plus the export and compile tests.
* ✅ Unsupported backbone/size combinations: fixed-size architectures are detected at construction (meta-device probe), smaller inputs are padded, and larger ones get a clear `ValueError` naming the `img_size` fix, instead of timm shape errors.

## Known risks

| Risk | Mitigation |
| --- | --- |
| Swin dynamic export hangs inside timm | Probe in a subprocess with a timeout. Document it. Offer fixed-size export. |
| The configuration surface grows with fusion × decoder × head combinations | Keep composition over configuration. `create_model` only parses into modules, and new options need an experiment or a test. |
| Benchmark results on a shared GPU | Contention is recorded with each result. Timing claims are re-measured on an idle GPU. |
| Leakage in remote-sensing splits | Official geographic splits, or grid-cell assignment. Document the protocol for each dataset. |
| Upstream API drift (timm, SMP, fastai) | Pin tested versions in results; network and slow tests run nightly. |
