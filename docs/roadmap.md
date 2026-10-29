# Roadmap

Status as of 2026-10-18, 09:00 AWST. ✅ done · 🔄 in progress · ⏳ planned.

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

## Phase 3: independent benchmark ✅

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
* ✅ CI workflow (Linux unit tests on every push; macOS/MPS on PRs and nightly; slow tier on push to main and nightly).

## Phase 4: expanded architecture support ✅

1. ✅ **Capability probe** (`manyseg.backbones.capabilities`) and generated
   `docs/backbone_compatibility.md` (46 timm models; 41 any-size; 38 export dynamically).
2. ✅ **Transformer backbones** via meta-device introspection: hierarchical models (NHWC
   normalised), plain ViT/DeiT/EVA/DINOv2 through `ViTPyramidNeck` (any patch size, incl. 14),
   and fixed-size models with `img_size` plus clear errors.
3. ✅ **Decoders**: U-Net, U-Net++, FPN, DeepLabV3+, UPerNet, SegFormer-MLP, fastai-style
   Dynamic U-Net (parameter- and FLOP-identical to fastai).
4. ✅ **Multi-input fusion**: early, intermediate and late; seven ops; shared encoders;
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
14. ✅ **Presets** (`manyseg.presets`): `fast` / `compact` / `balanced` / `accurate`, the E018
    accuracy/latency front, opt-in with their measured numbers.
15. ✅ **Image classification and regression**: models with only image heads build no
    decoder; FiLM metadata on the pooled feature; bench tasks (classification, multi-label,
    regression, segmentation + multi-label), a plain-timm provider, EuroSAT, Pet breeds and
    BigEarthNet v2 datasets (E019 parity with timm; E020: an auxiliary pixel head helped the
    classifier, +2.3 test mAP).
16. ✅ **Time-series inputs** (`"kind": "timeseries"`): one encoder per date, masked mean / max /
    attention / concat / diff aggregation per level, masks and acquisition dates, dynamic T in
    export (E021).
17. ✅ **Change detection**: `diff` / `concat_diff` fusion ops, per-input heads (`"input"`) and
    shared per-input heads (`"share"`) for semantic change (E022, E023).
18. ✅ **Feature / embedding output**: `features()`, `embed()` (encoder or decoder, per pixel or
    pooled), exportable `embedder()`; `select_heads()` for deployment.
19. ✅ **Multi-label pixel segmentation** (sigmoid per class; bench task `multilabel_seg`).
20. ✅ **Remote-sensing pretrained weights** (`manyseg.weights`): SSL4EO-S12 MoCo / DINO and
    SatlasPretrain into timm ResNet / ViT, hash-checked, with band order, units and the
    pretraining normalisation recorded; the bench applies each weights' own normalisation and
    maps dataset bands onto theirs (E024).
21. ✅ **Random-configuration fuzzer** (`tests/fuzz`): thousands of `create_model` configurations
    against shape, gradient, determinism, round-trip and export invariants; about 25 bugs
    found and fixed (see `testing_strategy.md`), including modules without gradient that broke
    DDP's default mode.

## Phase 5: research and optimisation 🔄

Done (see Phase 3 for results and `experiments/` for write-ups): E024 (remote-sensing weights with their own normalisation, ResNet-50: SSL4EO DINO and MoCo beat ImageNet on Sen1Floods11, BigEarthNet and PASTIS, DINO by +2.1 / +4.1 / +2.1; recommended. Satlas only on Sen1Floods11, not recommended). E001b, E002–E006, E009–E012,
E011b, E014, E007 (mixed: frozen encoder BN +2.6 val on Sen1Floods11 at batch 2, GroupNorm decoder +0.5 on Pet; neither passes on both). E013 (negative), E008 (all variants safe; found and fixed an int8 bug on CPUs without VNNI). E007b (frozen encoder BN is a real small-batch fix on both datasets; accumulation alone is not). E015 (native resolution adds nothing but halves grouped-model FLOPs; splitting bands into groups helps +0.9 to +1.3; found the shared-encoder BN bug, fixed). E016 (PASTIS: day of year adds nothing even for crop phenology; band grouping does not replicate). E017 (CloudSEN12+: band groups +0.37 val, under the 0.76 bar; grouping stays a single-dataset result). E023 (semantic change on SECOND: per-input land-cover heads beat fused (+1.5 Score); shared = separate, so share; Siamese +7.0 over early fusion). E021 (time series: several dates help a lot; stacking dates as channels beat the time-series input on PASTIS, the reverse on AI4Boundaries; attention ≈ mean). E022 (binary change: Siamese with diff / concat_diff beats early fusion on OSCD and LEVIR-CD, Siamese concat on LEVIR-CD only; no single op wins on both). E020 (BigEarthNet v2: parity with timm for classification and regression; band groups, S1 and metadata do not help; a segmentation head on the reference maps helps the classifier, +2.1 val / +2.3 test macro mAP; post hoc it looks like regularisation). E019/E019b (ManySeg image heads at val parity with timm classifiers in 6/6 cells; test −0.43, mostly ConvNeXt). E018 (Pareto front: four PVTv2-b0 configs are on the front of both datasets and ship as opt-in `manyseg.presets`; three of them (all but the SegFormer-head `fast`, which is 0.34 lower on Pet) dominate the ResNet-34 U-Net default). E025 (training speed against SMP, from a downstream report: the nearest-upsampling gather's scatter-add backward made U-Net steps 2–5% slower eager and up to 98% slower compiled; replaced where the values are the same, now at parity compiled and 18–25% faster eager; FPN / fastai U-Net 6–14% faster compiled). E026 (performance audit, values unchanged: float32 activations under bf16 autocast removed from fastai_unet, `input_cross`, GroupNorm, UPerNet and DeepLab decoders (−19 to −27% memory); SegFormer fuse per level (−26 to −38% memory); contiguous cross-attention tokens (−50% time, −77% memory); opt-in encoder gradient checkpointing; sync-free bench training loop (−24% step time for the PVTv2 presets); CloudSEN12+ training is disk-bound). E027 (round 2: static compiled evaluation with a padded last batch +12–27%; host arrays of device-resident datasets freed, −2.8 GB RAM; bilinear U-Net and concat-fusion resize dtype −4% time; evaluator syncs, object-metric copies and the ViT neck: no gain). Remaining candidates; a budget is set before each one starts:

| Id | Question | Budget |
| --- | --- | --- |
| — | `input_cross` default: only if E013 Q3 replicates *and* SMP parity is given up (currently a design commitment, so not planned) | — |
| E011c | Softplus density head as the default: rerun E011's density arm (OAM-TCD) with `density_head(activation="softplus")`, pre-registered, same protocol and seeds as E011. Downstream drone data (skw-yolo): softplus cut density count MAE 10.1 → 2.9 and ReLU died (removed); peak counting still won there (1.0). Change the default only if softplus is no worse than the plain head on OAM-TCD | E011's density-arm budget |

Stopping rule: an idea is kept only if it beats its baseline by more than 2 seed standard
deviations on at least two datasets, at no more than a small, stated cost in latency or memory.
Negative results are written up in `experiments/` like positive ones.

### Vector outputs (roads, lines, footprints, 3D buildings) 🔄

Design after a Codex review: one shared *output schema* (attributed graphs, polygons, later
meshes; GeoJSON / CityJSON export) and shared geometry utilities, but separate dense encodings
and decoders per family: open networks (roads, lines), closed polygons (footprints), roof
surfaces. Everything stays dense heads + decoding outside `forward`, like `manyseg.instances`.
Roads are not forced planar (overpasses keep no junction). Learned second stages (SAM-Road / HEAT
style edge classifiers) would be a separate optional module, never a head.

| Stage | Question | Status |
| --- | --- | --- |
| E028 | Oracle: perfect OSM vectors → candidate dense targets → controlled corruption → decode → score. Roads: the Sat2Graph-style graph tensor has the best ceiling (only one to keep overpasses: 150/160 vs 0/160; APLS 0.98 vs 0.89 on Murdoch) but missed vertices and occlusion break it faster than a mask, so it is dropped; a junction heatmap adds nothing. Footprints: the same graph tensor → faces is viable and gives the best polygons to level 2 (C-IoU 0.72–0.75 vs 0.53 for mask contours), failing all-or-nothing; a corner heatmap improves shape but loses detections, so it is dropped. One learned representation for both tasks is not supported | ✅ |
| 1 | Trained roads on City-Scale: mask baseline only (E028 dropped R1 and R2); lower priority. A mask + graph-tensor hybrid would need its own pre-registration | ⏳ |
| 2 | Trained footprints on WHU aerial: mask instances (B0) vs graph tensor → faces (B2). Decides whether ManySeg ships graph-tensor polygon heads | ⏳ next |
| 3 | Shared utilities and schema; one model with task-specific heads vs separate models | ⏳ |
| 4 | LoD1: footprint + ground / eave / roof height; roof-to-footprint offset for off-nadir imagery | ⏳ |
| 5 | LoD2 roofs: roof-face segmentation + ridge / edge evidence, planes fitted jointly from DSM / LiDAR, watertight solid, CityJSON. RGB-only heights are a learned prior, reported separately from topology | ⏳ |

## Phase 6: release preparation 🔄

* ✅ Runnable examples (`examples/`): single input, multispectral, multi-modal, multi-task,
  instances/points, export, quantisation, classification / regression, fastai, plus walkthroughs
  (metadata conditioning, instance fields, point counting). Each runs on CPU (seconds; the training walkthroughs 10–20 s); the slow tier runs them all.
* ✅ Docs site (MkDocs Material, `scripts/build_docs.py` → `build/site`): home, design docs, examples and every experiment write-up. Built locally and ready for **Read the Docs** (`.readthedocs.yaml`; the RTD-style pip build is tested). It goes live when the project is imported on RTD at release: RTD Community needs a public repo.
* ✅ API review before 0.1.0 (Codex as a new user): config vocabulary (`in_channels` /
  `out_channels` / `num_features`, head type follows scope, `decoder` as a dict), internal
  hooks private, keyword-only tuning arguments, `pack_metadata`, save format 2 with migration,
  weights citations; `CHANGELOG.md`.
* ✅ Python 3.13 and 3.14 pass every tier; 3.15rc1 everything except `torch.compile` (not yet
  supported by PyTorch). CI: Linux 3.10 / 3.14, Windows on PRs and nightly.
* ✅ GPU fuzzer (CUDA vs CPU, bf16, `torch.compile`): found a PyTorch Inductor bug in dynamic
  nearest upsampling, worked around (`docs/upstream/`).
* ✅ README status line and classifiers: alpha.
* ✅ PyPI packaging for `manyseg` and `manyseg-bench` independently: metadata, licence and
  `py.typed` in each wheel, PyPI READMEs, `twine check` clean, and the `manyseg` wheel installs and
  runs alone (CI `package` job on PRs and nightly). Both names are free on PyPI. **Not
  published**: that needs a decision on version and timing. ✅ Tested version range: the floor is **torch 2.10 + timm 1.0.20**. Bisected locally, 2.9 + 1.0.20 is the lowest that passes the export and compile tests, but torch 2.5–2.9 has an MPS view/stride bug (timm EdgeNeXt; see OmniCloudMask issue #45), so the floor was raised to 2.10 before the first release. On torch 2.4–2.8 most dynamic export and compile tests fail, because `torch.export` lacks deferred runtime asserts before 2.9. On timm 1.0.12, CAFormer dynamic export fails. Swin dynamic fullgraph compile fails on 2.9 and passes on 2.10. A nightly CI leg on the floor runs the unit tier plus the export and compile tests.
* ✅ Unsupported backbone/size combinations: fixed-size architectures are detected at construction (meta-device probe), smaller inputs are padded, and larger ones get a clear `ValueError` naming the `img_size` fix, instead of timm shape errors.

## Known risks

| Risk | Mitigation |
| --- | --- |
| Swin dynamic export fails inside timm (it used to hang) | Probe in a subprocess with a timeout. Document it. Offer fixed-size export. |
| The configuration surface grows with fusion × decoder × head combinations | Keep composition over configuration. `create_model` only parses into modules, and new options need an experiment or a test. |
| Benchmark results on a shared GPU | Contention is recorded with each result. Timing claims are re-measured on an idle GPU. |
| Leakage in remote-sensing splits | Official geographic splits, or grid-cell assignment. Document the protocol for each dataset. |
| Upstream API drift (timm, SMP, fastai) | Pin tested versions in results; slow tests and a torch/timm floor leg run nightly; network tests run locally. |
