# Roadmap

Status as of 2026-11-05. ✅ done · 🔄 in progress · ⏳ planned.

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
* ✅ E011c (softplus density head, OAM-TCD, 3 seeds; local 4090): no worse than the raw head (test count MAE 19.82 vs 19.33, within the 1.38 margin; signed bias +0.3 vs −3.3 trees per tile). `density_head()` now defaults to `activation="softplus"`; `activation=None` keeps the raw head.
* ✅ E006 (decoder Pareto; Setonix): unet `out_stride` 2 is free (no mIoU or boundary-F1 loss, 11–18% less memory); fastai_unet's Sen1Floods11 lead comes from its full-resolution stages.
* ✅ E012 (unet `input_cross`; Setonix): positive. It recovers 81–83% of fastai_unet's Sen1Floods11 gain at +14% FLOPs (fastai: 4×), with no effect on Pet. Opt-in; the default stays SMP-identical.
* ✅ Execution on Pawsey Setonix (MI250X) via Slurm arrays: `scripts/setonix/` (see its README for the container, offline weights, MIOpen and Lustre gotchas).
* ✅ Datasets prepared: AI4Boundaries ortho (13 GB cache), OAM-TCD (CC-BY, 1024², ~22 GB cache).
* ✅ CloudSEN12+ high (clouds/shadows, S2 L1C; CC0 labels + Copernicus Sentinel notice): `cloudsen12` (S2), `cloudsen12_s2_aux` (+ static S1GBM backscatter + DEM) and `cloudsen12_meta` (+ lat/lon, date, sun angles) on ROI-disjoint splits; 2048² samples tiled to 512².
* ✅ E013 (CloudSEN12+; Setonix): negative on all three questions. S1GBM/DEM layers, metadata (FiLM, lat/lon on unseen locations) and `input_cross` give no gain; `input_cross` stays a targeted option for tasks whose labels follow per-pixel input values. 5,126 Pawsey units (5,318 with a timed-out re-run).
* ✅ E014 (profiling sweep on an idle 4090): bf16 1.61× over fp32; compile 1.46× more for inference in bf16 (10/10 pairs), only 1.17× for training; cold compile 8–15 s (inference) / 28–54 s (train) per shape.
* ✅ CI workflow (Linux unit tests on pushes to main, skipped for docs-only changes, and on PRs; macOS and Windows on PRs and nightly, where GitHub's macOS runners skip the MPS device tests; slow tier on push to main and nightly).

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
6. ✅ **Export helper** `manyseg.export.export_model`. ✅ Quantisation entry points `manyseg.quantize` (`manyseg[quant]`, torchao): PT2E static int8 with dynamic H/W, int8 weight-only / dynamic, size and agreement reports. ✅ `manyseg.save` and `manyseg.load` keep quantised models and their metadata (`.safetensors` for weight-only, `.pt2` for PT2E; save format 2, marked by a `quantization` entry).
7. ✅ DDP verified (CPU gloo). FSDP2 (`shard_model`) was removed before 0.1.0: only ever tested on CPU with no known user, and ManySeg's models fit on one GPU. Bring it back if someone needs it and it can be tested on real GPUs.
8. ✅ **Instance heads** (`manyseg.instances`): boundary + distance presets, torch targets
   (exact EDT), seeded growth or watershed post-processing, with round-trip tests.
9. ✅ **Point and density heads** (`manyseg.points`): heatmap + offset and density presets
   (softplus density by default since E011c), targets, NMS peaks, counting, offset grouping and
   panoptic labels, with round-trip tests.
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
14. ⏳ **Presets**, removed before 0.1.0 and planned for a later version. The E018 names didn't
    match what people choose by: some used more training memory than the default, the CPU timings
    they shipped had no record (E018's own CPU run was noisy and used another setup), and
    "accurate" was a small encoder. Rebuild them around goals
    (low VRAM, fast GPU, fast CPU), measured on current code.
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
21. ✅ **Random-configuration fuzzer** (`packages/manyseg/tests/fuzz`): thousands of `create_model` configurations
    against shape, gradient, determinism, round-trip and export invariants; the bugs found and
    fixed have regression tests in `test_fuzz_regressions.py` (see `testing_strategy.md`), including modules without gradient that broke
    DDP's default mode.

## Phase 5: research and optimisation 🔄

Done (see Phase 3 for results and `experiments/` for write-ups): E063 (RGB weights vs scratch on Sentinel-2 inputs, ResNet-50: no general win; against the better scratch recipe +4.2 points on CloudSEN12+, +1.0 on PASTIS, −0.6 on Sen1Floods11, −1.0 on BigEarthNet; level with 12–252 Sen1Floods11 chips). E024 (remote-sensing weights with their own normalisation, ResNet-50: SSL4EO DINO and MoCo beat ImageNet on Sen1Floods11, BigEarthNet and PASTIS, DINO by +2.1 / +4.1 / +2.1; recommended. Satlas only on Sen1Floods11, not recommended). E001b, E002–E006, E009–E012,
E011b, E014, E007 (mixed: frozen encoder BN +2.6 val on Sen1Floods11 at batch 2, GroupNorm decoder +0.5 on Pet; neither passes on both). E013 (negative), E008 (all variants safe; found and fixed an int8 bug on CPUs without VNNI). E007b (frozen encoder BN is a real small-batch fix on both datasets; accumulation alone is not). E015 (native resolution adds nothing but halves grouped-model FLOPs; splitting bands into groups helps +0.9 to +1.3; found the shared-encoder BN bug, fixed). E016 (PASTIS: day of year adds nothing even for crop phenology; band grouping does not replicate). E017 (CloudSEN12+: band groups +0.37 val, under the 0.76 bar; grouping stays a single-dataset result). E023 (semantic change on SECOND: per-input land-cover heads beat fused (+1.5 Score); shared = separate, so share; Siamese +7.0 over early fusion). E021 (time series: several dates help a lot; stacking dates as channels beat the time-series input on PASTIS; on AI4Boundaries attention beat stacking and mean fell just short of the bar; attention ≈ mean). E022 (binary change: Siamese with diff / concat_diff beats early fusion on OSCD and LEVIR-CD, Siamese concat on LEVIR-CD only; no single op wins on both). E020 (BigEarthNet v2: parity with timm for classification and regression; band groups, S1 and metadata do not help; a segmentation head on the reference maps helps the classifier, +2.1 val / +2.3 test macro mAP; post hoc it looks like regularisation). E019/E019b (ManySeg image heads at val parity with timm classifiers in 6/6 cells; test −0.43, largest on ConvNeXt). E018 (Pareto front: four PVTv2-b0 configs are on the front of both datasets (they shipped as presets until removed before 0.1.0); three of them (all but the SegFormer-head `fast`, which is 0.34 lower on Pet) dominate the ResNet-34 U-Net default). E025 (training speed against SMP, from a downstream report: the nearest-upsampling gather's scatter-add backward made U-Net steps 2–5% slower eager and up to 98% slower compiled; replaced where the values are the same, now at parity compiled and 18–25% faster eager; FPN / fastai U-Net 6–14% faster compiled). E026 (performance audit, values unchanged: float32 activations under bf16 autocast removed from fastai_unet, `input_cross`, GroupNorm, UPerNet and DeepLab decoders (−7 to −27% memory); SegFormer fuse per level (−26 to −38% memory); contiguous cross-attention tokens (−50% time, −77% memory); opt-in encoder gradient checkpointing; sync-free bench training loop (−24% step time for the PVTv2 presets); CloudSEN12+ training is disk-bound). E027 (round 2: static compiled evaluation with a padded last batch +12–27%; host arrays of device-resident datasets freed, −2.8 GB RAM; bilinear U-Net and concat-fusion resize dtype −4% time; evaluator syncs, object-metric copies and the ViT neck: no gain). E011c (softplus density head no worse than the raw head on OAM-TCD; now the default). E028–E054: vector outputs, below. No candidates remain open: making `input_cross` the default is closed, since E013 Q3 did not replicate it and SMP parity is a design commitment. A budget is set before any new one starts.

Stopping rule: changing a default, or recommending an option over it, needs a gain of more than
2 seed standard deviations on at least two datasets, at no more than a small, stated cost in
latency or memory. An opt-in option needs a test or an experiment behind it, and the README's
"What the evidence says" table states how strong its evidence is, including "none". Negative
results are written up in `experiments/` like positive ones.

### Future work: stronger evidence

* **Find datasets where fancier options help.** Stacking bands and the plain U-Net have matched or
  nearly matched the alternatives so far, on few datasets. Look for tasks where fancier decoders
  and fusion should matter: thin or fine structures, boundaries, inputs at very different
  resolutions, or a sensor that's missing in part of the data. Include what has never been tested
  for accuracy: U-Net++ and the `add`, `attention` and `cross_attention` fusion ops.
* **Test the vector target encoders and decoders further.** Each decoder rests on one dataset and
  1-3 seeds. Run them on more datasets and regions, with more seeds, and check how far the tuned
  defaults carry over. The stopping rule above isn't met for any of them yet.

### Vector outputs (roads, lines, footprints, fields, 3D buildings) 🔄

Design after a Codex review: one shared *output schema* (attributed graphs, polygons, roof
meshes; GeoJSON / GeoPackage export) and shared geometry utilities, but separate dense encodings
and decoders per family: open networks (roads, lines), closed polygons (footprints, fields), roof
surfaces. Everything stays dense heads + decoding outside `forward`, like `manyseg.instances`.
Roads are not forced planar (overpasses keep no junction). Learned second stages (SAM-Road / HEAT
style edge classifiers) would be a separate optional module, never a head. Oracles score
synthetically corrupted perfect targets (levels L0–L3), not trained models.

| Stage | Question | Status |
| --- | --- | --- |
| E028 | Oracle: perfect OSM vectors → candidate dense targets → controlled corruption → decode → score. Roads: the Sat2Graph-style graph tensor has the best ceiling (only one to keep overpasses: 150/160 vs 0/160; APLS 0.98 vs 0.89 on Murdoch) but missed vertices and occlusion break it faster than a mask, so it is dropped; a junction heatmap adds nothing. Footprints: the same graph tensor → faces is viable and gives the best polygons to level 2 (C-IoU 0.72–0.75 vs 0.53 for mask contours), failing all-or-nothing; a corner heatmap improves shape but loses detections, so it is dropped. One learned representation for both tasks is not supported | ✅ |
| E028b | Oracle: graph footprints (B2) with a per-building fallback to mask polygons (B0). No gain over B2 at L1 and a loss at L2 (C-IoU × F1 0.25 / 0.34 vs 0.35 / 0.48), the levels the rule judged; not adopted. It is ahead at L3, where both are near zero (Murdoch 0.027 vs 0.007) | ✅ |
| E029 | City-Scale converted to pickle-free graphs with overpass points (`prepare.py`; the data E044 trains on; data preparation only, no experiment or decision) | ✅ |
| E030 | Trained footprints on WHU aerial: B0 vs B2. One seed (the others cancelled): B2 C-IoU 0.61 / detection F1 0.75 vs B0 0.71 / 0.85; a two-model fallback (E030b) reaches 0.64 / 0.81. The graph tensor was removed from the library; its code is in `experiments/E030_whu_polygons/` | ✅ |
| E031 | LoD2 oracle (3DBAG): roof-edge graph (A1) vs plane segmentation of the height model (A0). The graph's faces are far better (face F1 0.99 vs 0.48–0.62 clean), but buildings whose graph does not close get no roof, so heights fail (L2 RMSE 4.0–4.3 vs 1.9–2.0 m): not viable. E031b, graph with a plane-segmentation fallback, beats A0 at L1–L2 (L1 face F1 0.80–0.81 vs 0.43–0.55) within the height margin: a trained LoD2 experiment was worth running | ✅ |
| E032–E034 | Training fixes for B2 on WHU (destination loss, B2e; snap head, B2s; wall-tolerant vertex loss, B2t; the planned B2w and B2d were dropped unrun), seed-0 screens: val C-IoU × F1 0.411 / 0.423 / 0.411 vs B2 0.440. All screened out | ✅ |
| E035 / E035b | Oracle: B3 = B0's instance heads + corner heatmap + wall-direction field, straight walls between corners. E035 traced mask dents (N-ratio 2.0–4.7 at L1–L2). The corner-trusting decoder (E035b) keeps B0's detection F1 and beats B0 at every corrupted level (L1 C-IoU × F1 0.565 / 0.724 vs 0.523 / 0.650), below B2 in this oracle | ✅ |
| E036 | B3 trained on WHU, seed-0 screen against B0 on C-IoU × F1: val 0.482 vs 0.628, no advance. Post hoc: B3 loses on C-IoU's vertex count against WHU's staircase outlines while its matched IoU and PoLiS are the best | ✅ |
| E037 | Oracle: dense pixel graph (8-neighbour links, P8) and mask-supported links (R3). Roads killed at L0 (junction F1 0.967–0.983 < 0.995; one node per pixel turns crossings into junctions); every arm killed by ±0.25 px shifts (link disagreement 0.34–0.46, rule 0.05) | ✅ |
| E038 | B3 vs B0 vs B0 + regulariser on WHU, scored by boundary F against straight-line (Douglas-Peucker) outlines, 2 seeds. B3 +0.018 / +0.026 over B0, detection F1 −0.021 / −0.011: B3 is the footprint decoder (`polygons_from_corners`) | ✅ |
| E039 | Oracle: unified road heads R4 (mask, junction / end / crossing keypoints, two-mode direction). Keeps 100% / 76% of crossings at L0 / L1 (mask R0: 0% / 2%) at the mask's APLS (L1 0.569 vs 0.572), but junction F1 only +0.032 at L1 (rule +0.10): not viable by the rule | ✅ |
| E040 | Oracle: Frame Field Learning's frame field as B3's direction head and corner source. Below the corner heatmap at L0–L1 (C-IoU × F1 0.909 / 0.601 vs 0.949 / 0.623); the corner heatmap stays | ✅ |
| E041 | Oracle: edge-scoring road decoder (R5). Parked after four smoke-tile iterations (16–50 components at L0, GT 2: a graph-construction problem); no result | parked |
| E042 | Oracle: R4 + min-cost-path gap healing (R6). Junction F1 −0.185 and APLS −0.019 vs R0 at L1: not viable, worse than R4 | ✅ |
| E043 / E043b | Oracle: roof mesh from graph faces + one plane per face. E043's height-RMSE rule could not move (shared faces dominate it). E043b, orientation primary: robust height-only plane fit and J2 joining adopted (false steps −28 to −41%); slope rasters not adopted (normal error −2.9 / −4.4° at L2 / L3, false steps over the 10% allowance) | ✅ |
| E044 / E044b | R4 trained on City-Scale vs the R0 mask skeleton. Seed 0: junction F1 +0.024 (rule +0.05), not viable as a junction fix. E044b, seeds 1–2, safe-default rule: APLS +0.003 / +0.005, junction F1 +0.027 / +0.027, crossings kept +0.22 / +0.25. R4 (surface mode) is the road decoder (`graph_from_road_heads`) | ✅ |
| E045 | Fields oracle (FTW Netherlands / Denmark): B3 vs B0. B3's boundary F is 0.04 below B0's at L1 on both sets (rule 0.01): fields keep B0. Causes: straight walls on curved field edges, spikes to touching neighbours' corners | ✅ |
| E046 | Fields trained (FTW Netherlands, seed 0): B3 boundary F 0.565 vs B0 0.592, detection F1 +0.011: fields keep B0 (`polygons_from_labels`) | ✅ |
| E047 / E047b | Trained LoD2 roof faces from RGB + AHN nDSM (3DBAG targets, Netherlands). E047b (150 epochs, decides): faces from the mask outline (F0) face F1 0.580 / 0.625 vs plane segmentation 0.284 / 0.314, height RMSE −0.46 / −0.88 m: viable. Corners + lines (F3) not preferred (face F1 −0.008 / −0.036; corner F1 +0.14, C-IoU 0.64 vs 0.15–0.21). E047 (40 epochs) not decision-grade | ✅ |
| E048 / E048b | Joining trained F3 faces into a mesh. Vertex snapping: seam closure +0.11 / +0.12 (rule +0.20), not adopted. Planar joining (snap to neighbours, node, re-polygonise): seam closure 0.38 → 0.73 / 0.71 (3DBAG's own faces 0.79), vertex z error −0.07 / −0.04 m: adopted (`planar_faces`) | ✅ |
| E049 / E049b | LoD2 from RGB only, with predicted height (and slope) heads. Face F1 −0.38 / −0.28 and height RMSE +2.88 / +1.82 m vs RGB + nDSM: not viable. A rescaled slope head failed its val gate (correlation 0.42 / 0.37 < 0.5). Line closed: roofs need an nDSM | ✅ |
| E050 / E050b | Stitching tiled predictions for vector decoding (blend vs cutlines), scored by agreement with a single-pass decode. Roads: any cut removes the seam effect (junction F1 0.971–0.972 vs blend 0.954); min-cost and object-aware seams add nothing over the midline. Roofs: no merge rule helps at 512 / 128 px tiles; the gap was context: at 1024 / 256 all-face agreement is 0.979 (blend) / 0.987 (midline). Ships as the midline cut with ≥ 1024 / 256 tiles; the cut's gain on roofs was not pre-registered | ✅ |
| E051 | End to end through ManyPred (`merge="cut"`, `BlockPostprocess`, `block_decoder`): the block decode equals the whole-scene decode on a 1 km² roof scene (5537 faces, exact) and a 2048 px road scene (identical graph) | ✅ |
| E052 | Absolute vs normalised distance target on FTW fields, 2 seeds (per-tile normalisation changes 3–34 fields at seams in the end-to-end test). Absolute: boundary F +0.008, detection F1 +0.008: it is the fields default; roof faces untested | ✅ |
| E053 | Minimum seed size 8 vs 0 px in block decoding (a narrow part cut off by boundary bands got its own seed: an extra 17–25 px face at hip-roof tips, an extra 16 px field at field corners), on trained checkpoints. Test, 0 → 8: roofs face F1 +0.041, height RMSE 1.35 → 1.28 m (E047b, 1 seed); footprints detection F1 +0.024, boundary F +0.004 (E036, 2 seeds); fields detection F1 +0.008, boundary F −0.0002 (E052, 2 seeds). Passes for all three: `min_seed=8` is the `block_decoder` default (`instances_from_outputs` keeps 0) | ✅ |
| E054 | Courtyards in corner-decoded footprints, on E036 (2 seeds). Test, filled → kept: boundary F −0.0004, detection F1 0.000; passes, `holes=True` is the footprint default (roof faces stay outer rings) | ✅ |
| — | Shared schema and utilities: done (target encoders, `block_decoder` / `decode_scene`, `SceneRecords`, GeoJSON / GeoPackage). One model with task-specific heads vs separate models: not tested | ⏳ |
| — | LoD1: footprint + ground / eave / roof height; roof-to-footprint offset for off-nadir imagery | ⏳ |
| — | LoD2 remainder: watertight solids and CityJSON export (0.1.0 writes faces and a 3-D roof mesh) | ⏳ |

Ships in 0.1.0, marked experimental (names and defaults may change before 1.0):

* Footprints: B3 corners + lines (E038). Fields: B0 mask outlines (E046), with
  `distance="absolute"` (E052).
* Roads: R4 unified heads, surface mode (E044b). Road ends within 5 px of the scene edge count
  as at the border (was 3 px), so two cut ends are not bridged by a fake road along the edge
  (found by the end-to-end test).
* LoD2 roofs: trained faces (F3) + planar joining (E047b, E048b), robust plane per face + J2
  mesh (E043b). An nDSM input is required (E049).
* Footprints, fields and roofs decode with seeds of at least 8 px (E053).
* Known limitation: under the fields' 3 px seed threshold, a narrow strip of a field (under
  about 8 px, e.g. beside a hole) can split into pieces, also in a single pass.
* Known limitation: the tip of a hip-end roof face can be flooded from a neighbouring face's
  seed, so the face loses its tip (face IoU down to 0.75 on synthetic scenes).
* Large scenes: ManySeg's block decoders run in ManyPred's `merge="cut"` stitching with block
  post-processing, or in `decode_scene` on a stitched raster; block and whole-scene decodes
  were equal on E051's two scenes, though not on noise-like outputs (limitations in
  `api_design.md`). Tiles of at least 1024 px with 256 px overlap (E050b).
* Removed: the graph tensor (B2 / R2; E028, E030), and clip-edge corner suppression
  (`corner_targets(tile_edges="ignore")`), which changed nothing measurable in the end-to-end
  test (numbers in the CHANGELOG).

## Phase 6: release preparation 🔄

* ✅ Runnable examples (`examples/`): single input, multispectral, multi-modal, multi-task,
  instances/points, export, quantisation, classification / regression, fastai, plain PyTorch,
  Lightning, Accelerate, TorchGeo, vector outputs (footprints, fields, roads, roofs, points;
  14–18), plus walkthroughs (metadata conditioning, instance fields, point counting, vector
  presets). Each runs on CPU (seconds; the training walkthroughs 10–20 s); the slow tier runs them all.
* ✅ Notebooks (`notebooks/`) with real data and saved figures: flood mapping on Sen1Floods11, optical + radar with several heads, one per vector output (footprints, fields, roads, roofs, points). The nightly tier runs them in smoke mode. Add Colab links when the repo goes public.
* ✅ Docs site (MkDocs Material, `scripts/build_docs.py` → `build/site`): home, design docs, changelog, contributing guide, model builder, examples, notebooks, API reference and every experiment write-up. Built locally and ready for **Read the Docs** (`.readthedocs.yaml`; the RTD-style pip build is tested). It goes live when the project is imported on RTD at release: RTD Community needs a public repo.
* ✅ API review before 0.1.0 (Codex as a new user): config vocabulary (`in_channels` /
  `out_channels` / `num_features`, head type follows scope, `decoder` as a dict), internal
  hooks private, keyword-only tuning arguments, `pack_metadata`, save format 2 with migration,
  weights citations; `CHANGELOG.md`.
* ✅ Python 3.13 and 3.14 pass every tier locally; 3.15rc1 everything except `torch.compile` (not yet
  supported by PyTorch). CI: Linux 3.11 / 3.14, Windows on PRs and nightly.
* ✅ GPU fuzzer (CUDA vs CPU, bf16, `torch.compile`): found a PyTorch Inductor bug in dynamic
  nearest upsampling, worked around (`docs/upstream/`).
* ✅ README status line and classifiers: alpha.
* ✅ Packaging: `manyseg` and `manyseg-bench` build as independent wheels (metadata, licence and
  `py.typed` in each), and the `manyseg` wheel installs and runs alone (CI `package` job on PRs
  and nightly). `manyseg` is ready for PyPI (PyPI README, `twine check` clean); it is not yet
  published, and the release is 0.1.0. `manyseg-bench` stays in the repository: its results depend on
  `experiments/` and record the git commit that produced them, which a wheel install lacks. ✅ Tested version range: the floor is **torch 2.10 + timm 1.0.20**. Bisected locally, 2.9 + 1.0.20 is the lowest that passes the export and compile tests, but torch 2.5–2.9 has an MPS view/stride bug (timm EdgeNeXt; see OmniCloudMask issue #45), so the floor was raised to 2.10 before the first release. On torch 2.4–2.8 most dynamic export and compile tests fail, because `torch.export` lacks deferred runtime asserts before 2.9. On timm 1.0.12, CAFormer dynamic export fails. Swin dynamic fullgraph compile fails on 2.9 and passes on 2.10. A nightly CI leg on the floor runs the unit tier plus the export and compile tests.
* ✅ Unsupported backbone/size combinations: fixed-size architectures are detected at construction (meta-device probe), smaller inputs are padded, and larger ones get a clear `ValueError` naming the `img_size` fix, instead of timm shape errors.
* ✅ Hugging Face Hub (`manyseg.hub`): `push_to_hub` uploads a `manyseg.save` file with a
  generated model card; `load` / `read_metadata` take `"hf-hub:org/name[@revision]"`.
* ✅ Vector outputs ship in 0.1.0, marked experimental: guide in `api_design.md`, examples 14–18,
  CHANGELOG entry, one end-to-end test per task, and a full-pipeline test through ManyPred
  (`packages/manyseg/tests/integration/test_vector_pipeline.py`: vectors in a CRS → per-tile targets → stitching
  → block decoding → GeoPackage). The graph tensor was moved out of the library (E030).
* ✅ Fields default to `distance="absolute"` (E052).
* ✅ Clip-edge corner suppression (`tile_edges="ignore"`) removed from the API; its negative
  result is in the CHANGELOG.
* ✅ Seeds in narrow parts of an instance (extra faces at hip-roof tips, extra fields at field
  corners): the extra objects are gone with a minimum seed size of 8 px (E053). A hip-roof tip
  can still go to a neighbouring face, and a field's narrow strip can split under the 3 px seed
  threshold; both are documented limitations.
* ✅ API reference (one page per public module, from the docstrings); ManyPred section in
  the README.
* ✅ `manyseg.losses`: the bench losses moved into the library (re-exported by the bench), plus
  `multiclass_dice` and `multi_head_loss`.
* ✅ ManyPred's vector-blocks branch (`merge="cut"`, block post-processing) is on ManyPred's
  main (2026-11-02).
* ⏳ Make the repositories public, so Read the Docs can build them.
* ⏳ PyPI release of 0.1.0 (done by the user).

## Known risks

| Risk | Mitigation |
| --- | --- |
| Swin dynamic export fails inside timm (it used to hang) | Probe in a subprocess with a timeout. Document it. Offer fixed-size export. |
| The configuration surface grows with fusion × decoder × head combinations | Keep composition over configuration. `create_model` only parses into modules, and new options need an experiment or a test. |
| Benchmark results on a shared GPU | Contention is recorded with each result. Timing claims are re-measured on an idle GPU. |
| Leakage in remote-sensing splits | Official geographic splits, or grid-cell assignment. Document the protocol for each dataset. |
| Vector decoders chosen on one to three seeds, one dataset per task, and partly from oracles | Marked experimental (names and defaults may change before 1.0); each decision rule and its seeds are in `experiments/`. |
| Roofs need an nDSM, and were trained and tested only in the Netherlands, where the 3DBAG targets and the AHN height input share a source (E047) | Stated in the vector guide (`api_design.md`); RGB-only roofs were tested and failed (E049). |
| Large-scene vector decoding through ManyPred needs ManyPred features on its main but not yet released (`merge="cut"`, block post-processing) | Release ManyPred first; `decode_scene` works on any stitched raster without ManyPred. |
| Upstream API drift (timm, SMP, fastai) | Pin tested versions in results; slow tests and a torch/timm floor leg run nightly; network tests run locally. |
