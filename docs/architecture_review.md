# Architecture review of existing libraries

> **Written before the implementation.** This is the review that shaped the design, so its plans
> are plans and its numbers date from when it was written. For what ManySeg does today, see
> [API design](api_design.md).

Date: 2026-10-14. Versions read: segmentation-models-pytorch 0.5.0, fastai 2.8.12, timm 1.0.30,
torch 2.14.1, plus shallow clones of mmsegmentation (last commit 2024-03), TerraTorch, TorchGeo
and Hugging Face transformers (all three cloned on 2026-10-13/14). Every measured number
below was produced on this workstation. Commands and scripts are in
`experiments/E000_review_measurements/`.

Licenses: SMP MIT, timm Apache-2.0, fastai Apache-2.0, MMSegmentation Apache-2.0, TerraTorch
Apache-2.0, TorchGeo MIT, transformers Apache-2.0. ManySeg copies no code from these
libraries. Where a design follows one of them, the source is cited here and in the docstring.

---

## 1. fastai DynamicUnet vs SMP Unet

The claim under investigation is that fastai's DynamicUnet often beats SMP's Unet with the
same encoder. Both were built with a resnet34 encoder at 256×256 and compared at source level.

### 1.1 They are not the same size of model

| | fastai DynamicUnet | SMP Unet |
| --- | --- | --- |
| Encoder params | 21.28 M | 21.28 M |
| Decoder params | **19.94 M** | **3.15 M** |
| Decoder widths (deep→shallow) | 512→1024→512 middle block; 512@16, 384@32, 256@64, 96@128; 96@256; ResBlock(99)@256 | 256@16, 128@32, 64@64, 32@128, 16@256 |
| Forward GFLOPs @256² (2×MAC) | **62.8** (decoder ≈53.3, of which the full-res ResBlock is 23.1) | **15.7** (decoder 6.1) |
| fwd+bwd bs 8 (4090) | 61 ms, 2.48 GB | 25 ms, 1.36 GB |

fastai's block width is `up_in_c//2 + skip_c` with no halving (`fastai/vision/models/unet.py:29-31,75`).
It also always adds a 2×conv "middle" block on the bottleneck (`unet.py:69-71`). SMP halves the
width at every stage, starting from 256 (`decoders/unet/model.py:112`). **fastai's decoder has
about 6.3× the parameters and 8.8× the FLOPs.** Any accuracy comparison that does not control
for this is a comparison between model sizes, not between designs.

### 1.2 Structural differences

| Aspect | fastai | SMP |
| --- | --- | --- |
| Upsampling | `PixelShuffle_ICNR` (learned 1×1 conv → shuffle, ICNR init so the output starts as nearest-neighbour) `layers.py:369-392` | `F.interpolate(nearest)` (parameter-free) `decoder.py:50-54` |
| Decoder conv norm | **none** (DynamicUnet `norm_type=None`): Conv(bias) → ReLU | Conv → BN → ReLU after every conv |
| Skip handling | **BN on each skip before concat**, then ReLU on the concat `unet.py:29,44` | raw concat |
| Full-resolution stage | concatenates the **raw input image** (`last_cross`, `MergeLayer(dense=True)`) then ResBlock(99) `unet.py:87-90` | 16-channel block without skip, 3×3 head |
| Init | kaiming_normal, ICNR for the shuffle, BN bias 1e-3 | kaiming_uniform (fan_in, relu), xavier head. The variance at init is about the same. |
| Odd sizes | resize in `UnetBlock` plus `ResizeToOrig` | SMP 0.5.0 Unet allows non-divisible sizes (`requires_divisible_input_shape=False`). Other SMP architectures raise. |

The following are **not** differences: the loss (fastai's `CrossEntropyLossFlat` equals
`nn.CrossEntropyLoss` on NCHW), weight-variance init, odd-size support in SMP 0.5.0's Unet,
`BatchZero` residual init (not active in the default DynamicUnet), blur and self-attention
(both off by default).

### 1.3 The training recipe differs

`unet_learner(...).fine_tune(n)` (`fastai/vision/learner.py:266-293`, `callback/schedule.py:164-171`):

* Adds ImageNet normalisation automatically when `pretrained=True`. A hand-written SMP
  pipeline often gets this wrong or leaves it out.
* fastai's `Adam` is AdamW: β=(0.9, **0.99**), eps **1e-5**, decoupled wd 0.01. **No weight decay
  on BN or bias parameters**, and BN parameters keep training while the encoder is frozen
  (`learner.py:189-201`, `optimizer.py:26-41,138-140,233-245`).
* One frozen epoch (one-cycle, decoder LR 2e-3), then unfreezing with **discriminative LRs**
  1e-5 / 1e-4 / 1e-3 over 3 parameter groups, one-cycle (30% warmup, cosine), and momentum
  cycling.
* A typical SMP training loop uses Adam/AdamW with default eps and β2, decay on everything, a
  single constant or plateau LR, and no warmup or freezing.

Pretrained weights are another trap. fastai uses torchvision `DEFAULT` weights, which are
V1 for resnet34 but **V2 for resnet50**. SMP's `resnet34` uses the torchvision V1 port, but
SMP `tu-resnet34` loads timm's default `a1_in1k`, a different recipe. We verified that timm
`resnet34.tv_in1k`, SMP `resnet34`/imagenet and torchvision `IMAGENET1K_V1` are byte-identical.

### 1.4 Ranked hypotheses and how manyseg-bench tests them

1. **Decoder capacity** (most likely): about 6× the params and 9× the FLOPs. Control: SMP or
   ManySeg U-Net with fastai-like widths.
2. **Training recipe** (likely of similar size when the SMP loop is hand-rolled): one-cycle,
   freeze then unfreeze, discriminative LRs, no wd on norm/bias, eps 1e-5. Control: the
   `fastai_like` recipe applied to every provider (E002 stage B).
3. **Full-resolution refinement with the input image concatenated**, which may help boundaries
   (boundary F1 is reported).
4. **Normalisation placement**: BN on skips and none in the decoder, which is robust at small
   batch sizes.
5. **ICNR pixel-shuffle upsampling** (probably minor).

E002 measures (2) directly. If fastai still wins with an identical recipe, E003 widens the
ManySeg and SMP decoders to fastai's FLOPs to separate (1) from (3) to (5).

### 1.5 A shared spatial-alignment bias (found while testing ManySeg)

ManySeg's gradient-centroid alignment test found that standard U-Net decoders shift
predictions up and to the left (ManySeg nearest: about 6 px at init; SMP Unet: about 10 px).
Odd-kernel "same"-padded strided convs centre feature `j` at `s·j`, while nearest and
bilinear (`align_corners=False`) upsampling assume `s·j + (s−1)/2`. A grid-consistent
bilinear upsample removes the bias. See `experiments/E001_upsample_alignment`. E001b later
measured trained accuracy: removing the offset changed nothing, so nearest stays the default.

---

## 2. timm as the backbone provider

All results are from timm 1.0.30 with `pretrained=False`. Raw table:
`experiments/E000_review_measurements/timm_audit.md`.

### 2.1 Feature APIs

* `features_only=True` (`models/_builder.py:437-501`) wraps the model in `FeatureListNet`
  (default), a model-specific class (`EfficientNetFeatures`, `MobileNetV3Features`), or
  `FeatureGetterNet` (the default for vit, eva, hiera and levit).
* `feature_info.channels() / reduction() / module_name()` (`models/_features.py:128-138`) is a
  reliable source of static metadata.
* **All 19 families tested implement `forward_intermediates` / `prune_intermediate_layers`.**
  `feature_cls="getter"` worked for every one of them and always returned NCHW.
* Layout trap: `features_only` passes the model's own `output_fmt`, so **Swin and MambaOut
  return NHWC** from the default wrapper. The getter path returns NCHW.
* ViT `features_only` defaults to the last 3 blocks. Segmentation needs explicit
  `out_indices` (e.g. `(2, 5, 8, 11)`).

### 2.2 Dynamic resolution (output of the last stage at 224², 256×320, 257×301, 97×131)

| Family | Sizes accepted | Rounding |
| --- | --- | --- |
| resnet, regnet, efficientnet, mobilenetv3/v4, fastvit, pvt_v2, nextvit, mambaout | any | ceil |
| convnext, edgenext (patchify stem) | any | **floor** |
| caformer | any | mixed |
| vit (default) | fixed | n/a |
| vit/eva + `dynamic_img_size=True` | multiple of patch | n/a |
| vit/eva + `dynamic_img_size` + `dynamic_img_pad` | any | ceil (bottom/right pad) |
| swin + `strict_img_size=False` | multiple of 4 (`dynamic_img_pad` is silently ignored) | n/a |
| maxvit, coatnet, hiera, levit | fixed (rel-pos / attention bias); fixable with `img_size=` at create time | n/a |

ViT pos-embeds are resampled bicubically on each forward when `dynamic_img_size` is set
(`vision_transformer.py:1057`, `layers/pos_embed.py:19`), with prefix tokens kept separate.

Because rounding differs between families, aligning skips by "multiply by 2" is not safe.
ManySeg pads inputs to the backbone's divisor and resizes to each skip's exact size.

### 2.3 Input channels

`adapt_input_conv` (`models/_manipulate.py:289-318`): C=1 sums the RGB filters. Otherwise
it tiles RGB, slices to C and scales by 3/C. Only a channel-constant input with C=1 or
C % 3 == 0 keeps the pretrained response. For C=4 the R filter counts twice. ManySeg
reimplements this rule as `repeat` (tested bit-for-bit against timm) and adds `mean`,
`rgb_plus_zero`, `rgb_plus_random` and `projection`.

### 2.4 Compile and export

| Model | export (free H/W) | export (32·Dim) | compile fullgraph dynamic |
| --- | --- | --- | --- |
| resnet34 / convnext_tiny / efficientnet_b0 features | OK | OK | OK |
| vit_s16 + dynamic_img_size | fails (guard H % 16) | OK | OK |
| vit_s16 + dynamic_img_size + dynamic_img_pad | OK | OK | OK |
| swin_tiny strict_img_size=False | **hangs** (>600 s) | hangs | OK |

Swin as a ManySeg model behaves differently today: on 2026-11-02, free H/W export of
`create_model(backbone="swin_tiny_patch4_window7_224")` raised a constraint error after about
165 s (CPU, shared machine) rather than hanging.

An exported program with `32·Dim` dims still accepted a 257×301 input without error, so
divisibility must be validated outside the export.

### 2.5 Consequence: probe capabilities instead of keeping allow-lists

ManySeg will maintain a capability registry that is **generated by probing**, not hand-written:

1. Build with `features_only=True`. If that fails or returns an unsuitable layout, fall back to
   `feature_cls="getter"`.
2. Detect the layout by a dry forward (`shape[1] == channels` means NCHW).
3. If `__init__` accepts them (`inspect.signature`), enable `dynamic_img_size` /
   `dynamic_img_pad` / `strict_img_size=False`.
4. Classify sizes by forward probes at an odd, a non-square and a small size: `any`,
   `multiple_of(k)` or `fixed`.
5. Record rounding (ceil, floor or mixed) and the required divisor.
6. Probe export and compile in a subprocess with a timeout, because Swin hung.

Results are cached per (model name, kwargs) and checked into the repo as a generated
compatibility table that the slow test tier regenerates.

> **As implemented (0.1.0)** this plan was simplified: `TimmBackbone` always uses
> `features_only=True` and reads the layout from timm's `output_fmt`, and only fixed-size
> models are probed at build time. Size and export probing (in a subprocess with a
> timeout) lives in `scripts/backbone_compat.py`, which regenerates
> `docs/backbone_compatibility.md` when run by hand; no test tier runs it.

---

## 3. MMSegmentation

* Assembly: string-keyed registries plus config inheritance (`_base_`).
  `EncoderDecoder = backbone → neck → decode_head (+ auxiliary heads)`
  (`mmseg/models/segmentors/encoder_decoder.py`).
* `BaseDecodeHead` mixes input selection (`in_index`, `input_transform`), the classifier and
  **losses** in one class. This is the anti-pattern ManySeg avoids: heads return tensors and
  losses live elsewhere.
* It has the broadest faithful decoder zoo: UPerHead, FPN, ASPP / DepthwiseSeparableASPP
  (DeepLabV3+), SegFormer, DPT, SETR, PSP, OCR. Useful as a reference for each decoder.
* ViT to pyramid: `MultiLevelNeck` (resize + conv) and `Feature2Pyramid`
  (ConvT×2 / ConvT / Id / MaxPool, BEiT/MAE style).
* `resize_pos_embed` runs on every forward when the grid differs (dynamic H/W).
* Sliding-window inference with plain averaging. It mutates image metadata.
* Heavyweight: mmcv/mmengine, required metas dictionaries, SyncBN defaults, Python shape
  logic. Not export-friendly. Effectively unmaintained since 2024.

## 4. TerraTorch

* `MultiSourceRegistry` with prefixed names (`timm_`, `smp_`, `mmseg_`). Kwargs are routed by
  prefix (`backbone_`, `decoder_`, `head_`). `_check_all_args_used` errors on leftovers,
  which is a good idea.
* Necks implement `process_channel_list(channels)`, so channel bookkeeping is static. This
  is good, and ManySeg's `FeatureSpec` does the same job.
* `ReshapeTokensToImage` infers the grid from `sqrt(N)` and, on failure, mutates
  `self.remove_cls_token` and recurses. That is stateful and not compile-safe. ManySeg
  computes the grid from the input shape inside the backbone wrapper instead.
* Band adaptation (`select_patch_embed_weights.py`) copies channels by band name and leaves
  unmatched bands random. Named band metadata is valuable. The key lookup by string suffix
  is fragile.
* TerraMind multimodal: per-modality embeddings, tokens concatenated (mid fusion), a learned
  `missing_mod_token`, and modality dropout. These are good ideas for missing modalities.
* `TemporalWrapper` folds time into the batch and pools with mean, max, concat or diff.
* Weaknesses: monkeypatched forwards, Python shape branches in `forward`, near-duplicate
  factories, and heavy Lightning/jsonargparse coupling.

## 5. TorchGeo

* Weights enums carry `meta` (`in_chans`, `bands`, dataset, SSL method) and **transforms**
  (normalisation) together with the URL. This is the right idea for multispectral weights.
* There is no channel adaptation: the user must match the band set exactly.
* DOFA builds patch-embedding weights from wavelengths with a hypernetwork. This is a
  candidate "spectral adapter" for Phase 5.
* `datasets/splits.py` has leakage-aware splitting (`random_grid_cell_assignment`, ROI and
  time splits). manyseg-bench should use the same approach for datasets without official
  geographic splits.
* Tasks build SMP models. TorchGeo users therefore inherit SMP's decoder sizes.

## 6. Hugging Face transformers

* `BackboneMixin` / `load_backbone`: `out_features`/`out_indices` and `.channels`, with feature
  maps returned as a tuple. Feature hooks are installed at init to avoid graph breaks. The
  contract is equivalent to ManySeg's `feature_specs`.
* UperNet takes any backbone that exposes `.channels`.
* SegFormer head: per-level MLP, upsample to 1/4, concat, fuse. Logits stay at 1/4 resolution
  and are upsampled in the loss.
* DPT `ReassembleStage`: strips the cls token, reshapes using the **patch grid computed from
  the input**, readout ∈ {ignore, add, project}, then ConvT / Id / strided conv to make a
  pyramid. This is the model for ManySeg's ViT neck.
* `interpolate_pos_encoding`: bicubic in float32, unconditional while tracing so exported
  graphs stay dynamic.

## 7. Datasets considered for the benchmark

The selection and its justification are in `benchmark_design.md`. Licensing and access
details (as of 2026-10-14):

| Dataset | License | Access | Notes |
| --- | --- | --- | --- |
| Oxford-IIIT Pet | CC BY-SA 4.0 | direct (robots.ox.ac.uk; slow) | Images came from the HF mirror `antokun/...`; its annotations are byte-identical to the official tarball. |
| Sen1Floods11 | CC BY 4.0 | public GCS bucket / HF `blumenstiel/Sen1Floods11` | Event-correlated chips; also report the Bolivia holdout. |
| CamVid | research use | GitHub (SegNet-Tutorial) | 367/101/233. |
| WHU Building | research use; the HF mirror lists CC BY 4.0 | direct zip 5.4 GB | Use the official split, not the HF mirror's re-split. |
| MADOS | CC BY 4.0 | Zenodo 10664073 | Check that scene ids are disjoint across splits. |
| LoveDA | Zenodo lists CC BY 4.0, GitHub lists CC BY-NC-SA | Zenodo 5706578 | Test labels hidden. |
| ISPRS Vaihingen | no redistribution | password-protected | Optional. |

## 8. Design decisions for ManySeg

| # | Decision | Evidence |
| --- | --- | --- |
| D1 | Encoder contract: `list[Tensor]` NCHW plus static `feature_specs`. Layout conversion and the token→grid step happen inside the backbone wrapper. | HF `BackboneMixin`, timm `feature_info`; NHWC trap (§2.1). |
| D2 | Pad to the backbone divisor, then crop (bottom/right), as the default spatial policy. Decoders resize to the exact skip size. | Mixed ceil/floor rounding (§2.2). Tests: padding semantics are exact. |
| D3 | Heads return raw tensors. No losses in the model. Named dict outputs when there are several heads. | mmseg anti-pattern (§3). |
| D4 | Channel adaptation is a pure weight transform applied to the stem found via `pretrained_cfg['first_conv']`, with several strategies. | timm `adapt_input_conv` bias (§2.3). TerraTorch fragility (§4). |
| D5 | Capability probing and a generated compatibility table instead of allow-lists. | §2.5. |
| D6 | Reference decoders replicate their original library at default settings (U-Net = SMP). Improvements are opt-in until experiments justify them. | Benchmark integrity. |
| D7 | No `.item()`, `random`, metadata dictionaries or tensor-dependent branches in `forward`. Sizes use floor-division arithmetic and crop via negative padding. | Export/compile findings (testing_strategy.md). |
| D8 | ViT neck: DPT/ViTDet-style reassembly using the patch grid computed from the input. Configurable scales. | §3, §4, §6. |
| D9 | Multi-modal: an explicit fusion module per pyramid level and an explicit alignment policy. Missing-modality support (learned placeholder) is a later feature. | TerraMind (§4). |
