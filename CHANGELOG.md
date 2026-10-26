# Changelog

All notable changes to `manyseg` (the library) and `manyseg-bench` (the benchmark). Versions
follow [Semantic Versioning](https://semver.org/); before 1.0, minor versions may change the
API, and every such change is listed here.

## 0.1.0 (unreleased)

First release. Python 3.10–3.14, PyTorch ≥ 2.10, timm ≥ 1.0.20.

### Models (`manyseg.create_model`)

* **Config vocabulary:** an input's bands are `in_channels` (also the single-input argument),
  a head's outputs `out_channels`, a raw metadata vector's width `num_features`. Head
  `"type"` follows its `"scope"`: `segmentation` per pixel, `classification` per image,
  `regression` either. `decoder` takes a name plus `decoder_kwargs`, or one
  `{"type": ..., **kwargs}` dict like per-head decoders.

* **Encoders:** any timm model with multi-scale features: CNNs, hierarchical transformers
  and plain ViTs (through a feature-pyramid neck). 46 models are tested
  (`docs/backbone_compatibility.md`); fixed-size backbones pad smaller inputs and refuse
  larger ones clearly. Any number of input bands, with stem adapters (`channel_adapter`).
* **Decoders:** U-Net (default; SMP-identical), U-Net++, FPN, DeepLabV3+, UPerNet,
  SegFormer-MLP and a fastai-style Dynamic U-Net; reduced output resolution (`out_stride`)
  and per-head `scale`.
* **Several inputs:** early, intermediate or late fusion with seven ops; inputs on the same
  grid or covering the same extent at another resolution; shared (Siamese) encoders.
* **Metadata:** vector inputs (cyclic, continuous, coordinates, categorical; missing values)
  injected by FiLM, head bias or extra channels.
* **Outputs:** any number of named heads (segmentation, regression with optional
  uncertainty, classification; pixel or image scope), per-head decoders, instance
  (boundary + distance) and point / density presets with targets and post-processing.
* **Image classification and regression** without a decoder.
* **Time series** (`"kind": "timeseries"`) with masks, acquisition dates and mean / max /
  attention / concat / diff aggregation. *Experimental:* on our two datasets it beat
  stacking dates as channels on one and lost on the other (E021).
* **Change detection:** `diff` / `concat_diff` fusion and per-input heads for semantic change
  (E022, E023).
* **Remote-sensing pretrained weights** (`manyseg.weights`): SSL4EO-S12 Sentinel-2
  MoCo (ResNet-18/50, ViT-S/16) and DINO (ResNet-50, ViT-S/16), SSL4EO-S12 Sentinel-1 MoCo
  (ResNet-50) and SatlasPretrain Sentinel-2 (ResNet-50/152). SSL4EO beat ImageNet weights on three Sentinel-2
  datasets with a ResNet-50 (E024); cite the authors (`WEIGHTS[key].citation`).
* **Presets** (`manyseg.presets`): four measured speed / accuracy configurations, with GPU
  and CPU (eager and compiled) latency; on CPU they run at about the default's speed.
* **Features and embeddings:** `features()`, `embed()`, an exportable `embedder()`;
  `select_heads()` for deployment.
* Every parameter gets a gradient (DDP needs no `find_unused_parameters`): modules for
  feature levels nothing reads are not built.
* Mistakes are errors, not silently ignored: unknown input keys at `forward` (e.g. a
  misspelt `"s2_mask"`), a dict passed to a single-input model, `in_channels=` together with
  `inputs=`, an unknown `decoder` or unused `decoder_kwargs` when no shared decoder is built,
  `rgb_indices` with the `repeat` / `mean` adapters, a non-bool `uncertainty`, a non-integer
  `num_classes`, a string `fusion`. `select_heads("name")` takes one name.
* `manyseg.instances.instances_from_outputs` needs `logits=True` (raw head outputs) or
  `logits=False` (probabilities); there is no default.
* Renamed for the config vocabulary: `MetadataEncoder(num_features=)` (was `channels=`) and
  `Weights.in_channels` (was `in_chans`). The unimplemented `"token"` conditioning mode is no
  longer listed in `MODES`.
* **Pixel-head options** (all opt-in; the default head is still SMP's single conv, same
  parameters): `hidden` conv → norm → ReLU layers with per-layer `dilation` and `norm`
  (`"bn"`, `"gn"`, None) before the final conv, and `activation="softplus"` for regression
  heads, applied after resampling and after a metadata head bias. Custom head
  classes register with `manyseg.heads.HEADS.register(name)` and are built as `cls(spec,
  out_channels, **opts)`. `describe()`, save / load, `torch.export` / ONNX, the fuzzer and the
  web model builder cover them; `manyseg.save` stores `"none"` for a head that applies its own
  activation and refuses another.
* **Counting** (`manyseg.points`), from a downstream drone-imagery project:
  * `density_head(activation="softplus")` keeps the density positive (bias starts at -4).
    Validated on one downstream drone dataset (792 × 792 tiles, 1–20 plants of ~115 px,
    σ = 12, 45 val images, one seed): density count MAE 10.1 → 2.9 against the raw head
    (clamped at 0; 7.5 summed raw). It leaves a floor of about +1.75 objects per image, as
    softplus never reaches 0; peak counting on a heatmap still scored 1.0 there. The default
    stays no activation until E011's density arm is rerun with softplus.
  * A `"relu"` activation was tried and removed: on the same data the head died (the map
    went to 0, count MAE 3.5, correlation 0.37).
  * `density_loss`: a pixel loss against the mass-preserving target scaled by `2πσ²`, over
    an optional valid mask, plus an optional per-image count L1 term.
  * `count_from_density(value_scale=, mask=)`: it now sums in float32.
  * `find_peaks(smooth_sigma=, mask=)`: a separable Gaussian blur before NMS, and no peaks
    outside a mask. The defaults give the same peaks as before.
  * `points_from_boxes` turns YOLO, `xyxy` or COCO `xywh` boxes into `(y, x)` points, with an
    optional per-point sigma from the box size (at least `min_sigma`). `gaussian_heatmap` and
    `density_map` take a list of per-image sigma tensors, and refuse a sigma of 0.
  * `calibrate_threshold` chooses the peak threshold by count MAE or point F1 (E011b's
    calibration, previously in `manyseg-bench` only). It keeps peaks scoring above the
    threshold, as `find_peaks` does, and counts F1 matches with a maximum one-to-one matching.
    Its docstring warns that small validation sets give unstable thresholds.
  * The docs now qualify E011. Density counting won on many small objects per tile; on a few
    large ones per tile, peak counting was far better and stays the recommendation.
* **Single-tensor output:** `manyseg.utils.concat_heads(model, order)` for pipelines that
  expect one tensor, with each head's channel slice; `examples/09_fastai.py` trains a
  dict-output model with fastai (bf16, one-cycle, EMA; `get_preds(reorder=False)` for dict
  outputs).
* **Training frameworks:** examples for a plain PyTorch loop (`10_pytorch.py`), Lightning
  (`11_lightning.py`: a two-head model, checkpoints reloaded with `load_from_checkpoint`),
  Accelerate (`12_accelerate.py`) and TorchGeo (`13_torchgeo.py`: its segmentation task with
  `configure_models` swapped for a ManySeg model, keeping its losses, metrics and freezing).

### Performance (E026; values unchanged up to rounding)

* bf16 autocast no longer carries float32 activations where CUDA runs an op in float32:
  fastai_unet and U-Net `input_cross` (the input image), GroupNorm decoders and head layers,
  UPerNet and DeepLabV3+ (bilinear upsampling). Training memory −20% (fastai_unet), −27%
  (UPerNet), −19% (GroupNorm U-Net); steps up to 13% faster.
* SegFormer decoder: the fuse conv runs per level before upsampling (same parameters):
  −26 to −38% training memory, 10–15% faster.
* Cross-attention fusion: tokens are contiguous, so SDPA uses its fused kernels: −50% step
  time, −77% memory.
* Temporal mean sums bf16 features with a float32 accumulator instead of upcasting them;
  fastai_unet keeps channels_last through its pixel shuffles; `connected_components` uses
  pointer jumping (18–44× faster on long or field-like components, same labels).
* Bilinear resizes keep the activation dtype in U-Net `upsample="bilinear"` / `"aligned"`
  (−7% memory) and in late / `same_extent` fusion before a concat (E027).
* `model.set_grad_checkpointing()`: opt-in encoder gradient checkpointing (timm's), with
  in-place activations switched off and BatchNorm running statistics updated once per step,
  as without it. −22% memory for +5% step time (ResNet-34 U-Net), −61% for +18% (ConvNeXt-T).

### Deployment

* `manyseg.save` / `manyseg.load` (safetensors plus the stored configuration, normalisation,
  activations and class names). Save format 2; format-1 files (development versions, which
  said `"channels"` for all three) are migrated on load. Compatibility policy (`manyseg.io`):
  renamed or reinterpreted config keys, modules or parameters bump the format and get a
  migration; a committed golden file (`tests/data/golden_format2.safetensors`) must keep
  loading to the same outputs. A file from a newer ManySeg that this one cannot rebuild
  fails with both versions named and an upgrade hint, instead of a bare `create_model`
  error. The README warns against `torch.save(model)` pickles, which depend on internal
  module paths.
* `torch.compile(dynamic=True)`, `manyseg.export.export_model` (dynamic size; dynamic batch
  with `dynamic_batch=True`; dynamic date count except for `concat` / `diff` aggregation) and
  ONNX; int8 quantisation (`manyseg.quantize`, extra `quant`); FSDP2
  (`manyseg.distributed.shard_model`).
* Saved models carry the licence and citation of the pretrained weights they started from
  (`pretrained_weights` in the file metadata, `manyseg.weights.attribution(model)`), through
  load and re-save; `describe()` lists them. Fine-tuned weights stay under those licences
  (SSL4EO-S12 CC BY 4.0, SatlasPretrain ODC-BY, timm's per-checkpoint licences). Each
  record also gives the weights' pretraining normalisation (`normalization`: mean, std,
  units, clip), taken from timm's `pretrained_cfg` per weights tag, which is not always
  ImageNet's (0.5 / 0.5 for ViT AugReg and SigLIP, CLIP's for EVA-02).
* `model.describe()` shows what was built; `model.describe(sample)` adds shapes and FLOPs.
  An adapted stem says which pretrained filter each input channel starts from (so a mask
  channel given an RGB filter by `repeat` shows), and `manyseg.save` on a wrapper names the
  model inside it to save instead. The docs cover non-image channels (`rgb_plus_zero`),
  missing bands and which conditioning mode to use.
* `export_model` and `quantize_pt2e` export in eval mode and then restore every submodule's
  train / eval flag; `dynamic_batch=True` needs an example with batch size ≥ 2.
* Decoders upsample to their skip connections with the exact integer nearest rule, because
  `torch.compile(dynamic=True)` on CUDA miscomputes `F.interpolate(mode="nearest", size=...)`
  (draft upstream report in `docs/upstream/`). With stride padding (the default) this is
  bit-identical to `F.interpolate`; without it, a few rows can differ from PyTorch's float32
  rounding.
* **Training speed (E025).** That rule was implemented as an integer-index gather, whose
  backward is an atomic scatter-add: a U-Net training step was 2–5% slower than SMP's eager
  and 83–98% slower compiled (bf16, batch 40, 509², `edgenext_small` / `convnextv2_femto`).
  The U-Net, U-Net++, FPN and fastai U-Net now use `F.interpolate(scale_factor=2)` (which
  compiles correctly) or a plain crop wherever the sizes give the same values: in eager mode
  whenever a level is 2n or 2n − 1 of the next, compiled or exported when the model pads
  inputs to the stride (a static flag, no traced size check). The gather remains for other
  sizes. Outputs are the same (gradients up to summation order), and channels-last encoder
  features now stay channels-last through the decoder; the upsampled maps keep the input
  dtype under autocast (which runs upsampling in float32), so bf16 decoders use no extra
  memory. U-Net steps are now at parity with SMP compiled and 18–25% faster eager (SMP's two
  finest decoder blocks run in NCHW), with no more memory; FPN and the fastai U-Net are 14% and
  6% faster compiled. Whether a step is exactly 2x is decided per pair of levels, so sparse
  `out_indices` (e.g. strides 4, 16, 32) keep the gather where levels are 4x apart. A crop to an unchanged size no longer copies.

### Testing

* Unit tests in CI on Linux, with macOS and Windows on pull requests and nightly; the slow
  tier (integration and examples: compile, export, ONNX, quantisation) on Linux. Python
  3.13 and 3.14 pass every tier locally; Python 3.15 (release candidate) passes everything
  except `torch.compile`, which PyTorch does not support on 3.15 yet.
* A random-configuration fuzzer (`packages/manyseg/tests/fuzz`) checks thousands of
  `create_model` configurations on CPU and CUDA (bf16, `torch.compile`); about 25 bugs it
  found are regression tests.
* Training-framework integration tests (`tests/integration/test_trainers.py`): fastai,
  Lightning (fp32 and bf16-mixed), Accelerate and TorchGeo fit on CPU, with checkpoint and
  `manyseg.save` round trips. CI's slow job installs the frameworks and sets
  `MANYSEG_REQUIRE_TRAINERS=1`, so a missing one fails rather than skips.
* A `heavy` marker for slow tests that take minutes on CI (the Swin compile and export
  probes): pushes run `slow and not heavy`, nightly and manual runs all of `slow`. The
  Swin compile test uses one block per stage and one input size: 2.8x faster locally (438 s
  to 155 s), which should fit the floor job's 900 s limit.

### Benchmark (`manyseg-bench`)

* Providers for ManySeg, segmentation-models-pytorch, fastai and plain timm; GPU-resident
  seeded data loading; recipes; JSON run records with provenance.
* Datasets: Oxford Pet, Sen1Floods11, CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS, EuroSAT,
  BigEarthNet v2, OSCD, LEVIR-CD, SECOND.
* Pre-registered experiments E000–E027 with write-ups in `experiments/`, negative results
  included.
* Training loop without per-step host syncs on CUDA: fused AdamW, a non-finite loss skips the
  update on the device (as before, without waiting for the GPU), losses summed on the device,
  one gradient-scaling kernel; the loader copies the epoch order to the GPU once and normalises
  in place. 24–26% faster steps for the PVTv2-b0 presets (E026).
* With `compile`, evaluation uses its own static compiled model and pads the short last batch
  (+12–27% evaluation throughput, E027); device-resident datasets free their host arrays
  once on the GPU (−2.8 GB host RAM on Sen1Floods11).
