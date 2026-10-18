# Changelog

All notable changes to `manyseg` (the library) and `manyseg-bench` (the benchmark). Versions
follow [Semantic Versioning](https://semver.org/); before 1.0, minor versions may change the
API, and every such change is listed here.

## 0.1.0 (unreleased)

First release. Python 3.10–3.14, PyTorch ≥ 2.9, timm ≥ 1.0.20.

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
* **Presets** (`manyseg.presets`): four measured speed / accuracy configurations.
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

### Deployment

* `manyseg.save` / `manyseg.load` (safetensors plus the stored configuration, normalisation,
  activations and class names). Save format 2; format-1 files (development versions, which
  said `"channels"` for all three) are migrated on load.
* `torch.compile(dynamic=True)`, `manyseg.export.export_model` (dynamic size; dynamic batch
  with `dynamic_batch=True`; dynamic date count except for `concat` / `diff` aggregation) and
  ONNX; int8 quantisation (`manyseg.quantize`, extra `quant`); FSDP2
  (`manyseg.distributed.shard_model`).
* `model.describe()` shows what was built; `model.describe(sample)` adds shapes and FLOPs.
* `export_model` and `quantize_pt2e` export in eval mode and then restore every submodule's
  train / eval flag; `dynamic_batch=True` needs an example with batch size ≥ 2.
* Decoders upsample to their skip connections with an exact integer nearest rule
  (`manyseg.utils.spatial.resize_nearest`) because `torch.compile(dynamic=True)` on CUDA
  miscomputes `F.interpolate(mode="nearest")` (draft upstream report in `docs/upstream/`).
  With stride padding (the default) this is bit-identical to `F.interpolate`; without it, a
  few rows can differ from PyTorch's float32 rounding.

### Testing

* Unit tests in CI on Linux, with macOS and Windows on pull requests and nightly; the slow
  tier (integration and examples: compile, export, ONNX, quantisation) on Linux. Python
  3.13 and 3.14 pass every tier locally; Python 3.15 (release candidate) passes everything
  except `torch.compile`, which PyTorch does not support on 3.15 yet.
* A random-configuration fuzzer (`packages/manyseg/tests/fuzz`) checks thousands of
  `create_model` configurations on CPU and CUDA (bf16, `torch.compile`); about 25 bugs it
  found are regression tests.

### Benchmark (`manyseg-bench`)

* Providers for ManySeg, segmentation-models-pytorch, fastai and plain timm; GPU-resident
  seeded data loading; recipes; JSON run records with provenance.
* Datasets: Oxford Pet, Sen1Floods11, CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS, EuroSAT,
  BigEarthNet v2, OSCD, LEVIR-CD, SECOND.
* Pre-registered experiments E000–E024 with write-ups in `experiments/`, negative results
  included.
