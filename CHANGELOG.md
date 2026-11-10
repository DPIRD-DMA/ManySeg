# Changelog

All notable changes to `manyseg` (the library) and `manyseg-bench` (the benchmark). Versions
follow [Semantic Versioning](https://semver.org/); before 1.0, minor versions may change the
API, and every such change is listed here.

## 0.1.0 (unreleased)

First release. Python 3.11–3.14, PyTorch ≥ 2.10, timm ≥ 1.0.20.

### Models (`manyseg.create_model`)

* **Encoders:** any timm model with multi-scale features: CNNs, hierarchical transformers and
  plain ViTs (through a feature-pyramid neck). 46 models are tested
  ([compatibility table](docs/backbone_compatibility.md)); fixed-size backbones pad smaller
  inputs and refuse larger ones clearly. Any number of input bands, with stem adapters
  (`channel_adapter`) that start each band from a pretrained filter.
* **Decoders:** U-Net (the default, matching SMP's), U-Net++, FPN, DeepLabV3+, UPerNet,
  SegFormer-MLP and a fastai-style Dynamic U-Net; reduced output resolution (`out_stride`) and
  per-head `scale`.
* **Several inputs:** early, intermediate or late fusion with seven ops; inputs on the same grid
  or covering the same extent at another resolution; shared (Siamese) encoders, with `diff` /
  `concat_diff` fusion and per-input heads for change detection.
* **Metadata:** vector inputs (cyclic, continuous, coordinates, categorical, with missing
  values) injected by FiLM, a head bias or extra channels.
* **Outputs:** any number of named heads (segmentation, regression with optional uncertainty,
  classification; per pixel or per image), per-head decoders, and image classification or
  regression without a decoder. Pixel heads take optional hidden layers, dilation, norm and a
  softplus activation; custom head classes register with `manyseg.heads.HEADS`.
* **Time series** (`"kind": "timeseries"`) with masks, acquisition dates and mean / max /
  attention / concat / diff aggregation. *Experimental.*
* **Remote-sensing pretrained weights** (`manyseg.weights`): SSL4EO-S12 Sentinel-2 MoCo and DINO,
  SSL4EO-S12 Sentinel-1 MoCo and SatlasPretrain Sentinel-2, with their licences, citations and
  pretraining normalisation.
* **Features and embeddings:** `features()`, `embed()`, an exportable `embedder()`;
  `select_heads()` for deployment; `manyseg.utils.concat_heads` for pipelines that expect one
  tensor.
* `model.describe()` shows what was built, stage by stage; `describe(sample)` adds shapes and
  FLOPs.
* Every parameter gets a gradient, so DDP needs no `find_unused_parameters`.
* Configuration mistakes are errors: unknown input keys at `forward`, unused `decoder_kwargs`,
  conflicting arguments and similar are refused rather than ignored.
* `model.set_grad_checkpointing()`: opt-in encoder gradient checkpointing.

### Instances, points and counting

* Instance outputs (`manyseg.instances`): boundary + distance head presets, targets, losses and
  post-processing (`instances_from_outputs`, with a deterministic tie-break so a window decodes
  as the whole scene does).
* Points and counting (`manyseg.points`): heatmap and density heads, targets from points or
  boxes (YOLO, `xyxy`, COCO), peak finding, density counting, `density_loss` and threshold
  calibration.

### Vector outputs (`manyseg.vectors`, `manyseg[vectors]`). *Experimental*

Names and defaults may change before 1.0, and each decoder was tuned on one dataset.

* Building footprints (straight walls), field boundaries, road graphs (junctions and overpasses),
  LoD2 roof faces (with an nDSM input) and points, each with a head preset, target encoders from
  vector data in a CRS, losses and a decoder.
* `decode_scene(fields, block_decoder(task))` decodes a scene of any size block by block, giving
  the same result as a whole-scene decode; results write to GeoJSON or GeoPackage in the scene's
  CRS.
* Building blocks: `graph_from_mask`, `polygons_from_labels`, `to_pixels`, `to_geojson`,
  `pixel_to_world` / `world_to_pixel`, dihedral augmentation, and metrics (APLS, junction F1,
  polygon C-IoU, PoLiS and others) in `manyseg.vectors.metrics`.

### Losses (`manyseg.losses`)

* `soft_dice`, `dice_bce`, `multiclass_dice`, `masked_regression`, `heatmap_focal`,
  `distance_weighted_bce`, `density_loss` and `multi_head_loss`. Masked positions are excluded
  before any arithmetic, so NaN nodata under a mask is safe.

### Saving, loading and deployment

* `manyseg.save` / `manyseg.load`: one safetensors file with the weights and the stored
  configuration, normalisation (a ManyPred spec), activations, class and band names, and the
  licence and citation of the pretrained weights the model started from. No pickled data.
  Save format 2; see `manyseg.io` for the compatibility policy.
* Hugging Face Hub: `manyseg.push_to_hub` with a generated model card, and
  `manyseg.load("hf-hub:org/name[@revision]")`.
* `tiling_spec()` tells [ManyPred](https://github.com/DPIRD-DMA/ManyPred) how to run the model
  over large scenes.
* `torch.compile(dynamic=True)`, `manyseg.export.export_model` (dynamic size, optionally dynamic
  batch) and ONNX.
* int8 quantisation (`manyseg.quantize`, `manyseg[quant]`, torchao ≥ 0.18): weight-only /
  dynamic (`quantize_weights`) and PT2E static (`quantize_pt2e`), both savable with
  `manyseg.save`, plus size and agreement reports.

### Examples and notebooks

* `examples/`: runnable scripts for each task, and for training with plain PyTorch, fastai,
  Lightning, Accelerate and TorchGeo.
* `notebooks/`: flood mapping on Sen1Floods11, optical + radar with several heads, and one
  notebook per vector output, saved with their figures.

### Benchmark (`manyseg-bench`, not published)

* Providers for ManySeg, segmentation-models-pytorch, fastai and plain timm; GPU-resident seeded
  data loading; recipes; JSON run records with provenance.
* Datasets: Oxford Pet, Sen1Floods11, CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS, EuroSAT,
  BigEarthNet v2, OSCD, LEVIR-CD, SECOND.
* The experiments in `experiments/` that the defaults and the README's evidence table rest on,
  negative results included.
