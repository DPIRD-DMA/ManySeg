# Changelog

All notable changes to `manyseg` (the library) and `manyseg-bench` (the benchmark). Versions
follow [Semantic Versioning](https://semver.org/); before 1.0, minor versions may change the
API, and every such change is listed here.

## 0.1.0 (2026-11-11)

First release. Python 3.11–3.14, PyTorch ≥ 2.10, timm ≥ 1.0.20.

### Models (`manyseg.create_model`)

* **Encoders:** any timm model with multi-scale features: CNNs, hierarchical transformers and
  plain ViTs (through a feature-pyramid neck). 46 models are tested
  ([compatibility table](docs/backbone_compatibility.md)); fixed-size backbones pad smaller
  inputs and refuse larger ones clearly. Any number of input bands, with stem adapters
  (`channel_adapter`) that start each band from a pretrained filter.
* **Decoders:** U-Net (the default, matching SMP's), U-Net++, FPN, DeepLabV3+, UPerNet,
  SegFormer-MLP and a fastai-style Dynamic U-Net; reduced output resolution (`out_stride`) and
  per-head `scale`. With a dilated encoder (`output_stride=16`), every decoder and image head
  reads the last level of each stride.
* **Several inputs:** early, intermediate or late fusion with seven ops; inputs on the same grid
  or covering the same extent at another resolution; shared (Siamese) encoders, with `diff` /
  `concat_diff` fusion and per-input heads for change detection.
* **Metadata:** vector inputs (cyclic, continuous, coordinates, categorical, with missing
  values) injected by FiLM, a head bias or extra channels. Fields are encoded in float32
  whatever the model's dtype, so a bfloat16 or float16 model keeps category ids and
  coordinates exact when given float32 metadata.
* **Outputs:** any number of named heads (segmentation, regression with optional uncertainty,
  classification; per pixel or per image), per-head decoders, and image classification or
  regression without a decoder. Pixel heads take optional hidden layers, dilation, norm (the
  same names as the decoders: `"batch"`, `"group"`, `None`) and a softplus activation; custom head classes register with `manyseg.heads.HEADS`.
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
  boxes (YOLO, `xyxy`, COCO), peak finding, density counting (`count_from_density(sigma=)` undoes
  `density_loss`'s 2πσ² target scale; a density target sums to the number of points inside the
  tile, and a point near a shared edge counts in exactly one tile), `density_loss` and threshold
  calibration. `instances_from_outputs` treats non-finite distances as 0.

### Vector outputs (`manyseg.vectors`, `manyseg[vectors]`). *Experimental*

Names and defaults may change before 1.0, and each decoder was tuned on one dataset.

* Building footprints (straight walls), field boundaries, road graphs (junctions and overpasses),
  LoD2 roof faces (with an nDSM input) and points, each with a head preset, target encoders from
  vector data in a CRS, losses and a decoder.
* `decode_scene(fields, block_decoder(task))` decodes a scene of any size block by block; with
  a halo that also covers the decoder's reach it gives the same result as a whole-scene decode
  (see `manyseg.vectors.blocks`). Every result (`SceneRecords`, `VectorGraph`, `ScenePoints`)
  has `.write(path)` to GeoJSON or GeoPackage in the scene's CRS; without a transform the
  coordinates stay in pixels and no CRS is written. Block decoders work on the grid of the
  heads they read, so a reduced-resolution head needs a scene for that grid.
* Road graphs from a georeferenced scene carry its transform and CRS (`VectorGraph.write()`,
  `.to_geojson()`). Geometry conversions keep Z coordinates.
* Building blocks: `graph_from_mask`, `polygons_from_labels`, `to_pixels`, `to_geojson`,
  `pixel_to_world` / `world_to_pixel`, dihedral augmentation, and metrics (APLS, junction F1,
  polygon C-IoU, PoLiS and others) in `manyseg.vectors.metrics`.

### Losses (`manyseg.losses`)

* `soft_dice`, `dice_bce`, `multiclass_dice`, `masked_regression`, `heatmap_focal`,
  `distance_weighted_bce`, `density_loss` and `multi_head_loss`. `soft_dice`, `dice_bce`,
  `masked_regression` and `multiclass_dice` drop masked (or `ignore_index`) pixels before any
  arithmetic, so NaN nodata there is safe; a float mask weights pixels. `dice_bce`'s options
  (`pos_weight`, `mask`, weights) are keyword-only. The target must have the prediction's
  shape (a `(B, H, W)` target for `(B, 1, H, W)` logits is an error, not a broadcast).

### Saving, loading and deployment

* `manyseg.save` / `manyseg.load`: one safetensors file with the weights and the stored
  configuration, normalisation (a ManyPred spec, its per-band values checked against every
  raster and time-series input), activations, class and band names, and the licence and
  citation of the pretrained weights the model started from, which a loaded model keeps (its
  stem is rebuilt for the weights' bands without downloading them). No pickled data.
  Save format 2; see `manyseg.io` for the compatibility policy.
* Hugging Face Hub: `manyseg.push_to_hub` with a generated model card (naming the int8 mode
  and the `manyseg[quant]` extra for a quantised model), and
  `manyseg.load("hf-hub:org/name[@revision]")`.
* `tiling_spec()` tells [ManyPred](https://github.com/DPIRD-DMA/ManyPred) how to run the model
  over large scenes.
* `torch.compile(dynamic=True)`, `manyseg.export.export_model` (dynamic H/W, or fixed with
  `dynamic=False` for window attention; optionally dynamic batch) and ONNX.
* int8 quantisation (`manyseg.quantize`, `manyseg[quant]`, torchao ≥ 0.18): weight-only /
  dynamic (`quantize_weights`) and PT2E static (`quantize_pt2e`, on a copy of the model), both
  savable with `manyseg.save`, plus size and agreement reports. A loaded `.pt2` runs on any
  `map_location`, takes multi-input dicts in any key order and can be saved again;
  `quantize_weights` keeps shared encoders' weights shared.
* `manyseg.describe`, `manyseg.export` and `manyseg.quantize` are reachable after
  `import manyseg` (loaded on first use).

### Examples and notebooks

* `examples/`: runnable scripts for each task, and for training with plain PyTorch, fastai,
  Lightning, Accelerate and TorchGeo.
* `notebooks/`: flood mapping on Sen1Floods11, optical + radar with several heads, and one
  notebook per vector output, saved with their figures.

### Known issues

* On Windows, PyTorch's CPU bfloat16 can crash with an illegal-instruction error
  (`0xc000001d`) on some processors, inside `nn.Linear`. Use float32 on the CPU there; GPUs
  and other platforms are not affected.

### Benchmark (`manyseg-bench`, not published)

* Providers for ManySeg, segmentation-models-pytorch, fastai and plain timm (SMP and fastai
  build pixel models only; image-level tasks use timm or ManySeg); GPU-resident seeded data
  loading; recipes; JSON run records with provenance. `evaluate` prints each split's primary
  metric, and `profile --output` profiles image-level models.
* Datasets: Oxford Pet, Sen1Floods11, CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS, EuroSAT,
  BigEarthNet v2, OSCD, LEVIR-CD, SECOND.
* Metrics: per-class scores and macro means cover only classes present in the ground truth;
  boundary F1 skips images with no edge in either prediction or target; validation loss is
  the training objective (label smoothing included; for `seg_multilabel` the segmentation
  plus weighted classification loss) per non-ignored pixel; change-class scores
  and regression R² follow the same absent-class (NaN) rule; multi-label AP groups tied
  scores; instance matching is exact without scipy. Label smoothing applies only to the
  cross-entropy tasks, and other tasks refuse it. Run records in `experiments/` written
  before 2026-11-10 used the older boundary-F1 and validation-loss rules, and records written
  before this release may differ in change-class scores, R², tied-score AP, instance matching
  below IoU 0.5, absolute-distance instance decoding (now `seed_threshold` 3 px) and
  `seg_multilabel` validation loss. Reports group seeds whose task settings differ only by
  restating a default, and show every declared test split (e.g. Sen1Floods11's Bolivia).
* Runs record the normalisation source, the settings each task applied (`task_settings`) and
  how each provider adapted a pretrained stem to more or fewer than 3 bands (`model_notes`).
  `contended` means another process was on the run's GPU (its own utilisation no longer
  counts); `code_version` is `-dirty` for a modified tracked file anywhere in the repository;
  image-level runs record `arch: none` (the requested value as `requested_arch`). Older records
  may say `contended: true` for uncontended runs and keep their requested arch.
* LEVIR-CD ignores edge padding when `--size` doesn't divide 1024. `report` and
  `experiments/analyse_runs.py` skip non-run JSON and non-training records.
* The experiments in `experiments/` that the defaults and the README's evidence table rest on,
  negative results included.
