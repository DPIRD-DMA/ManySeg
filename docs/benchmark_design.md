# manyseg-bench design

`manyseg-bench` is a standalone, implementation-neutral benchmark for PyTorch segmentation
models. It depends on `torch`, `numpy`, `pillow` and `pyyaml`. ManySeg, SMP and fastai are
optional extras that are imported lazily by their providers. The `manyseg` library never
imports the benchmark; a test enforces this.

## Principle: the benchmark owns everything except the network

| Owned by the benchmark (identical for every provider) | Owned by the provider |
| --- | --- |
| Dataset preparation, resizing, splits | Building the `nn.Module` from a `ModelSpec` |
| Normalisation (mean/std from the dataset, per pixel) | Mapping canonical weight ids to library weights |
| Augmentation (seeded, on device) | Splitting parameters into encoder and decoder |
| Optimiser, schedule, epochs, batch size, loss | Unwrapping library-specific output types |
| Precision, `torch.compile`, channels_last | |
| Metrics, evaluation protocol, timing protocol | |
| Provenance (environment, versions, contention) | |

If a provider cannot meet the contract exactly (for example, fastai's `DynamicUnet` needs
`img_size` at construction), the difference is handled in the provider and recorded in
`BuiltModel.notes`.

## Provider contract

```python
spec = ModelSpec(arch="unet", encoder="resnet34", weights="tv_in1k",
                 in_channels=3, num_classes=3, options={})
built = get_provider("smp").build(spec, img_size=(256, 256))
built.model            # nn.Module: (N, C, H, W) normalised float -> (N, K, H, W) logits
built.encoder_params   # used by recipes with encoder LR multipliers or freezing
built.decoder_params
built.weights_fingerprint  # sha256 of the first conv weight (16 hex chars)
```

External libraries plug in through `register_provider(MyProvider)` without any change to
the benchmark.

### Weight fairness

Canonical weight ids map to library-specific names. `tv_in1k` (torchvision ImageNet V1)
resolves to:

* ManySeg: timm `resnet34.tv_in1k`
* SMP: `resnet34`, `encoder_weights="imagenet"`
* fastai: torchvision `IMAGENET1K_V1`

These were verified **byte-identical** on 2026-10-14 (all 216 shared tensors). Every result
records the encoder fingerprint, and `test_pretrained_weights_identical_across_providers`
guards the equivalence. Note the trap here: timm's *default* `resnet34` tag is `a1_in1k`, a
different training recipe. SMP's `tu-resnet34` therefore loads different weights from SMP's
`resnet34`.

## Recipes

A `Recipe` is a frozen dataclass with every optimisation choice. A run records the whole
recipe. Built-in recipes:

* `standard`: AdamW (lr 5e-4, wd 1e-4, no decay on norm/bias), 1-epoch linear warmup, then
  cosine. One LR for all parameters. bf16 autocast, batch 16.
* `fastai_like`: reproduces fastai `unet_learner` + `fine_tune` defaults (AdamW β=(0.9,0.99),
  eps 1e-5, wd 1e-2, no decay on norm/bias; 1 epoch with the encoder frozen and norm layers
  trainable; then encoder LR ×0.1; one-cycle). Any provider can be trained with it, so the
  recipe effect can be separated from the architecture effect.
* `smoke`: 1 epoch, for CI.

Model selection uses the **final epoch**, with no early stopping, so every model gets the
same budget. The best validation epoch is recorded for information only.

Recipe options added by experiments, all off by default and recorded in every run:

* `freeze_encoder_bn`: the encoder's BatchNorm layers (found by module path, so non-affine BN
  is included) stay in eval mode, keeping their running statistics at the pretrained values.
  Affine weights still train (E007).
* `grad_accum_steps = k`: one optimiser step per k micro-batches of `batch_size`. Each
  micro-batch's loss is weighted by its image count, and the lr schedule counts optimiser
  steps. BatchNorm still sees micro-batches (E007b).
* `calibrate` (point tasks; on by default since E011b): the peak threshold is chosen on
  validation, and test is scored once at it.

## Data protocol

* Each dataset module provides `INFO` (classes, ignore index, normalisation, license,
  source), `prepare(raw, cache)` with integrity checks (list counts, label value sets), and
  fixed splits.
* Prepared splits are uint8 `.npy` arrays. Small datasets are moved to the GPU once.
  Batching, random-resized-crop and flip augmentation (`grid_sample`: bilinear for images,
  nearest for masks) and normalisation all run on the device, seeded per run. The
  dataloader therefore never differs between providers, and it is not the bottleneck.
* **Test splits are not used for model selection.** Validation comes from the training data
  (for Pet, a deterministic 10% of `trainval` by name hash). `--evaluate-test` is an
  explicit flag, recorded in the result, for final comparisons of fixed configurations.
  It is never used during architecture search.
* Remote-sensing datasets must use spatially disjoint splits: official geographic splits
  where they exist, otherwise grid-cell or region assignment.

## Metrics

* Quality: confusion-matrix mIoU, mean Dice, pixel accuracy, per-class IoU/Dice/accuracy,
  boundary F1 (2 px tolerance, per image), validation loss, training-loss curve,
  non-finite step count (a stability signal).
* Efficiency: parameters, serialised `state_dict` bytes, forward FLOPs (`FlopCounterMode`,
  2×MAC).
* Inference: median, p10 and p90 latency at batch sizes 1, 8 and 32; images/s;
  megapixels/s; peak memory; eager vs compiled; fp32, bf16 and fp16; channels_last.
* Training: steady-state images/s and peak memory per step; wall-clock per epoch and in
  total; time to reach a target mIoU (from the history).

### Timing protocol

Warmup iterations come first, then each timed call is followed by device synchronisation
(`cuda.synchronize` or `mps.synchronize`), and the median is reported. CPU, CUDA and MPS
results are kept apart and never compared across hardware.

### Provenance and contention

Each result JSON contains: the Python, torch, CUDA and cuDNN versions; the versions of
every model library; GPU name and memory; the seed; the recipe; the spec; the weight
fingerprint; and a snapshot of **other GPU processes and utilisation** at the start and end.
The development workstation is shared, so a result with `contended: true` is labelled
*exploratory* for speed. Accuracy is unaffected by contention, but timing is not.

### Re-evaluating saved runs

`manyseg-bench evaluate <run.json> --variants ...` reloads a `--save-model` checkpoint (sha256
checked) with the run's own spec, split scheme and normalisation statistics, and scores it under
`fp32`/`bf16`/`fp16` (autocast on the GPU) and `int8_pt2e`, `int8_pt2e_compiled` and
`int8_weight_only` (CPU, `manyseg.quantize`). Every batch also goes through the fp32 reference
on the same device, so `agreement` is a paired pixel argmax agreement. It is defined only for
single full-resolution logits, and is `null` otherwise. A failing variant is recorded and does
not discard the others (E008).

### Remote execution (Pawsey Setonix)

Experiments emit their exact runs as task lines (`EMIT_TASKS`), which run as Slurm arrays
(`scripts/setonix/`). Datasets too large to copy once per task (CloudSEN12+, 191 GB) run as
8-GCD **pack** jobs that stage only the needed cache files once into the job's node-local NVMe.
A pack holds all its GCDs until its slowest run ends, so only runs of similar length are packed
together. Steps are named by task index. Usage is charged at 64 units per GCD-hour; costs are
forecast before submission.

## Statistics

* Important comparisons use at least 3 seeds and report mean ± standard deviation.
* A difference smaller than the seed standard deviation is reported as *no detectable
  difference*.
* Accuracy is compared against FLOPs, latency and parameters as Pareto fronts
  (`reporting.pareto_front`), not as one aggregate score.

## Datasets (selection)

| Tier | Dataset | Why |
| --- | --- | --- |
| Smoke / core | Oxford-IIIT Pet (3-class trimap, 256²) | Cheap natural images; boundary class. |
| Smoke / core | Sen1Floods11 hand-labelled (S1 + S2) | Multi-input SAR + optical; binary; has an event-level holdout (Bolivia). |
| Core | CloudSEN12+ high (S2 L1C, 4 classes) | Clouds, thin cloud and shadows; 13 bands; ROI-disjoint (geographic) splits; static auxiliary layers and acquisition metadata for fusion and conditioning. |
| Core | PASTIS single-date (S2 L2A, 10 bands, 19 classes, 128²) | Crop types from one acquisition; the date carries real signal (phenology), so metadata conditioning can be tested where it should matter; official folds. |
| Core | CamVid (11 classes, 480×360) | Small and thin objects; video sequences split. |
| Core | WHU Building aerial (binary, 512², 0.3 m) | Sharp boundaries, small buildings; official geographic split. |
| Core | MADOS (Sentinel-2, 15 classes) | Multispectral (11 bands), rare and small objects; sparse labels. |
| Core | LoveDA (7 classes, 1024²) | Domain shift between urban and rural; split by district. Test labels are hidden, so the official val serves as test. |
| Optional | ISPRS Vaihingen (IRRG + DSM) | Multimodal with complex boundaries. Download needs a password and redistribution is restricted. |

Licenses and access details are in `datasets/*.py` docstrings and in
`docs/architecture_review.md` §6.

### CloudSEN12+ high (`cloudsen12`, `cloudsen12_s2_aux`, `cloudsen12_meta`)

* **Source and licence.** CloudSEN12+ expert-labelled subset (Aybar et al. 2024,
  doi:10.1016/j.dib.2024.110852), from the TACO v3 repackaging
  `isp-uv-es/cloudsen12plus-l1c-high-taco`. Sentinel-2 L1C, 13 bands at 10 m, DN = 10000 ×
  TOA reflectance (stored as reflectance, float16). Classes: clear, thick cloud, thin cloud,
  cloud shadow. Licence: CC0-1.0 for labels and curation; the imagery carries the
  Copernicus Sentinel data notice ("Contains modified Copernicus Sentinel data 2017-2022"),
  which the dataset's licence string (recorded in every run) and any redistribution must
  state. `cloudsen12_s2_aux` adds S1GBM (CC BY 4.0) and Copernicus DEM GLO-30 attributions.
* **Splits.** The official train / validation / test (9,175 / 612 / 1,060 samples). They are
  ROI-disjoint (2,847 ROIs; `prepare` re-checks), so they are geographic and conditioning on
  latitude/longitude can be evaluated honestly: every test location is unseen.
* **Tiling and padding.** The 847 2048×2048 samples are cut into 16 non-overlapping 512×512
  tiles (`<sample_id>__t<row>_<col>`) that inherit the parent's split. Valid data is 509×509
  (or 2000×2000); the release labels the padding "clear", so padding is set to the ignore
  index (255) and NaN in the inputs.
* **Auxiliary layers (`cloudsen12_s2_aux`).** `s1gbm` (VV, VH in dB) is the **Sentinel-1
  Global Backscatter Model**, a static multi-year land-backscatter mosaic. It is *not* a
  Sentinel-1 acquisition coincident with the S2 image and cannot see the clouds; it can
  only describe the surface (e.g. help separate bright soil or snow from cloud, or water
  from shadow). `dem` is Copernicus GLO-30, relative to the tile median. Both come from the
  `cloudsen12-extra` TACO v1 blobs on the same 10 m grid (checked per sample; all 10,847
  matched). S1GBM has no data (stored 0, kept NaN, i.e. 0 after normalisation) on ~4% of
  valid pixels, and 3% of tiles have none at all. The release stores the signed DEM as
  uint16 (below-sea-level values wrap to ~65,000); it is reinterpreted as int16.
* **Leakage rules.** The extra blobs also contain `cloudmask_*` layers (QA60, Sen2Cor,
  s2cloudless, Cloud Score+, UNetMobV2 v1/v2, SEnSeI v2). They are never read: they are other
  cloud detectors' outputs, and UNetMobV2 was trained on CloudSEN12. The label-derived
  cloud percentages in the sample table and the socioeconomic `rai:*` columns of the extra
  index are never inputs; `load_vectors` refuses label-derived fields.
* **Metadata (`cloudsen12_meta`).** Tile-centre lat/lon (from the GeoTIFF georeferencing,
  UTM → WGS 84), day of year, solar zenith and azimuth angles.
* **Prepared.** Tiles train / val / test = 19,450 / 1,767 / 2,335 (685 / 77 / 85 parent
  2048² samples; 2,383 / 184 / 280 ROIs). Labelled-pixel class fractions (train): clear
  0.59, thick 0.23, thin 0.10, shadow 0.08; 4.9-8.5% of pixels are padding (ignored).
* **Storage.** The float16 cache (23,552 tiles, 16 channels, 190 GB) stays memory-mapped on
  the host (`device_resident: False`); batches are moved to the GPU on demand.

### PASTIS single-date (`pastis`, `pastis_meta`, `pastis_meta_shuffled`, `pastis_meta_tileshuffled`)

* **Source and licence.** PASTIS (Sainte Fare Garnot & Landrieu, ICCV 2021), Zenodo record
  5012942 (doi:10.5281/zenodo.5012942, `PASTIS.zip`, 28.8 GB, md5
  `cfc441bf18137ff0bbf4fad58828fb98`; `prepare` requires an `MD5_OK` line in
  `raw/pastis/download.log`). Licence CC BY 4.0; cite the paper. Imagery: THEIA L2A
  ("value-added data processed by the CNES for the Theia data cluster using Copernicus
  data"); labels from the French LPIS (IGN). The licence string is recorded in every run.
* **Data.** 2,433 patches, 128×128 px at 10 m, four Sentinel-2 tiles (T30UXV, T31TFJ,
  T31TFM, T32ULU). Each patch is a time series of 38–61 acquisitions (Sept 2018 – Nov 2019),
  10 bands in wavelength order (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12; 20 m bands
  bilinearly upsampled), stored as reflectance (DN / 1e4, float16). Labels: 0 background
  (kept as a class, as in the paper), 1–18 crop types, 19 void (out-of-nomenclature crops,
  parcels mostly outside the patch) → ignore (255). 35 S2 series in the release have no
  metadata or annotation and are skipped.
* **Single-date samples.** One sample per (patch, date) for 8 dates per patch: the season
  window 2018-11-01..2019-10-31 (365 days, so a cyclic day of year is unambiguous) is cut into
  8 equal parts, and the acquisition nearest each part's centre is taken. Every tile has one
  date list shared by its patches, so all patches of a tile get the same 8 dates. Names are
  `<patch>_<tile>_<YYYYMMDD>`.
* **Splits.** Official fold I: train folds 1–3 (1,455 patches, 11,640 samples), val fold 4
  (482, 3,856), test fold 5 (496, 3,968). All dates of a patch stay in its fold; the release
  keeps adjacent patches in one fold, and every fold has patches from all four tiles.
  `prepare` re-checks that no patch id crosses splits. Normalisation statistics come from
  train only (`meta.json`).
* **Metadata and leakage rules.** `pastis_meta` adds a vector input `meta` with `doy`
  (cyclic, period 365.25), parsed from the sample name. Two controls keep the doy marginal
  distribution and break its link to the image: `pastis_meta_shuffled` (permuted across the
  split's samples, fixed seed) and `pastis_meta_tileshuffled` (permuted within each tile).
  The second exists because each tile's dates are unique to it, so the exact doy identifies
  the tile and hence the region's crop mix; that control keeps the tile information and
  removes only the seasonal signal. The label-derived `N_Parcel` and `Parcel_Cover`
  properties of `metadata.geojson` are never read, and `load_vectors` refuses them.
* **Prepared.** Labelled-pixel class fractions (train): background 0.44, meadow 0.20, corn
  0.10, soft winter wheat 0.07, then 15 classes between 0.003 (potatoes) and 0.032
  (grapevine); 9–10% of pixels are void. Clouds were not filtered beyond THEIA's 90%
  tile-cover rule: 15% of samples have a patch-mean blue reflectance > 0.2 (mostly cloud,
  some snow), concentrated on a few tile dates.
* **Storage.** 6.3 GiB float16 cache, memory-mapped on the host (`device_resident: False`).
