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
| Core | CamVid (11 classes, 480×360) | Small and thin objects; video sequences split. |
| Core | WHU Building aerial (binary, 512², 0.3 m) | Sharp boundaries, small buildings; official geographic split. |
| Core | MADOS (Sentinel-2, 15 classes) | Multispectral (11 bands), rare and small objects; sparse labels. |
| Core | LoveDA (7 classes, 1024²) | Domain shift between urban and rural; split by district. Test labels are hidden, so the official val serves as test. |
| Optional | ISPRS Vaihingen (IRRG + DSM) | Multimodal with complex boundaries. Download needs a password and redistribution is restricted. |

Licenses and access details are in `datasets/*.py` docstrings and in
`docs/architecture_review.md` §6.

### CloudSEN12+ high (`cloudsen12`, `cloudsen12_s2_aux`, `cloudsen12_meta`)

* **Source and licence.** CloudSEN12+ expert-labelled subset (Aybar et al. 2024,
  doi:10.1016/j.dib.2024.110852), CC0-1.0, from the TACO v3 repackaging
  `isp-uv-es/cloudsen12plus-l1c-high-taco`. Sentinel-2 L1C, 13 bands at 10 m, DN = 10000 ×
  TOA reflectance (stored as reflectance, float16). Classes: clear, thick cloud, thin cloud,
  cloud shadow.
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
  `cloudsen12-extra` TACO v1 blobs on the same 10 m grid (checked per sample).
* **Leakage rules.** The extra blobs also contain `cloudmask_*` layers (QA60, Sen2Cor,
  s2cloudless, Cloud Score+, UNetMobV2 v1/v2, SEnSeI v2). They are never read: they are other
  cloud detectors' outputs, and UNetMobV2 was trained on CloudSEN12. The label-derived
  cloud percentages in the sample table and the socioeconomic `rai:*` columns of the extra
  index are never inputs; `load_vectors` refuses label-derived fields.
* **Metadata (`cloudsen12_meta`).** Tile-centre lat/lon (from the GeoTIFF georeferencing,
  UTM → WGS 84), day of year, solar zenith and azimuth angles.
* **Storage.** The float16 cache (23,552 tiles, 16 channels, ~200 GB) stays memory-mapped on
  the host (`device_resident: False`); batches are moved to the GPU on demand.
