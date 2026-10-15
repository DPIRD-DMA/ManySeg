# E019: Image classification parity: ManySeg image heads vs plain timm (pre-registered)

Status: **planned** (registered 2026-10-15, before any run).

## Question
ManySeg can build classifiers and regressors: an encoder plus an image-scope head (global
average pool of the deepest feature, then a linear layer), with no decoder. Before we
recommend it for image-level tasks, does it match a plain `timm.create_model(...,
num_classes=K, in_chans=C)` classifier with the same encoder, the same pretrained weights
(verified identical, also for 13-band stems: `test_pretrained_weights_identical_for_image_level_providers`)
and the same recipe? A gap would be a ManySeg defect to explain or fix before E020.

Known structural difference: timm's ConvNeXt head applies a pretrained LayerNorm after
pooling; ManySeg's image head pools the encoder's last stage directly. ResNet-50 and PVTv2-b0
heads are pool + linear in both libraries.

## Protocol
* Providers: `manyseg` (`output="image"`) and `timm`. Local RTX 4090, no Pawsey units.
* Encoders: ResNet-50 (`tv_in1k`), ConvNeXt-T (`fb_in1k`), PVTv2-b0 (`in1k`, the E018 preset
  encoder).
* Datasets:
  * EuroSAT-MS, 13 bands, 64×64, TorchGeo spatial split (16,200 / 5,400 / 5,400), 30 epochs,
    batch 64, random-resized crops (scale 0.5–1) with h/v flips;
  * Oxford-IIIT Pet breeds, 37 classes, RGB 256×256 (the `oxford_pet` cache and split),
    20 epochs, batch 32, random-resized crops with h flips.
* Recipe `fastai_like` (1 frozen epoch, encoder LR ×0.1, one-cycle, AdamW wd 0.01, bf16).
  Seeds 0–2. 2 providers × 3 encoders × 2 datasets × 3 seeds = 36 runs, about 2.5 h (smoke: ~12 s per ResNet-50 EuroSAT epoch).
* Metric: val accuracy (test and macro F1 reported).

## Decision rule
* **Parity** for an encoder × dataset cell if |ManySeg − timm| ≤ 2 pooled SD of val accuracy
  (or ≤ 0.3 points when the pooled SD is smaller than 0.15, so seed-noise-free cells are not
  failed on hundredths).
* All six cells at parity: ManySeg image heads are documented as equivalent to timm
  classifiers and E020 proceeds with them.
* A cell outside parity: explain it (e.g. the ConvNeXt head norm), fix ManySeg if the gap is
  a ManySeg defect, re-run that cell, and report both results.
