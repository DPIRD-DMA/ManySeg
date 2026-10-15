# E019: Image classification parity: ManySeg image heads vs plain timm (pre-registered)

Status: **done** (registered 2026-10-15 before any run; run at code `1da0559`, E019b at `30a1364`). Result: **parity** in all six cells on val; ManySeg is 0.43 points lower on test on average (ConvNeXt most), and timm-style head init (E019b) narrows it to 0.27 but misses its adoption bar, so the default is unchanged.

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

## E019b: image-head initialisation (pre-registered follow-up, before any E019b run)

E019 passed its val parity rule in all six cells, but ManySeg's **test** accuracy was lower in
5 of 6 cells (mean −0.43 points; per cell 1.1–2.5 pooled SD). Known differences: timm
initialises the PVTv2 and ConvNeXt classifiers with normal(0, 0.02) weights and zero bias,
while ManySeg's image head used PyTorch's default Linear init (std 0.036 and a random bias for
PVTv2's 256 features); ResNet-50, where both use the default init, is the one cell where
ManySeg was ahead on test (+0.79). ConvNeXt additionally has timm's pretrained head LayerNorm.

* Arm `manyseg-init02`: ManySeg with `{"head": {"init_std": 0.02}}` (normal(0, 0.02) weights,
  zero bias), all 3 encoders × 2 datasets × seeds 0–2 (18 runs, ~35 min, local 4090). The
  timm arms are reused from E019.
* **Decision:** `init_std=0.02` becomes the image-head default if, over the six cells, the mean
  ManySeg − timm test gap moves from −0.43 to within ±0.20 points and the mean val gap is no
  worse than E019's (−0.16). Otherwise the default stays and the gap is documented. Per-cell
  results are reported either way (ConvNeXt's head norm is not addressed by this arm).

## Results

36 E019 runs + 18 E019b runs, no failures (local 4090). Val/test accuracy, mean of seeds 0–2;
gap = ManySeg − timm in points; bar = the parity threshold. Full tables:
`uv run python experiments/E019_classification_parity/analyse.py`.

| dataset | encoder | val ManySeg / timm | val gap (bar) | test gap (pooled SD) | E019b val gap | E019b test gap |
|---|---|---|---|---|---|---|
| EuroSAT-MS | ResNet-50 | 0.9574 / 0.9578 | −0.04 (1.06) | +0.79 (0.38) | −0.31 | +0.59 |
| EuroSAT-MS | ConvNeXt-T | 0.9627 / 0.9653 | −0.26 (0.70) | −0.86 (0.43) | −0.13 | −0.78 |
| EuroSAT-MS | PVTv2-b0 | 0.9566 / 0.9570 | −0.04 (0.64) | −0.79 (0.57) | +0.01 | −0.24 |
| Pet breeds | ResNet-50 | 0.9264 / 0.9264 | +0.00 (1.11) | −0.49 (0.43) | −0.09 | −0.12 |
| Pet breeds | ConvNeXt-T | 0.9504 / 0.9566 | −0.62 (1.87) | −0.76 (0.31) | +0.18 | −0.60 |
| Pet breeds | PVTv2-b0 | 0.9122 / 0.9122 | +0.00 (2.71) | −0.49 (0.37) | −0.44 | −0.45 |
| **mean** | | | **−0.16** | **−0.43** | **−0.13** | **−0.27** |

* **Decision (E019):** all six cells are at parity on val. ManySeg image heads are documented
  as equivalent to timm classifiers, with the test caveat below, and E020 proceeds.
* **Test caveat:** ManySeg was lower on test in 5 of 6 cells (−0.43 on average; 1.1–2.5 pooled
  SD per cell). Same parameters (ConvNeXt: timm has 1,536 more, its head LayerNorm), same
  pretrained weights (fingerprint test), same throughput (±6%).
* **Decision (E019b):** timm's classifier init (`{"head": {"init_std": 0.02}}`) moved the mean
  test gap to −0.27, outside the pre-registered ±0.20, so it stays an option and the default
  init is unchanged. It closed most of the PVTv2 EuroSAT and ResNet-50 Pet gaps; the remaining
  gap is mostly ConvNeXt (−0.78, −0.60), consistent with timm's pretrained pre-logits
  LayerNorm, which ManySeg's image head (pool + linear on the encoder's last stage) does not
  have.
* **Not tested here:** heads with a norm before the linear layer; larger datasets; other
  recipes. For ConvNeXt classification today, plain timm is marginally better on test.
