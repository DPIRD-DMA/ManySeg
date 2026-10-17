# E024: ImageNet vs remote-sensing pretrained weights (pre-registered)

Status: **planned** (registered 2026-10-17 before any run; epochs below; timing from 1-epoch smoke
runs outside `runs/`).

## Question
`manyseg.weights` loads Sentinel-2 pretrained encoders (SSL4EO-S12 MoCo / DINO, SatlasPretrain)
into timm architectures. Do they beat the default ImageNet weights when fine-tuned on Sentinel-2
tasks? Users ask this first; published comparisons are mixed.

## Giving each set of weights its best chance (normalisation)
Each arm is fed inputs scaled exactly as in its own pretraining (checked in the authors' code,
2026-10-17), not with the dataset statistics used for the ImageNet arm:

* **SSL4EO-S12 S2** (MoCo, DINO): pretraining (`pretrain_*_s2c.py`, launched with
  `--dtype uint8` and without `--normalize`) used DN / 10000 → 8-bit → `ToTensor`, i.e.
  reflectance clipped to [0, 1], no mean/std step (`Normalize` commented out). Here: cached
  reflectance (DN / 1e4) clipped to [0, 1]; 13 L1C bands in the order B01…B08, B8A, B09–B12.
* **SatlasPretrain S2 MS** (`si_ms`): Satlas `Normalization.md`: TCI (B04, B03, B02; ESA's 8-bit
  true-colour product) / 255 and B05–B08, B11, B12 / 8160, clipped to [0, 1]. ESA's L1C TCI
  saturates at reflectance 0.2 (DN 2000), so RGB = DN / 2000 clipped stands in for TCI / 255.
* **ImageNet** (`tv_in1k`): the bench default: per-band dataset mean/std, `repeat` stem adapter.

Bands a dataset lacks are filled with the pretraining mean (0 after normalisation); the run
records which. Pretraining was on L1C; BigEarthNet and PASTIS are L2A (surface reflectance),
a mismatch every user of these weights on L2A data faces.

| dataset | bands | SSL4EO missing | Satlas missing |
|---|---|---|---|
| Sen1Floods11 (`sen1floods11_s2`, L1C) | 13 | — | — |
| BigEarthNet v2 (`bigearthnet`, L2A) | 12 | B10 | — |
| PASTIS (`pastis`, single date, L2A) | 10 | B01, B09, B10 | — |

## Protocol
* ResNet-50 for every arm (the architecture all four weights exist for); U-Net decoder for the
  dense tasks; image head (no decoder) for BigEarthNet. Recipe `fastai_like` (1 frozen epoch,
  encoder LR ×0.1, one-cycle), bf16, seeds 0–2, local RTX 4090 (no Pawsey).
* Sen1Floods11 S2 only: 60 epochs, batch 16 (E018). BigEarthNet: 12 epochs, batch 64 (E020;
  multi-label, macro mAP). PASTIS single date: 30 epochs, batch 16 (E016; mIoU).
* Arms (`--tag e024-<dataset>-<arm>`): `imagenet`, `ssl4eo_moco`, `ssl4eo_dino`, `satlas`.
  3 datasets × 4 arms × 3 seeds = 36 runs.
  **Smoke (frozen first epoch):** Sen1Floods11 ~10 s, BigEarthNet ~21 s, PASTIS ~24 s per epoch
  (all arms similar) → about 3.2 h + 1.4 h + 4 h ≈ 8.5 h of 4090 time.
* Metric: val mIoU (Sen1Floods11, PASTIS) or val macro mAP (BigEarthNet); test reported.

## Decision rule
* A remote-sensing weight is **recommended** if it beats `imagenet` by more than 2 pooled SD of
  the val metric on at least 2 of the 3 datasets, with test agreeing on those, and is not worse
  than `imagenet` by more than 2 pooled SD on the third.
* If none qualifies, the docs say ImageNet weights were as good on these tasks.
* "Test agreeing" is a veto only; arms are compared on validation.
