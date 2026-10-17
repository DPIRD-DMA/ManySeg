# E024: ImageNet vs remote-sensing pretrained weights (pre-registered)

Status: **done** (registered 2026-10-17 before any run; all 36 runs launched from a frozen
worktree at `9a2d519` on the local RTX 4090, no Pawsey units; the run JSONs do not record the
commit or GPU). Result: **with each weight fed its own pretraining normalisation, both SSL4EO-S12
weights (MoCo, DINO) beat the ImageNet baseline on all three datasets and are recommended;
SatlasPretrain is not** (better on Sen1Floods11 only, slightly worse on PASTIS).

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

## Results

All 36 runs completed (no failures). Final epoch, mean ± SD over seeds 0–2; full tables:
`uv run python experiments/E024_rs_pretrained_weights/analyse.py`.

| arm | Sen1Floods11 val mIoU | test | BigEarthNet val mAP | test | PASTIS val mIoU | test |
|---|---|---|---|---|---|---|
| imagenet | 0.8882 ± 0.0038 | 0.8853 | 0.7381 ± 0.0031 | 0.6624 | 0.2097 ± 0.0022 | 0.2073 |
| ssl4eo_moco | 0.9061 ± 0.0034 | 0.8989 | 0.7510 ± 0.0002 | 0.6774 | 0.2211 ± 0.0029 | 0.2190 |
| ssl4eo_dino | **0.9095 ± 0.0016** | 0.8996 | **0.7788 ± 0.0011** | **0.6995** | **0.2305 ± 0.0017** | **0.2277** |
| satlas | 0.8999 ± 0.0027 | **0.9070** | 0.7338 ± 0.0016 | 0.6621 | 0.2060 ± 0.0012 | 0.2028 |

Differences from `imagenet` in points (val, with 2 pooled SD; test):

| arm | Sen1Floods11 | BigEarthNet | PASTIS | decision |
|---|---|---|---|---|
| ssl4eo_moco | +1.79 (0.71); +1.36 | +1.29 (0.44); +1.50 | +1.15 (0.51); +1.17 | **recommended** (3/3) |
| ssl4eo_dino | +2.13 (0.58); +1.43 | +4.07 (0.47); +3.71 | +2.09 (0.40); +2.05 | **recommended** (3/3) |
| satlas | +1.17 (0.66); +2.18 | −0.43 (0.50); −0.03 | −0.37 (0.36); −0.45 | not recommended |

* **Decision:** SSL4EO MoCo and DINO each beat ImageNet by more than 2 pooled SD on all three
  datasets with test agreeing (the rule needed two of three). SatlasPretrain beats it on
  Sen1Floods11 only and is worse than ImageNet on PASTIS by just over 2 SD (−0.37 against
  0.36), so it fails the "not worse on the third" condition too.
* **DINO ≥ MoCo** on every dataset (+0.3, +2.8, +0.9 val); not a registered comparison.
* **Missing bands** (filled with the pretraining mean): SSL4EO lacked B10 on BigEarthNet and
  B01, B09, B10 on PASTIS and still won there; Satlas had all its bands everywhere.
  The protocol said each run records them; the runner did not save the provider's notes
  (fixed after this experiment: runs now carry `model_notes`), so `analyse.py` derives them
  from the recorded `band_names` and the registry, which is equivalent.
* **PASTIS absolute numbers** are low because `pastis` is E016's single-date task over all
  eight dates of the season, cloudy winter scenes included (E016's ResNet-34 baseline:
  0.199 val); E021's single-date arm used one July date (0.326).
* **Throughput:** no consistent difference on the dense tasks; BigEarthNet timings ranged
  2,059–2,373 img/s across arms and are noisy (shared GPU; not a timing study).

## Caveats

* One architecture (ResNet-50), one recipe, one fine-tuning schedule per dataset; full
  fine-tuning after one frozen warm-up epoch (no frozen-encoder or few-label setting, where
  pretraining usually matters more).
* Each arm is weights plus their own input normalisation (the experiment's design), so the gain
  belongs to that combination; whether SSL4EO weights still win with dataset statistics, or
  ImageNet weights gain from another normalisation, was not tested.
* Satlas RGB was fed as reflectance / 0.2 clipped, standing in for ESA's 8-bit TCI product. If
  the TCI rendering differs materially (gamma, clipping), Satlas was disadvantaged; untested.
* Pretraining was on L1C; BigEarthNet and PASTIS are L2A. SSL4EO won despite this.

## Docs

* `api_design.md` (Remote-sensing pretrained weights): evidence note; SSL4EO DINO / MoCo
  recommended for Sentinel-2 ResNet-50 encoders, Satlas not.
* `README.md`: tip updated.

