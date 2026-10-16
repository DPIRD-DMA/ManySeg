# E022: Binary change detection: early fusion vs Siamese fusions (pre-registered)

Status: **planned** (registered 2026-10-16 before any run; epochs set by the timing rule below
from 1-epoch smoke runs outside `runs/`).

## Question
ManySeg can build change-detection models from two inputs (`t1`, `t2`). The simplest is early
fusion: both dates stacked as channels in one input (FC-EF). Siamese models encode each date
with one shared encoder (`"backbone": "shared:t1", "share_norm": true`) and fuse per level
with `concat` (FC-Siam-conc), the new `diff` (`|f1 - f2|`, FC-Siam-diff) or `concat_diff`
(`[f1, f2, |f1 - f2|]`). Which should the docs recommend?

## Data
* **OSCD** (`oscd`): urban change, Sentinel-2 L1C 13 bands at 10 m, official 14 train / 10 test
  cities, 3 train cities held out as val (`OSCD_VAL`, sha1-chosen before any run); 128×128
  tiles: 390 / 106 / 230; 2.4% / 1.8% / 5.2% changed pixels. Small and imbalanced: expect
  noisy results.
* **LEVIR-CD** (`levircd`): building change, 0.5 m RGB, official 445 / 64 / 128 pairs of
  1024×1024, tiled 256×256 (7,120 / 1,024 / 2,048).

## Protocol
* ResNet-34 (`tv_in1k`) U-Net, recipe `fastai_like`, batch 16, seeds 0–2, local RTX 4090. The
  stacked arm adapts the RGB stem to 2C channels (`repeat`); Siamese arms use it per date.
* Arms (`--tag e022-<dataset>-<arm>`): `stack`, `siam-concat`, `siam-diff`, `siam-concat_diff`.
  2 datasets × 4 arms × 3 seeds = 24 runs.
* Epochs: OSCD 100 (390 tiles); LEVIR-CD the largest of 30 / 20 / 15 for which its 12 runs fit
  in 6 h at the smoke-timed rate (recorded here before any LEVIR-CD run). **Smoke:** frozen
  first epoch 19 s (`stack`) to 26 s (`siam-concat_diff`), so ~40 s per full epoch, 30 epochs
  × 12 runs ≈ 4 h: **30 epochs**.
* Metric: change-class F1 on val (test F1, IoU, precision, recall reported).

## Decision rule
* A Siamese arm *beats* `stack` if its val F1 is higher by more than 2 pooled SD on **both**
  datasets with test agreeing. The best such arm is the documented recommendation; otherwise
  the docs say early fusion is as good on these data.
* `diff` or `concat_diff` is recommended over `siam-concat` only if it beats it by 2 pooled SD
  on both datasets; otherwise the simpler `concat` is the documented Siamese default.
