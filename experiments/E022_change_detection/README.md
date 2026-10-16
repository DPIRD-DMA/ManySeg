# E022: Binary change detection: early fusion vs Siamese fusions (pre-registered)

Status: **done** (registered 2026-10-16 before any run; run at code `fb1d90b`). Result: Siamese
encoders beat early fusion on both datasets; no difference operator beats plain `concat` on
both, so `concat` is the documented Siamese default.

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

## Notes added during the run (2026-10-16, Codex review; no protocol change)
* Not compute-matched: Siamese arms run the shared encoder twice (OSCD ~6.8–7.0 GFLOPs/tile vs
  4.5 for `stack`), so a Siamese win is an architecture result, not a pure fusion-operator one.
  GFLOPs and parameters are reported per arm. The three Siamese arms are comparable to each
  other (fusion widths and parameter counts differ slightly).
* "Test agreeing" is a veto only: arms are compared and chosen on validation.

## Results

All 24 runs completed (local 4090, no failures; RAM never below 95 GB available). Change-class
F1, final epoch, mean ± SD over seeds 0–2.

| dataset | arm | val F1 | test F1 | test IoU | GFLOPs | params |
|---|---|---|---|---|---|---|
| OSCD | stack | 0.2593 ± 0.0381 | 0.2785 | 0.1619 | 4.5 | 24.5 M |
| OSCD | siam-concat | 0.3032 ± 0.0047 | 0.4112 | 0.2589 | 7.0 | 25.2 M |
| OSCD | siam-diff | 0.4608 ± 0.0279 | 0.4817 | 0.3173 | 6.8 | 24.5 M |
| OSCD | siam-concat_diff | 0.3640 ± 0.0369 | 0.4625 | 0.3009 | 7.0 | 25.5 M |
| LEVIR-CD | stack | 0.8883 ± 0.0007 | 0.8870 | 0.7970 | 16.0 | 24.4 M |
| LEVIR-CD | siam-concat | 0.9069 ± 0.0018 | 0.9031 | 0.8233 | 25.8 | 25.1 M |
| LEVIR-CD | siam-diff | 0.8998 ± 0.0020 | 0.8916 | 0.8043 | 25.2 | 24.4 M |
| LEVIR-CD | siam-concat_diff | 0.9083 ± 0.0018 | 0.9031 | 0.8233 | 26.0 | 25.5 M |

(GFLOPs per tile: OSCD 128², LEVIR-CD 256².)

* **Siamese vs early fusion (rule 1): met.** `siam-diff` (+20.2 val on OSCD, 2 SD 6.7; +1.15 on
  LEVIR-CD, 2 SD 0.30) and `siam-concat_diff` (+10.48, 2 SD 7.50; +2.00, 2 SD 0.27) beat `stack` on both datasets with test agreeing. `siam-concat` beats it on
  LEVIR-CD (+1.86, 2 SD 0.27) but not on OSCD (+4.40, 2 SD 5.42; test +13.3). Recommendation:
  a Siamese encoder (`"backbone": "shared:t1", "share_norm": true`) over stacking the dates.
* **Difference operators vs `concat` (rule 2): not met.** `siam-diff` beats `concat` on OSCD
  (+15.8, 2 SD 4.0) but is significantly worse on LEVIR-CD (−0.71, 2 SD 0.38; test −1.15).
  `siam-concat_diff` beats it on OSCD (+6.1, 2 SD 5.3) and ties on LEVIR-CD (+0.14, 2 SD 0.35;
  test ±0.00). Per the registered rule, plain `concat` stays the documented Siamese default.
* **Post hoc (not pre-registered):** `concat_diff` is the only arm that never lost to another
  arm it was compared with; `diff` is strongest on the small, noisy, 2%-change Sentinel-2 data
  and weakest of the Siamese arms on 0.5 m RGB. The best operator is data-dependent; the docs
  present `concat` as the default and the other two as options to try, with these numbers.
* **Caveats:** Siamese arms run the encoder twice (~1.5× FLOPs). OSCD is tiny (390 training
  tiles, 3 val cities) and its val/test spreads are large; its val–test gaps (e.g. `siam-concat`
  0.30 vs 0.41) reflect the city split. ResNet-34 only.
