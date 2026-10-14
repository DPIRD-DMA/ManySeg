# E004: Pretrained stem adaptation for 13-band Sentinel-2 (pre-registered)

Status: **complete** (2026-10-14; 18 runs on Pawsey Setonix, AMD MI250X, code `537848a`).

## Question
How should ImageNet RGB stem weights be adapted to 13 Sentinel-2 bands? The default `repeat`
(timm's rule) is compared with alternatives and with training from scratch.

## Protocol
* Data: Sen1Floods11 S2-only variant (`sen1floods11_s2`, 13 bands, reflectance/1e4, train-split
  normalisation). Official splits. Test and Bolivia are evaluated once per run.
* Model: ManySeg U-Net with resnet34 (`tv_in1k`). Recipe `fastai_like`, 60 epochs, batch 16.
  Seeds 0–2.
* Arms:
  | arm | channel adapter | notes |
  |---|---|---|
  | repeat | `repeat` | timm default: RGB filters cycled ×3/13 |
  | mean | `mean` | channel-constant response preserved |
  | rgb_zero | `rgb_plus_zero`, `rgb_indices=(3,2,1)` | exact RGB model on B4/B3/B2 at init |
  | rgb_random | `rgb_plus_random`, `rgb_indices=(3,2,1)` | RGB filters on B4/B3/B2, others random |
  | projection | `projection`, `rgb_indices=(3,2,1)` | learned 1×1 13→3 in front of the RGB stem |
  | scratch | none (random init) | lower bound |
* Metrics: water IoU and mIoU (val, test, Bolivia); convergence (val mIoU at epochs 10 and 30).
* Budget: 18 runs, about 1 h.

## Decision rule
Change the default adapter only if an arm beats `repeat` on val by more than 2 pooled
seed-SDs and points the same way on test.

## Results (water IoU, mean ± SD over 3 seeds; MI250X, torch 2.7.1/ROCm 6.3)

| adapter | val water IoU | test water IoU | Bolivia water IoU | val mIoU at epoch 10 | Welch t vs repeat (val / test) |
|---|---|---|---|---|---|
| repeat (default) | 0.8109 ± 0.0069 | 0.8104 ± 0.0094 | 0.7830 ± 0.0090 | 0.8222 ± 0.0348 | |
| scratch (no pretraining) | 0.8108 ± 0.0086 | 0.8105 ± 0.0063 | **0.8083 ± 0.0069** | 0.8166 ± 0.0417 | −0.0 / +0.0 |
| rgb_random (B4/B3/B2) | 0.8052 ± 0.0178 | 0.8092 ± 0.0143 | 0.7215 ± 0.0516 | **0.8374 ± 0.0240** | −0.5 / −0.1 |
| rgb_zero (B4/B3/B2) | 0.8032 ± 0.0065 | 0.8092 ± 0.0089 | 0.7134 ± 0.0540 | 0.8279 ± 0.0075 | −1.4 / −0.2 |
| mean | 0.7971 ± 0.0127 | 0.8141 ± 0.0061 | 0.7565 ± 0.0145 | 0.8297 ± 0.0145 | −1.7 / +0.6 |
| projection | 0.7520 ± 0.0075 | 0.7205 ± 0.0315 | 0.7407 ± 0.0133 | 0.6782 ± 0.0052 | **−10.0 / −4.7** |

## Interpretation
* **No adapter beats `repeat`** under the pre-registered rule, so the default does not change.
* **ImageNet initialisation buys nothing at the end of training here.** Training from scratch
  equals `repeat` on val and test, and is the best arm on the Bolivia holdout (+2.5 points; n = 15,
  indicative only). With 252 training chips and 60 epochs, the 13-band S2 encoder learns its
  own stem. RGB pretraining helps only early on: the RGB-preserving adapters converge fastest
  (best epoch-10 mIoU) but end level.
* **`projection` is clearly harmful** (learned 13→3 bottleneck in front of a frozen-shape RGB
  stem). We recommend against it for multispectral inputs.
* RGB-preserving adapters have the most seed-variable Bolivia scores (SD 0.05), consistent with
  them anchoring on visible bands that shift between flood events.

## Decision
Keep `repeat` (timm-compatible) as the default. Document that `projection` is not recommended
for multispectral inputs and that scratch training is competitive on Sen1Floods11-sized S2
data. Caveat: one dataset, resnet34. A multispectral-pretrained encoder (e.g. SSL4EO) is the
natural follow-up comparison.
