# E004: Pretrained stem adaptation for 13-band Sentinel-2 (pre-registered)

Status: **planned** (registered 2026-10-14 before any E004 run).

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
