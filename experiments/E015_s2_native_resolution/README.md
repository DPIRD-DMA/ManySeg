# E015: Native-resolution Sentinel-2 band groups vs the 10 m stack (pre-registered)

Status: **planned** (registered 2026-10-15, before any run).

## Question
Every experiment so far feeds the 13 S2 bands as one 10 m stack, with the 20 m and 60 m bands
resampled onto the 10 m grid, as the datasets ship them. Do inputs at native resolution do
better, or as well for less compute? These are one input per resolution group, aligned by ground
extent (`same_extent`): 10 m (B2, B3, B4, B8), 20 m (B5, B6, B7, B8A, B11, B12) and
60 m (B1, B9, B10).

## Protocol
* Local RTX 4090 (no Pawsey units). Sen1Floods11 S2 only (`sen1floods11_s2`, 13 bands),
  U-Net, ResNet-34 `tv_in1k`, recipe `fastai_like`, 60 epochs, batch 16. Seeds 0–2. Test and
  Bolivia (held-out event) are evaluated once per run.
* Every arm sees the **same stacked 10 m data** from the loader. The groups arms split it
  inside the model wrapper (`band_groups`, bench provider) and block-average the 20 m and 60 m
  groups by 2 and 6, recreating native-like resolutions. This is close to, but not identical to,
  the original L1C products. With ceil pooling, a 256-px training crop gives a 43-px 60 m grid,
  so its extent is 1% larger than the crop. This is a stated approximation.
* Arms (9 runs, about 30–45 min):

  | Tag | Model |
  | --- | --- |
  | `e015-stack` | baseline: one 13-channel input (`repeat` stem adapter) |
  | `e015-groups` | 3 inputs, an independent ResNet-34 each, intermediate concat fusion |
  | `e015-groups-shared` | 3 inputs, one shared encoder; the 20 m and 60 m inputs reach it through `ProjectionAdapter`s (E004 found projection harmful; a stated confound) |

* Reported: val, test and Bolivia mIoU, boundary F1, params, FLOPs, train throughput.

## Decision rule
* A groups arm is **better** if val mIoU − stack > 2 pooled SD and test agrees. It is **as good
  for less** if within 1 pooled SD with fewer FLOPs.
* **CloudSEN12+ follow-up (Pawsey):** submitted only if a groups arm is better here, or as good
  for at least 20% fewer FLOPs. Otherwise the result is reported as negative and no Pawsey units
  are spent.
* The library gets a `band_groups`-style input helper only if a groups arm is better.

## Amendment (2026-10-15 13:20, diagnostic; after seed 0 of the shared arm)
`e015-groups-shared` seed 0 collapsed at evaluation (val 0.56, water IoU 0.21) while its
training loss fell normally (0.056). Hypothesis: the shared encoder's BatchNorm running
statistics blend three inputs with very different statistics (10 m bands, plus the projected 20 m
and 60 m groups). Training normalises each pass with its own batch statistics, but evaluation
applies the blend to every group. Diagnostic arm `e015-groups-shared-frozenbn` (the same model
with `freeze_encoder_bn=True`, so the pretrained statistics are never blended), seeds 0–2. It is
not part of the pre-registered comparison. It tests the hypothesis, and so a possible pitfall in
`"backbone": "shared:..."` across heterogeneous inputs.

## Amendment 2 (2026-10-15 13:28, after the pre-registered arms; before any CloudSEN12+ spend)
`e015-groups` passed the rule (val +0.83 vs 2 pooled SD 0.68; test +1.39), but it has three
encoders (68 M vs 24.5 M parameters), so capacity and native resolution are confounded. Control
arm `e015-groups-10m`: the same three independent encoders and band split, every group at
10 m (factor 1). Only the resolution differs from `e015-groups`. If `groups-10m` ≈ `groups`,
the gain is capacity, not native resolution, and no CloudSEN12+ follow-up is run. Seeds 0–2,
local 4090.
