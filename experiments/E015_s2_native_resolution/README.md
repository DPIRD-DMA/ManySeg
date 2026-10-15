# E015: Native-resolution Sentinel-2 band groups vs the 10 m stack (pre-registered)

Status: **complete** (2026-10-15; 18 runs, local RTX 4090, no Pawsey units). **Native resolution itself adds nothing, but it halves the compute of a grouped model. Splitting the bands into groups is what helps (+0.9 to +1.3 mIoU). The shared-encoder arm exposed a library bug (BatchNorm blended across inputs), now fixed.**

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

## Amendment 3 (2026-10-15 13:36, library fix verification)
The diagnostic confirmed the BN-blending hypothesis: `e015-groups-shared-frozenbn` scored
0.9029 / 0.9043 / 0.9049 val, against 0.50–0.65 without frozen BN. The library now gives
`shared:` encoders per-input BatchNorm by default (commit `0a4025d`). Verification arm
`e015-groups-shared-perbn`: the shared arm unchanged except for the new default (BN statistics
still update, per input), seeds 0–2, local 4090.

## Results (Sen1Floods11 S2, mean ± seed SD, 3 seeds)

| Arm | n | Val mIoU | Test mIoU | Bolivia | Params | GFLOPs | Train img/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `e015-stack` (13-band 10 m stack) | 3 | 0.8899 ± 0.0045 | 0.8909 ± 0.0107 | 0.8590 | 24.5 M | 66.7 | 521 |
| `e015-groups` (3 encoders, native resolution) | 3 | 0.8982 ± 0.0015 | 0.9049 ± 0.0026 | 0.8553 | 68.1 M | 77.4 | 229 |
| `e015-groups-10m` (3 encoders, all at 10 m; amendment 2) | 3 | 0.8987 ± 0.0008 | 0.9015 ± 0.0049 | 0.8598 | 68.1 M | 144.0 | 255 |
| `e015-groups-shared` (1 encoder, BN shared; old library) | 3 | 0.5715 ± 0.0741 | 0.5633 ± 0.1021 | 0.4453 | 25.5 M | 77.3 | 252 |
| `e015-groups-shared-frozenbn` (diagnostic; amendment 1) | 3 | 0.9040 ± 0.0010 | 0.9052 ± 0.0042 | 0.8473 | 25.5 M | 77.3 | 251 |
| `e015-groups-shared-perbn` (fixed library default; amendment 3) | 3 | 0.9024 ± 0.0033 | 0.9038 ± 0.0073 | 0.8616 | 25.5 M | 77.3 | 247 |

Comparisons (val Δ in points vs 2 pooled SD; test Δ):

* groups − stack: +0.83 (0.68), test +1.39. **Passes the pre-registered rule.**
* groups − groups-10m: −0.05 (0.24), test +0.34. **Native resolution adds no accuracy.**
* groups-10m − stack: +0.88 (0.65), test +1.06. **The band split is the effect.**
* shared-perbn − stack: +1.25 (0.80), test +1.29, at 25.5 M parameters (+4%).
* shared-perbn − shared-frozenbn: −0.16 (0.49). The library fix matches the diagnostic.

## Decision
* **Pre-registered rule:** the groups arm is better than the stack. **The capacity control
  (amendment 2) attributes the gain**: the same three encoders on 10 m groups score the same,
  so the gain comes from splitting the 13 bands into resolution groups with their own stems and
  encoders, not from native resolution. Native resolution does buy compute: 77 vs 144 GFLOPs for
  the same accuracy (−46%). It is still 16% more than the single stack.
* **Best model:** one shared encoder over the three groups, with per-input BatchNorm (now the
  library default for `shared:` encoders). +1.25 val / +1.29 test over the stack for +4%
  parameters and +16% FLOPs. Throughput halves (three encoder passes).
* **CloudSEN12+ follow-up: not run.** Its trigger was the native-resolution hypothesis, which the
  control refutes. Band grouping is a *new* hypothesis that was not pre-registered. It should be
  tested on a second dataset before any recommendation: PASTIS (S2, 10 bands at 10/20 m, local,
  no Pawsey units) is being prepared for E016 and can host it.
* **Library:** no band-group helper yet (the bench's `band_groups` stays a bench option) until
  the grouping effect replicates. The **construction bug** is fixed (commit `0a4025d`):
  `shared:` encoders used to share BatchNorm, which blends running statistics across inputs with
  different statistics and collapsed evaluation (0.57 val). They now keep BN per input, and
  `"share_norm": True` restores full sharing.
* Earlier results are unaffected: no earlier experiment used a shared encoder.
