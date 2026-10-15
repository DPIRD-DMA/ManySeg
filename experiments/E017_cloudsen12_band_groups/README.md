# E017: Do Sentinel-2 band groups help cloud masking? (CloudSEN12+; pre-registered)

Status: **done, negative** (registered 2026-10-15 before any run; Setonix job 50937758 at code `d0edcb9`). Band groups do not pass on CloudSEN12+; grouping stays a single-dataset result (Sen1Floods11).

## Question
E015 (Sen1Floods11) found that feeding the 13 S2 bands as three resolution groups (10 m, 20 m,
60 m) through **one shared encoder** with per-input BatchNorm beat the 13-band stack: +1.25 val
and +1.29 test mIoU, for +4% parameters and +16% FLOPs. A capacity control showed the gain is the
band split, not native resolution. E016 (PASTIS) did not replicate it (+0.13), but PASTIS has
only 10 bands, no 60 m group, and pre-upsampled 20 m bands. CloudSEN12+ has the full 13 L1C bands,
like Sen1Floods11, and is the deciding dataset.

## Protocol
* Pawsey Setonix MI250X, **one 6-GCD pack job** (both arms on the same node and code). U-Net,
  ResNet-34 `tv_in1k`, recipe `fastai_like`, 20 epochs, batch 16, as E013. Seeds 0–2.
* Arms: `e017-stack` (the `cloudsen12` 13-band stack, i.e. E013's `e013-s2`, re-run on the same
  code and node as the groups arm) and `e017-groups-shared` (bench `band_groups`: 10 m B2 B3 B4 B8,
  20 m B5 B6 B7 B8A B11 B12 block-averaged ×2, 60 m B1 B9 B10 ×6; one shared encoder with
  per-input BatchNorm, the library default since `0a4025d`; intermediate concat fusion).
* Budget: 6 runs. Forecast about 1,300 Pawsey units (6 GCDs × ~3.3 h × 64, held by the slower
  groups arm). The worst case at the 5 h limit is 1,920.

## Decision rule
* Groups help if `groups-shared` − `stack` > 2 pooled SD on val mIoU and test agrees.
* If they help here, band grouping holds on 2 of 3 datasets, both with full 13-band L1C/L2A
  (Sen1Floods11, CloudSEN12+). The docs then recommend it for 13-band S2, and a library helper
  for band groups is considered. If they do not, it stays a single-dataset result (Sen1Floods11).
* Secondary: thin-cloud and shadow IoU deltas, throughput.

## Results

One 6-GCD pack, all 6 runs completed (2 h 19 min wall, **890 units** against the 1,300 forecast;
`pawseyAccountBalance -jobs`). Val is the final-epoch val mIoU; mean ± SD over seeds
0–2.

| arm | val mIoU | test mIoU | test IoU clear / thick / thin / shadow | params | GFLOPs (512²) | train time |
|---|---|---|---|---|---|---|
| stack | 0.7308 ± 0.0028 | 0.6832 ± 0.0068 | 0.904 / 0.865 / 0.398 / 0.565 | 24.5 M | 66.8 | ~100 min |
| groups-shared | 0.7345 ± 0.0046 | 0.6883 ± 0.0021 | 0.908 / 0.868 / 0.399 / 0.578 | 25.5 M | 77.4 | ~132 min |

* **Decision:** groups − stack = **+0.37 val** against 2 pooled SD = 0.76, so it **does not pass**.
  Test moves the same way (+0.51). Per the rule, band grouping stays a single-dataset result
  (Sen1Floods11, E015): the docs do not recommend it for 13-band S2 and no library helper is
  added.
* **Secondary:** the small gain is mostly cloud shadow (+1.3 test IoU); thin cloud is unchanged
  (+0.1). Grouping costs +16% FLOPs and ~30% more training time here.
* **Across the three datasets:** +1.25 (Sen1Floods11), +0.13 (PASTIS), +0.37 (CloudSEN12+) val.
  It is never negative, but only Sen1Floods11 clears the bar, so it is an option to try, not a
  default.
