# E017: Do Sentinel-2 band groups help cloud masking? (CloudSEN12+; pre-registered)

Status: **planned** (registered 2026-10-15, before any run). It decides between E015 and E016.

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
