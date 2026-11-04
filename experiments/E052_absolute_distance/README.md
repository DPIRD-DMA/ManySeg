# E052: absolute vs normalised distance target, trained on FTW fields (pre-registered 2026-10-29 22:45)

Why: in the large-scene e2e tests with perfect targets (`tests/integration/test_vector_pipeline.py`)
the normalised distance target (distance / the instance's maximum) changes 3-34 fields at tile
seams (merge="cut"; 8-30 with blend), because a clipped instance has a different maximum in each
tile. The absolute distance (px) removes this (0 with cut, <= 1 with blend) and makes field block
decoding independent of the block grid. Before making `distance="absolute"` the default we check
that a trained model is not worse with it. This is the check offered to the maintainer for 0.1.0.

## Design
E046's B0 setup unchanged (FTW Netherlands chips, 8 channels, ResNet34 U-Net, fastai_like,
40 epochs, batch 16, dihedral augmentation, InstanceTask boundary width 2), two arms:
**N** `distance="normalized"` (today's default) and **A** `distance="absolute"`. Both arms are
retrained here with this code, seeds 0 and 1 (four runs, ~5 min each, local 4090; the GPU is
shared, timings are contended).

## Evaluation (`evaluate52.py`, E046's metrics and code)
Per run on val: instance thresholds extent {0.4, 0.6} x boundary {0.3, 0.5} x seed, with seed
{0.15, 0.3} for N (E046) and {1.5, 3.0} px for A (the seed threshold is in the target's units;
same grid size), then simplify {0.5, 1, 1.5, 2}, by boundary F. Test once per run, after all
four runs are tuned. Primary: test boundary F (E038 definition, chip-edge points excluded);
secondary: detection F1, C-IoU, PoLiS.

## Decision rule (fixed now)
Mean over seeds 0 and 1 on test, A - N:
* **absolute becomes the default** for instance targets (fields) if boundary F >= -0.01 and
  detection F1 >= -0.01;
* otherwise it stays an option and the default stays normalised.
No further seeds. Roof faces (CornerLineTask) are not tested here: whatever the result, their
default follows the shared `instance_targets` default only if the maintainer agrees, and this is
reported as untested for roofs.

## Results (23:00; NL test 388 chips; GPU otherwise idle)
Training ~5 min per run. N_s0 reproduces E046's B0 (test boundary F 0.592, detection F1 0.593).

| run | thresholds (extent/boundary/seed) | simplify | val boundary F | test boundary F | test P / R | test det F1 | C-IoU | PoLiS px |
|---|---|---|---|---|---|---|---|---|
| N_s0 | 0.6/0.3/0.3 | 0.5 | 0.592 | 0.592 | 0.617 / 0.569 | 0.593 | 0.262 | 1.62 |
| N_s1 | 0.4/0.3/0.3 | 0.5 | 0.590 | 0.586 | 0.614 / 0.561 | 0.582 | 0.260 | 1.57 |
| A_s0 | 0.4/0.3/3.0 px | 0.5 | 0.597 | 0.596 | 0.607 / 0.585 | 0.592 | 0.263 | 1.53 |
| A_s1 | 0.4/0.3/3.0 px | 0.5 | 0.604 | 0.599 | 0.627 / 0.573 | 0.599 | 0.265 | 1.56 |

Mean A - N: boundary F +0.008, detection F1 +0.008, C-IoU +0.003, PoLiS -0.05 px.
**Decision (rule above): absolute becomes the default for fields.** Notes: every A run chose the
largest seed threshold in its grid (3 px), so a larger one might do better still (a grid edge;
it can only favour A). The absolute loss is larger in scale (final train loss 2.2 vs 1.0, the
distance term in px) with no harm seen. Roof faces untested.
