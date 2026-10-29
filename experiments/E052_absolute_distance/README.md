# E052: absolute vs normalised distance target, trained on FTW fields (pre-registered 2026-10-29 22:45)

Why: in the large-scene e2e tests with perfect targets (`tests/integration/test_vector_pipeline.py`)
the normalised distance target (distance / the instance's maximum) changes 3-34 fields at tile
seams (merge="cut"; 8-30 with blend), because a clipped instance has a different maximum in each
tile. The absolute distance (px) removes this (0 with cut, <= 1 with blend) and makes field block
decoding independent of the block grid. Before making `distance="absolute"` the default we check
that a trained model is not worse with it. This is the check offered to the user for 0.1.0.

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
default follows the shared `instance_targets` default only if the user agrees, and this is
reported as untested for roofs.
