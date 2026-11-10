# manyseg-bench

Implementation-neutral benchmarking of PyTorch segmentation and image models. It compares
ManySeg, SMP (segmentation-models-pytorch), fastai and plain timm models under one protocol, on
the same data, recipes and metrics.

```bash
# not on PyPI: from a clone of the repository (uv sync installs every extra), or
pip install -e "packages/manyseg-bench[manyseg,smp,geo]"   # geo: readers for the RS datasets
# providers are extras: manyseg, smp, fastai, timm. SMP and fastai build segmentation models
# (semantic tasks), timm builds classifiers (image tasks); none of them needs ManySeg. Instance,
# point and vector tasks need ManySeg, which also builds models for every other task.
export MANYSEG_BENCH_DATA=/path/to/data
manyseg-bench prepare oxford_pet          # or sen1floods11, ai4boundaries, oam_tcd, cloudsen12
                                          # (--size: tile size; fixed-size datasets refuse it)
manyseg-bench train --provider smp --arch unet --encoder resnet34 --seeds 0,1,2 --evaluate-test \
    --save-model                          # evaluate needs the saved checkpoint
manyseg-bench profile --provider manyseg --size 512,512 --compile   # --output image: classifiers
                                          # (the default for --provider timm)
manyseg-bench evaluate runs/<run>.json --variants fp32,bf16,int8_weight_only
```

* **Providers:** ManySeg, SMP, fastai and timm behind one interface, with weight
  fingerprints; remote-sensing weights fed their own pretraining normalisation. Each library
  builds its model by its own rules: for an ImageNet stem on more than 3 input channels,
  SMP, timm and ManySeg (default) cycle the RGB filters and scale them by 3/C, while fastai
  keeps the RGB filters and zeroes the extra channels' (each run's `model_notes` records it).
* **Tasks:** semantic and multi-label segmentation, instances (PQ, object F1), points (point
  F1, count MAE), image classification, multi-label classification and regression, binary
  and semantic change detection, time series, and vector-output training (corner-line
  instances, unified roads; targets from geometry, dihedral augmentation of direction fields).
* **Datasets:** Oxford Pet, Sen1Floods11, CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS,
  EuroSAT, BigEarthNet v2, OSCD, LEVIR-CD and SECOND, on their official splits as published
  (OAM-TCD also checks each image's licence). Those splits are not all spatially independent:
  AI4Boundaries' is random, not spatially blocked, and Sen1Floods11's train/val/test share
  flood events (its Bolivia event is held out; `--split events:K:F` re-splits leave-events-out).
  Normalisation: per-channel statistics of the training split for the multispectral
  datasets (Sen1Floods11, CloudSEN12+, PASTIS, EuroSAT, BigEarthNet, OSCD, AI4Boundaries S2);
  ImageNet constants for the RGB ones (Oxford Pet, OAM-TCD, AI4Boundaries, LEVIR-CD, SECOND).
  Each run records its source under `norm_stats`.
* **Provenance:** every training run's JSON records the environment, code version (git
  commit, `-dirty` if a tracked file in the repository is modified), GPU contention (other
  processes on the run's GPU), FLOPs, throughput and peak memory. Image-level models build no
  decoder, so their records say `arch: none` (the `--arch` given is kept as `requested_arch`).

See [GitHub](https://github.com/DPIRD-DMA/ManySeg) for the benchmark design and the
experiments. MIT licence.
