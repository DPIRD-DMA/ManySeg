# manyseg-bench

Implementation-neutral benchmarking of PyTorch segmentation and image models. It compares
ManySeg, SMP (segmentation-models-pytorch), fastai and plain timm models under one protocol, on
the same data, recipes and metrics.

```bash
# not on PyPI: from a clone of the repository (uv sync installs every extra), or
pip install -e "packages/manyseg-bench[manyseg,smp,geo]"   # geo: readers for the RS datasets
# providers are extras: manyseg, smp, fastai, timm; the core and SMP / fastai / timm semantic
# and image tasks run without ManySeg (instance, point and vector tasks need it)
export MANYSEG_BENCH_DATA=/path/to/data
manyseg-bench prepare oxford_pet          # or sen1floods11, ai4boundaries, oam_tcd, cloudsen12
manyseg-bench train --provider smp --arch unet --encoder resnet34 --seeds 0,1,2 --evaluate-test \
    --save-model                          # evaluate needs the saved checkpoint
manyseg-bench profile --provider manyseg --size 512,512 --compile
manyseg-bench evaluate runs/<run>.json --variants fp32,bf16,int8_weight_only
```

* **Providers:** ManySeg, SMP, fastai and timm behind one interface, with weight
  fingerprints; remote-sensing weights fed their own pretraining normalisation.
* **Tasks:** semantic and multi-label segmentation, instances (PQ, object F1), points (point
  F1, count MAE), image classification, multi-label classification and regression, binary
  and semantic change detection, time series, and vector-output training (corner-line
  instances, unified roads; targets from geometry, dihedral augmentation of direction fields).
* **Datasets:** Oxford Pet, Sen1Floods11, CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS,
  EuroSAT, BigEarthNet v2, OSCD, LEVIR-CD and SECOND, on their official splits as published
  (OAM-TCD also checks each image's licence). Those splits are not all spatially independent:
  AI4Boundaries' is random, not spatially blocked, and Sen1Floods11's train/val/test share
  flood events (its Bolivia event is held out; `--split events:K:F` re-splits leave-events-out).
  Dataset normalisation statistics come from the training split only.
* **Provenance:** every training run's JSON records the environment, code version, GPU contention,
  FLOPs, throughput and peak memory.

See [GitHub](https://github.com/DPIRD-DMA/ManySeg) for the benchmark design and the
experiments. MIT licence.
