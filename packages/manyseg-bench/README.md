# manyseg-bench

Implementation-neutral benchmarking of PyTorch segmentation and image models. It compares
ManySeg, SMP (segmentation-models-pytorch), fastai and plain timm models under one protocol, on
the same data, recipes and metrics.

```bash
# not on PyPI: from a clone of the repository (uv sync installs every extra), or
pip install -e "packages/manyseg-bench[manyseg,smp,geo]"   # geo: readers for the RS datasets
export MANYSEG_BENCH_DATA=/path/to/data
manyseg-bench prepare oxford_pet          # or sen1floods11, ai4boundaries, oam_tcd, cloudsen12
manyseg-bench train --provider smp --arch unet --encoder resnet34 --seeds 0,1,2 --evaluate-test
manyseg-bench profile --provider manyseg --size 512,512 --compile
manyseg-bench evaluate runs/<run>.json --variants fp32,bf16,int8_weight_only
```

* **Providers:** ManySeg, SMP, fastai and timm behind one interface, with weight
  fingerprints; remote-sensing weights fed their own pretraining normalisation.
* **Tasks:** semantic and multi-label segmentation, instances (PQ, object F1), points (point
  F1, count MAE), image classification, multi-label classification and regression, binary
  and semantic change detection, time series, and vector-output training (corner-line
  instances, unified roads; targets from geometry, dihedral augmentation of direction fields). Datasets include Oxford Pet, Sen1Floods11,
  CloudSEN12+, AI4Boundaries, OAM-TCD, PASTIS, EuroSAT, BigEarthNet v2, OSCD, LEVIR-CD and
  SECOND, with licence checks and leakage-safe splits.
* **Provenance:** every result JSON records the environment, code version, GPU contention,
  FLOPs, throughput and peak memory.

See [GitHub](https://github.com/DPIRD-DMA/ManySeg) for the benchmark design and the
experiments. MIT licence.
