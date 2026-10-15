# manyseg-bench

Implementation-neutral benchmarking of PyTorch segmentation models. It compares ManySeg, SMP
(segmentation-models-pytorch) and fastai models under one protocol, on the same data, recipes
and metrics. The [`manyseg`](https://pypi.org/project/manyseg/) model library never imports
this package.

```bash
pip install "manyseg-bench[manyseg,smp]"
export MANYSEG_BENCH_DATA=/path/to/data
manyseg-bench prepare oxford_pet          # or sen1floods11, ai4boundaries, oam_tcd, cloudsen12
manyseg-bench train --provider smp --arch unet --encoder resnet34 --seeds 0,1,2 --evaluate-test
manyseg-bench profile --provider manyseg --size 512,512 --compile
manyseg-bench evaluate runs/<run>.json --variants fp32,bf16,int8_weight_only
```

* **Providers:** ManySeg, SMP and fastai behind one interface, with weight fingerprints.
* **Tasks:** semantic segmentation, instances (PQ, object F1) and points (point F1, count MAE).
  Datasets include Oxford Pet, Sen1Floods11, AI4Boundaries, OAM-TCD and CloudSEN12+, with
  licence checks and leakage-safe splits.
* **Provenance:** every result JSON records the environment, code version, GPU contention,
  FLOPs, throughput and peak memory.

See [GitHub](https://github.com/DPIRD-DMA/ManySeg) for the benchmark design and the
experiments. MIT licence.
