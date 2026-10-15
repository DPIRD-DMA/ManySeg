# DRAFT upstream issue (not posted). Target: pytorch/ao

**Title:** X86InductorQuantizer: compiled int8 silently diverges from the PT2E reference on CPUs
without VNNI (default `reduce_range=False`)

## Summary

On an x86 CPU without VNNI (AMD Zen 3, AVX2 only), a model quantised with
`X86InductorQuantizer` and the default config (`reduce_range=False`), then compiled with
`torch.compile` and inductor freezing, produces outputs far from the converted model's own
eager quantise/dequantise reference. Nothing warns or errors. With `reduce_range=True` the
compiled model matches the reference.

The docstring of `get_default_x86_inductor_quantization_config` already says to set
`reduce_range=True` "on earlier CPUs without VNNI to avoid accuracy issue". In practice the
default silently produces wrong results on such CPUs, which include current AMD desktop parts
(Zen 3), and the user has no signal.

## Reproduction

Standalone script (torch + torchao + timm): `repro_x86_reduce_range.py` (attached / inline
below). It quantises a stock `resnet34.tv_in1k` trunk, calibrates on smooth natural-image-like
inputs, and compares `torch.compile(converted)` against `converted` (eager).

```
torch 2.14.1+cu130 | VNNI: False | capability: AVX2
reduce_range=False: compiled vs eager mean relative difference 0.853
reduce_range=True : compiled vs eager mean relative difference 0.069
```

<details><summary>repro_x86_reduce_range.py</summary>

(paste the contents of `docs/upstream/repro_x86_reduce_range.py`)

</details>

## Impact observed downstream

In a segmentation library (ResNet-34 U-Nets, trained, 3 seeds), the compiled int8 model lost
**6.5 ± 0.6 mIoU** on Oxford-IIIT Pet (pixel agreement with fp32 fell to 94.5%), while the eager
PT2E reference was within 0.04 mIoU. A multispectral flood-mapping model (Sen1Floods11) was
unaffected. The loss depends on the data's activation ranges, so it can pass a quick check on
one dataset and fail on another. With `reduce_range=True` the compiled models were within
0.10–0.13 mIoU of fp32 on both datasets.

Minimal checks that did **not** reproduce it, which may help localise it: a single
`Conv2d(64, 32, 3)` with non-negative inputs and weights; a tiny conv / BN / ReLU /
upsample / concat U-Net with random weights; and the full ResNet-34 classifier compared only by
top-1 class on Gaussian noise (the logits' argmax happened to survive). With real weights, the
stage-4 features diverge strongly.

## Environment

* torch 2.14.1 (+cu130 wheel, run on CPU), torchao 0.18.0, timm 1.0.30, Python 3.12, Linux
* CPU: AMD Ryzen 9 5950X (Zen 3). `torch.cpu._is_vnni_supported()` is False and
  `torch.backends.cpu.get_cpu_capability()` is "AVX2".

## Suggestions

1. Default `reduce_range` from the CPU (e.g. `not torch.cpu._is_vnni_supported()`), or
2. warn in `X86InductorQuantizer` / at lowering when `reduce_range=False` on a CPU without VNNI,
   or
3. have inductor's int8 lowering avoid the saturating u8 × s8 path (or fall back) on
   non-VNNI CPUs.

(Downstream workaround, for reference: ManySeg's `quantize_pt2e` now defaults to
`reduce_range=True` on non-VNNI CPUs, warns when it produces a full-range program, and records
the choice so a deployment-time check can refuse a non-VNNI CPU.)
