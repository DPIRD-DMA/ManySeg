# ManySeg public API design

Status: v0.2 (2026-10-14). Everything below is implemented and tested unless marked *(planned)*.

## Goals

1. One function, `create_model`, covers the common case, and the objects it builds are ordinary
   `nn.Module`s that can also be composed directly.
2. A model's behaviour (inputs, outputs, shapes) is predictable from its construction arguments.
   Nothing about the return type depends on runtime values.
3. Every module in a forward pass is compile- and export-friendly: no tensor-dependent Python
   control flow, no hidden device transfers, no in-forward dictionary mutation keyed on data.
4. The configuration surface stays small. Options are added only when there is a test or an
   experiment that justifies them.

## Layered structure

```
inputs ──► InputAdapter ──► Backbone ──► (Neck) ──┐
                                                  ├─► Fusion ─► Decoder ─► Heads ─► outputs
inputs ──► InputAdapter ──► Backbone ──► (Neck) ──┘
```

| Layer | Contract |
| --- | --- |
| `InputAdapter` | Maps `C_in` channels to what the backbone stem expects. It is either folded into the stem conv (`repeat`, `mean`, `random`, `zero`), or it is a small module in front of the stem (`projection`). |
| `Backbone` | `forward(x) -> list[Tensor]`, all NCHW, plus a static `feature_specs: tuple[FeatureSpec, ...]`. |
| `Neck` | Optional. Converts single-scale token maps (plain ViT) into a pyramid. Same contract as a backbone. |
| `Fusion` | Combines several pyramids into one *(phase 4)*. |
| `Decoder` | Constructed from `in_specs`. `forward(list[Tensor]) -> Tensor`. Exposes `out_spec: FeatureSpec`. |
| `Head` | Constructed from a `FeatureSpec`. Maps decoder (or encoder) features to a named output. |

### FeatureSpec

```python
@dataclass(frozen=True)
class FeatureSpec:
    channels: int
    stride: int        # reduction relative to the model input
    name: str = ""     # module name in the backbone, for debugging
```

Feature *tensors* stay plain `list[Tensor]` so that `torch.compile` and `torch.export` never
trace through dataclass construction. The static metadata lives on the module.

All features are returned NCHW. A backbone that produces NHWC (Swin, some hybrids) or token
sequences (ViT) is converted inside the backbone wrapper, so decoders never need to branch on
layout.

## `create_model`

### Single input, single output

```python
from manyseg import create_model

model = create_model(
    backbone="resnet34",      # any timm model name with a supported feature API
    pretrained=True,
    in_channels=4,
    decoder="unet",           # "unet" | "fpn" | ... (registry)
    num_classes=3,
)
logits = model(x)             # x: (N, 4, H, W) -> (N, 3, H, W); returns a Tensor
```

`model(x)` returns logits at the input resolution `(H, W)` for any `H, W` the backbone supports,
including odd and non-square sizes (see "Spatial policy").

### Presets

`manyseg.presets` names the configs on the measured accuracy/latency Pareto front (E018). They
are keyword sets for `create_model` and are opt-in; the defaults above keep SMP parity.

```python
from manyseg.presets import PRESETS, preset

model = create_model(**preset("balanced"), in_channels=4, num_classes=3)
PRESETS["balanced"].measured   # latency, GFLOPs, params, val mIoU per dataset
```

| preset | config | 4090 bf16 b8 512² | Pet val | Sen1Floods11 val |
|---|---|---|---|---|
| `fast` | PVTv2-b0 + SegFormer head | 4.5 ms | 0.820 | 0.901 |
| `compact` | PVTv2-b0 + U-Net (64…8) | 4.9 ms | 0.823 | 0.901 |
| `balanced` | PVTv2-b0 + U-Net (128…8) | 5.3 ms | 0.825 | 0.902 |
| `accurate` | PVTv2-b0 + U-Net (256…16) | 7.3 ms | 0.827 | 0.901 |
| (default) | ResNet-34 + U-Net (256…16) | 10.9 ms | 0.823 | 0.890 |

Two datasets and one GPU; validate on your own data. On CPU (fp32, contended) every preset was
at least as fast as the default, by a smaller margin; see the E018 write-up. The ResNet-34 default stays because it is
the familiar SMP baseline, not because it is on the front.

### Multiple outputs

Passing `heads=` instead of `num_classes=` makes the model return a `dict[str, Tensor]` keyed by
head name. The return type is fixed when the model is built.

```python
model = create_model(
    backbone="convnext_tiny",
    in_channels=4,
    decoder="unet",
    heads={
        "segmentation": {"type": "segmentation", "channels": 3},
        "height":       {"type": "regression", "channels": 1},              # per-pixel
        "water_area":   {"type": "regression", "channels": 1, "scope": "image"},
        "scene":        {"type": "classification", "channels": 10, "scope": "image"},
    },
)
out = model(x)
out["segmentation"]  # (N, 3, H, W)
out["water_area"]    # (N, 1)
```

Heads return raw logits or values: activations and losses live outside the model. `scope` is
`"pixel"` (default) or `"image"`. Image heads pool the deepest encoder feature by default
(`"source": "decoder"` selects the decoder output). `{"type": "regression", "uncertainty": True}`
returns `2·C` channels, `[mean, log_variance]`, for heteroscedastic losses such as Gaussian NLL.

### Image classification and regression

A model whose heads are all `"scope": "image"` builds **no decoder**: it is an encoder, a
global average pool of the deepest feature and a linear layer per head, i.e. a classifier or
regressor. Everything else composes as for dense models: multiple inputs at different
resolutions, fusion, metadata conditioning (`film` modulates the pooled feature; `head` and
`channels` as usual), uncertainty heads, export and quantisation.

```python
model = create_model(
    inputs={
        "s2_10m": {"channels": 4, "backbone": "resnet50"},
        "s2_20m": {"channels": 6, "backbone": "resnet50", "alignment": "same_extent"},
        "meta":   {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}},
    },
    fusion={"type": "intermediate", "op": "concat"},
    heads={
        "crop":    {"type": "classification", "channels": 10, "scope": "image"},
        "harvest": {"type": "regression", "channels": 1, "scope": "image", "uncertainty": True},
    },
)
out = model({"s2_10m": a, "s2_20m": b, "meta": doy})   # crop (N, 10), harvest (N, 2)
```

For a single RGB/multispectral tensor and one output this matches
`timm.create_model(backbone, num_classes=K, in_chans=C)`: same encoder, pretrained weights and
stem adaptation, pool + linear head. Exception: families whose timm head has a pre-logits norm
(ConvNeXt's LayerNorm) are pooled without it here.

> **Evidence (E019):** on EuroSAT-MS (13 bands) and Pet breeds, with ResNet-50, ConvNeXt-T and
> PVTv2-b0, ManySeg and timm classifiers are at parity on validation accuracy in all six cells.
> On test ManySeg averaged 0.43 points lower, mostly ConvNeXt, whose timm head has a pretrained
> LayerNorm that ManySeg's image head lacks. `{"init_std": 0.02}` in an image head's config uses
> timm's classifier init; it narrowed the gap to 0.27 (E019b), not enough to become the default.
>
> **Evidence (E020, BigEarthNet v2, 19-class multi-label, ResNet-50):** parity with timm for
> classification and for class-fraction regression. Resolution band groups, Sentinel-1 (stacked
> or fused) and date + country metadata did not help. A segmentation head on the reference maps
> (U-Net decoder; the classifier still pools the encoder) raised test macro mAP by +2.3 (val
> +2.1), mostly on hard classes. One encoder, a 60k-patch subset and a fixed 12-epoch budget; a
> post-hoc look suggests regularisation (no gain at the classifier's early peak, less late
> overfitting), so the gain may shrink with early stopping. Worth trying when pixel labels exist;
> cost: the decoder's parameters and ~37% lower training throughput. Mix image and pixel heads freely: the decoder is built only when a pixel
head needs it. After training with auxiliary heads, `model.select_heads(["crop"])` keeps only
the heads you deploy and removes decoders (and FiLM / head-bias layers) only the dropped heads
used; the kept outputs are unchanged in eval mode. To load such a checkpoint, rebuild the model
and call `select_heads` with the same names first. Head and input names must not be Python keywords (`yield`, `class`, ...):
`torch.export` reads modules as attributes.

### Multiple inputs

```python
model = create_model(
    inputs={
        "optical": {"channels": 4, "backbone": "convnext_tiny", "pretrained": True},
        "sar":     {"channels": 2, "backbone": "resnet18",      "pretrained": True,
                    "alignment": "same_extent"},       # may be at a different resolution
    },
    fusion={"type": "intermediate", "op": "gated", "strides": [8, 16, 32]},
    decoder="unet",
    heads={...},
)
out = model({"optical": o, "sar": s})
```

* **Early fusion**: `fusion={"type": "early"}`. Inputs are concatenated on the channel axis
  in front of one shared backbone. Inputs must share spatial size (`same_grid`). Per-input
  encoder keys (`backbone`, `pretrained`, `channel_adapter`, `out_indices`,
  `backbone_kwargs`) must agree across inputs, and they then override the top-level arguments.
  Conflicting values, and a per-input `rgb_indices`, are errors.
* **Intermediate fusion**: each input has its own backbone. Features with the same stride
  are fused by an op (`concat`, `add`, `gated`, `attention`, `cross_attention`) whose output
  has the primary input's channels. Strides that are not selected, or that exist only in the
  primary, pass the primary feature through unchanged. The first input in `inputs` is the
  primary.
* **Late fusion**: `fusion={"type": "late", "op": ...}`. Each input has its own backbone and
  decoder (same type and options). The decoder outputs are fused before the heads. A
  `same_extent` input's decoder output is resampled by its valid region onto the primary's
  valid region, as in intermediate fusion. Image-scope heads read the primary input's deepest
  encoder level.
* Inputs without a `backbone` key use the top-level `backbone=` argument.
* **Shared encoder**: `{"backbone": "shared:optical"}` reuses another input's backbone weights.
  Each input keeps its own input adapter **and its own BatchNorm layers** (affine weights and
  running statistics). One BN shared by inputs with different statistics blends its running
  statistics, and evaluation then applies the blend to every input: in E015 that collapsed a
  shared-encoder model from 0.90 to 0.50–0.65 val mIoU. `"share_norm": True` shares BN too (the
  old behaviour); use it only for inputs with matching statistics (e.g. two dates of the same
  sensor).

#### Time series

`{"kind": "timeseries", "channels": C, "temporal": ...}` takes `(B, T, C, H, W)` for any `T`.
One 2D encoder (any timm backbone, pretrained weights unchanged) runs on every date, and each
feature level is aggregated over time, so fusion, decoders and heads see an ordinary pyramid.
Optional entries in the input dict: `"<name>_mask"` `(B, T)` bool (False = missing, cloudy or
padded date) and `"<name>_dates"` `(B, T)` days (e.g. day of year).

```python
model = create_model(
    inputs={
        "s2":  {"kind": "timeseries", "channels": 10, "temporal": "attention"},
        "dem": {"channels": 1, "alignment": "same_extent"},
    },
    heads={"crop": {"type": "segmentation", "channels": 20}},
)
out = model({"s2": s2, "s2_mask": valid, "s2_dates": doy, "dem": dem})
```

| `temporal` | aggregation per level |
|---|---|
| `"mean"` (default) | masked mean over valid dates |
| `"max"` | masked per-channel max |
| `"attention"` | per-pixel temporal attention over dates (L-TAE style: learned query, keys from features + a sinusoidal date encoding, `heads` channel groups). Starts as the masked mean (zero query). Options: `heads` (4), `key_dim` (16), `date_dim` (32), `max_period` (10000) |

Time series combine with other rasters (any alignment), metadata vectors and every head type;
fusion must be intermediate or late (early fusion cannot stack a `T` axis). Exports with
dynamic `T`, `H` and `W`. The encoder runs once per date, so memory scales with `T`. Masked
dates never affect the output: in training mode only valid dates are encoded (so they stay out
of BatchNorm batch statistics; a partly masked batch has a data-dependent shape, so
`torch.compile(fullgraph=True)` of a training step needs fully valid batches), and NaNs in
masked dates are ignored.
Not yet: `shared:` backbones between a time series and another input, and metadata in
`"channels"` mode targeting a time series.

#### Change detection

Before / after images are two inputs on the same grid. Binary change: one segmentation head on
the fused features; `"op": "diff"` (`|f1 - f2|`) or `"concat_diff"` (`[f1, f2, |f1 - f2|]`) are
the Siamese change fusions, `"share_norm": True` makes the shared encoder fully Siamese, and
stacking both dates in one input (early fusion) is the simple baseline. For a series of dates,
the order-keeping temporal aggregations are `{"type": "concat", "dates": T}` and `"diff"`
(two dates).

**Semantic change** (what changed into what) adds a land-cover head per date. A head with
`"input": "<name>"` reads that input's own (unfused) features through its own decoder
(default: the model's decoder type); `"share": "<head>"` reuses another per-input head's
decoder and head, so both dates are mapped by the same weights:

```python
model = create_model(
    backbone="resnet34",
    inputs={"t1": {"channels": 3},
            "t2": {"channels": 3, "backbone": "shared:t1", "share_norm": True}},
    fusion={"type": "intermediate", "op": "concat_diff"},
    heads={"change": {"type": "segmentation", "channels": 2},          # fused features
           "lc1": {"type": "segmentation", "channels": 7, "input": "t1"},
           "lc2": {"type": "segmentation", "channels": 7, "input": "t2", "share": "lc1"}},
)
out = model({"t1": before, "t2": after})   # change, lc1 (date 1), lc2 (date 2)
```

`lc1` depends only on `t1` (tested), and with `share` `lc2(t1, t2) == lc1(t2, t1)`. Per-input
heads need intermediate fusion and raster (not time-series) inputs.

#### Spatial alignment

Inputs with different pixel sizes are **not** assumed to cover the same ground extent. Each
input declares an alignment policy:

| Policy | Meaning |
| --- | --- |
| `"same_grid"` (default) | Inputs must have identical `(H, W)`. A mismatch raises an error. |
| `"same_extent"` | The input covers the same ground extent as the primary input at a different resolution. At each fused stride, the *valid* (unpadded) region of its features is resampled (bilinear) onto the primary's valid region, so different amounts of padding cannot shift the alignment. Verified by a gradient-centroid test. |

Inputs that are co-registered only partially must be cropped or resampled by the user's data
pipeline. Reprojecting or georeferencing data is outside the scope of a model library.

### Conditioning inputs (metadata vectors)

Non-raster inputs such as acquisition date, sun or view angles, lat/lon, sensor id, GSD or
tabular features are declared as `"kind": "vector"` inputs next to the rasters. They condition
the raster model. They never replace it: a model with only vector inputs is an error.
Pretrained encoders are not modified.

```python
model = create_model(
    backbone="resnet34",
    inputs={
        "s2":   {"channels": 13},
        "meta": {"kind": "vector", "fields": {
            "doy":      {"type": "cyclic", "period": 365.25},
            "sun_elev": {"type": "continuous", "mean": 45.0, "std": 15.0},
            "latlon":   {"type": "coords"},                  # (lat, lon) in degrees
            "sensor":   {"type": "categorical", "num": 3},
        }, "dim": 64, "dropout": 0.1, "conditioning": "film"},
    },
    num_classes=2,
)
meta = model.conditioner.encoders["meta"].pack(
    {"doy": doy, "sun_elev": elev, "latlon": latlon, "sensor": sensor})   # (B, 5)
out = model({"s2": s2, "meta": meta})                   # optional: "meta_mask": (B, 4)
```

`{"kind": "vector", "channels": N}` declares `N` pre-normalised floats instead of `fields`.

**Input convention.** A vector input is **one packed float tensor `(B, N)`**. Its columns are
the fields in declaration order: `coords` takes two columns (lat, then lon), `categorical`
takes one column holding the id, and every other type takes one column. A dict of per-field
tensors is rejected with a `TypeError` that points to `MetadataEncoder.pack`. One tensor per
vector input keeps the model input a flat `dict[str, Tensor]`: a single ONNX input whose only
dynamic dimension is the batch, and a single array per sample in a data loader.

**Field encodings** (`manyseg.conditioning.FIELD_TYPES`):

| type | options | encoding | width |
| --- | --- | --- | --- |
| `continuous` | `mean`, `std` | `(v - mean) / std` | 1 |
| `cyclic` | `period` | `[sin, cos](2πv / period)`, continuous across the period boundary | 2 |
| `coords` | `frequencies=1` | point `u` on the unit sphere (continuous at the dateline and the poles) plus `frequencies-1` octaves `[sin, cos](2^k π u)`, which keep both properties | 3 + 6(f-1) |
| `categorical` | `num`, `dim=8` | learned embedding; a value that is not an integer or lies outside `[0, num)` counts as missing | `dim` |

The encodings are concatenated and mixed by a 2-layer MLP (`Linear-GELU-Linear`) into a
`(B, dim)` embedding.

**Missing values.** A field is missing where the optional `x["<name>_mask"]` `(B, F)` (one
column per field, 1 = present) is 0, or where any of its columns is NaN, or where a category
is out of range. A missing field's encoding is replaced by a learned per-field "unknown"
vector, so a model runs with every field missing (for example, all-NaN metadata). With
`"dropout": p`, training drops each field of each sample with probability `p`, so the model
learns the unknown path. All of this is masking with `torch.where`, with no data-dependent
branching, so it compiles and exports.

**Injection modes** (`"conditioning": mode` or `{"mode": mode, **options}` on the vector
input):

| mode | what it does | init |
| --- | --- | --- |
| `film` (default) | FiLM `x·(1+γ(e)) + β(e)` per channel on every encoder feature level the decoder reads (just before it enters the decoder) and on the decoder output. Applied around every decoder: the shared one, per-head task decoders and the late-fusion decoder. `{"on": ["output"]}` limits it to the output. Image-scope heads that pool the encoder (classification / regression, including models with no decoder) get one FiLM on the deepest encoder feature, i.e. on the pooled vector. | exact identity |
| `head` | Image heads: `Linear([pool(f), e])`, implemented as `Linear(pool(f)) + Linear_e(e)`. With `{"pixel": True}`, pixel heads also get the same per-sample term broadcast over space (a 1×1 kernel on the broadcast embedding). | exact identity |
| `channels` | The field encodings (no MLP) are broadcast as extra input channels of one raster input (`{"target": name}`, default the first raster) and go through that input's channel adapter. This is an early-fusion baseline. | depends on the adapter |
| `token` | Metadata as extra ViT tokens. Raises `NotImplementedError`. | n/a |

Why the option sits on the vector input rather than in `fusion=`: `fusion=` describes how
rasters combine. Conditioning is orthogonal to that. It works with a single raster and with
early, intermediate or late fusion, and several vector inputs can use different modes (for
example, `film` for acquisition metadata and `head` for tabular features).

Why FiLM wraps decoders instead of editing them: decoders keep their
`forward(list[Tensor]) -> Tensor` contract, so a FiLM layer can only see what crosses that
boundary. In practice, "each decoder level" means each encoder level the decoder consumes,
plus the decoder output. It does not mean the decoder's internal blocks. The levels a decoder
reads come from its optional `read_levels` attribute. Otherwise they are found once at
construction by a gradient probe (for example, DeepLabV3+ reads only strides 4 and 32), so
every FiLM parameter receives a gradient. The probe takes the union over a square and an odd,
non-square size. It does not touch the global RNG, and it restores every submodule's
train/eval flag and every buffer. Read sets that depend on the input size in other ways are not
supported. Encoders are built after the raster model, so the raster weights match the
unconditioned model's for the same seed. All γ/β projections start at
zero, so a freshly built conditioned model computes exactly what the unconditioned model with
the same weights computes (tested with `rtol=atol=0` for every decoder).

With a single raster input and no `fusion=`, the raster model is the ordinary single-input
model (any decoder works, including `fastai_unet`). With several rasters it is the multi-input
model. `export_model` gives vector inputs and masks no spatial dimensions (only the batch, with
`dynamic_batch=True`).

**Leakage.** In remote sensing, metadata often identifies the scene, event or region (lat/lon
above all). Evaluate conditioned models on splits that hold out whole events or regions. See
`experiments/E009_metadata_conditioning`.

## Spatial policy

Most CNN encoders need inputs divisible by their total stride (32) so that skip connections
line up exactly. ManySeg handles this explicitly:

* `pad_to_stride=True` (default): the input is padded on the bottom/right to a multiple of the
  required divisor, the network runs, and outputs are cropped back to `(H, W)`. Pixels keep
  their positions, so pixel `(i, j)` of the output corresponds to pixel `(i, j)` of the input.
  The pad value is 0 (applied after the user's normalisation, so 0 is the per-channel mean).
* Decoders resize the upsampled feature to each skip's exact size rather than assuming
  `2×`, so they also work when padding is disabled and a backbone rounds feature sizes up.
* Backbones that only run at one size (MaxViT, CoAtNet, LeViT, EfficientFormerV2, Hiera:
  window partitions or positional tables baked in) are detected at construction by a
  meta-device probe and marked `fixed_size` at their native size, or at `img_size` when
  `backbone_kwargs={"img_size": ...}` is given. Smaller inputs are padded up to it and the
  output is cropped back. Larger inputs raise a `ValueError` that names the backbone and the
  `img_size` fix. Nothing is ever silently resized.

### Reduced output resolution

Output resolution can be traded for speed and memory in two independent places.

**Decoder `out_stride`** (`decoder_kwargs={"out_stride": s}` or a task decoder's config). The
U-Net family (`unet`, `unetplusplus`, `fastai_unet`) accepts 1 (default), 2, 4, ... (a power of
two below the deepest stride). Stages finer than `s` are not built, so they cost no parameters
or FLOPs, and `out_spec.stride == s`. The reduced decoder is a strict prefix of the default:
every kept module has the same name, shape and computation (state_dict keys are a subset).
`out_stride=1` is the default, so `unet` stays identical to SMP's Unet and `fastai_unet` to
fastai's DynamicUnet (bench parity tests). What is dropped:

| decoder | `out_stride=2` drops | `out_stride=4` also drops |
| --- | --- | --- |
| `unet` | the last block (stride 2 → 1) | the block fusing the stride-2 skip |
| `unetplusplus` | the tail block to stride 1 | the stride-2 row of nested nodes `X[0][*]` (10 → 6 nodes) |
| `fastai_unet` | the final pixel shuffle, the input-image cross and the input-resolution residual block (`last_cross`) | the up-block fusing the stride-2 skip |

Measured cost (ResNet-34, 512², 2 classes, `FlopCounterMode`; total model and change vs
`out_stride=1`). FLOPs understate the saving: the dropped stages are the memory- and
bandwidth-heaviest, so measure latency and memory too (E006).

| decoder | `out_stride=1` | `out_stride=2` | `out_stride=4` |
| --- | --- | --- | --- |
| `unet` | 24.44 M, 62.6 GFLOPs | 24.43 M, 58.9 (−5.9%) | 24.38 M, 52.8 (−15.6%) |
| `unetplusplus` | 25.77 M, 107.3 | 25.77 M, 103.6 (−3.4%) | 25.53 M, 72.1 (−32.8%) |
| `fastai_unet` | 41.22 M, 252.1 | 41.01 M, 154.1 (−38.9%) | 40.63 M, 117.1 (−53.6%) |

For reference, the stride-4 decoders cost 48.0 (`segformer`), 54.9 (`fpn`), 41.6
(`deeplabv3plus`) and 143.9 (`upernet`) GFLOPs.

(For encoders whose shallowest feature is stride 4, `fastai_unet` keeps the final pixel shuffle
at `out_stride=2`: it is the step from stride 4 to 2. See the `fastai_unet` module docstring.)
`fpn`, `deeplabv3plus`, `upernet` and `segformer` have a fixed stride (4 for usual encoders).
They accept `out_stride` only if it equals that stride; anything else is a `ValueError`.

**`input_cross`** (`unet` only: `decoder_kwargs={"input_cross": True}`, default off). It adds
fastai's full-resolution stage to the SMP-style U-Net: the raw input image is concatenated after
the last block, followed by fastai's residual block (about +7k parameters). It requires
`out_stride=1` and a single raster input.

> **Evidence (E012, Sen1Floods11):** it recovered 83% (ResNet-34) and 81% (PVTv2-b2) of
> fastai_unet's val mIoU gain over `unet`. On test it matched or beat fastai_unet (+2.1 / +1.8
> mIoU over `unet`), at +14% FLOPs against fastai's 4×. On Oxford Pet it did nothing, and on
> CloudSEN12+ cloud masking (E013) it did not replicate (+0.31 val, −0.01 test). Try it when the
> label at a pixel is largely decided by that pixel's own input values at full resolution (a
> spectral threshold or index rule, e.g. water, burnt area, bare soil, snow), and validate it on
> your data. It is not a general improvement, so the default stays SMP-identical.

**Head `scale`** (`{"type": ..., "channels": ..., "scale": s}`, pixel heads only; image heads
reject it). `s` ∈ {1, 1/2, 1/4, 1/8}, default 1. With `f = 1/s`, a primary input of `H × W`
gives an output of `ceil(H / f) × ceil(W / f)` (ceil, so every input pixel belongs to exactly
one output pixel; floor would drop the last partial row and column). The head reads a decoder
feature of stride `d`:

* `d == f`: only the head conv runs. Nothing is resampled. This is the saving: a decoder with
  `out_stride=4` feeding a `scale=1/4` head never touches full resolution.
* `d > f`: conv, then bilinear upsampling by `d / f` (as for `scale=1`).
* `d < f`: block-average by `f / d`, then the conv on the coarse grid.

The model crops the head output to the scaled size of the *unpadded* primary input, so
`pad_to_stride` keeps working (with `pad_to_stride=False` it resizes to that size instead).
Multi-input models use the primary input's grid. Sizes are computed by floor division and
crops are negative padding, so `torch.compile(dynamic=True)`, `torch.export` and ONNX keep
dynamic H/W (tested).

**Alignment.** Output pixel `(i, j)` of a head with `scale = 1/f` covers input pixels
`[i·f, (i+1)·f) × [j·f, (j+1)·f)`. With pixel centres at integer coordinates, its centre is
the input coordinate `(i + 0.5)·f − 0.5` (`manyseg.utils.to_input_coords`; the inverse is
`to_output_coords`). A known-answer test (an area-averaging backbone with an identity head)
checks that the output equals the input's block means, and gradient-centroid tests check
every `out_stride` × `scale`. With `upsample="aligned"` the centroid is within
`1.5 + (out_stride − 1)/2` px of the documented centre. The extra `(out_stride − 1)/2` is the
encoder's own geometry: a "same"-padded strided conv centres feature `k` on input pixel
`k·d`, not on its block centre (E001). Default nearest upsampling keeps its known up-left
bias (about 6 px at init) at every stride.

**Targets and predictions at head scale** (`manyseg.utils`, outside the model graph):

| helper | for | rule |
| --- | --- | --- |
| `downsample_labels(y, s, ignore_index, mode="mode")` | class ids | majority per block; ignored pixels and padding never vote; all-ignored block → `ignore_index`; ties → lowest id. `mode="nearest"` takes the block-centre pixel. Pass `num_classes` to avoid device syncs. |
| `downsample_mean(x, s, valid=None)` | continuous targets | mean over each block's in-image (and `valid`) pixels |
| `downsample_sum(x, s)` | density targets | block sums, so the count is preserved |
| `downsample_max(x, s)` | heatmaps, thin boundaries | block maximum: keeps peaks; "any" for boolean maps |
| `upsample_output(y, (H, W), s, mode)` | predictions | bilinear (logits, values) or nearest (labels, any dtype), cropped to `H × W` |

Instance and point post-processing runs on the grid it is given. `find_peaks(hm, scale=s)`
returns float input-pixel coordinates. `group_by_offsets` and `instances_from_outputs` return
head-grid labels; map them back with `upsample_output(labels, (H, W), s, mode="nearest")`.
`manyseg-bench` computes losses on each head's own grid and compares at input resolution.
Native targets are built according to what each target means:

* semantic labels and instance extent: majority vote;
* boundaries: any;
* distance: mean over instance pixels;
* heatmaps: max;
* offsets: mean divided by `f`;
* density: sum;
* point validity: the fraction of valid pixels in each block.

For metrics, semantic and instance predictions are upsampled to input resolution, points are
picked on the native heatmap and returned in input-pixel coordinates, and counts integrate the
native density. Native-scale semantic metrics are optional (`native_metrics=True`). See the
`manyseg_bench.tasks` module docstring.

## Input channel adaptation

```python
create_model(backbone="resnet34", pretrained=True, in_channels=13, channel_adapter="repeat")
```

| `channel_adapter` | Behaviour on the pretrained stem conv `W ∈ R^{O×3×k×k}` |
| --- | --- |
| `"repeat"` (default; timm-compatible) | Cycle the RGB filters to `C` channels and scale by `3/C`, so a constant input gives the same response as before. |
| `"mean"` | Every channel gets `mean_c(W)·3/C`. |
| `"rgb_plus_random"` | The first 3 channels keep `W` unchanged. The others are kaiming-initialised at a small scale. |
| `"rgb_plus_zero"` | The first 3 channels keep `W`. The others are zero, so the model is initially identical to RGB-only. |
| `"projection"` | A learned 1×1 conv `C→3` in front of the unchanged stem. RGB-identity initialisation when `C ≥ 3`. |

`in_channels=3` with a pretrained backbone leaves the weights untouched for every adapter. The
benchmarks in Phase 5 decide whether the default should change.

## Lower-level composition

```python
from manyseg.backbones import create_backbone
from manyseg.decoders import UNetDecoder
from manyseg.heads import SegmentationHead
from manyseg.models import DenseModel

bb = create_backbone("resnet34", pretrained=True, in_channels=4)
dec = UNetDecoder(bb.feature_specs)
model = DenseModel({"image": bb}, dec, {"mask": SegmentationHead(dec.out_spec, 3)},
                   single_output="mask")
```

`create_model` is a thin layer of argument parsing over this composition. Every component
can be swapped by passing a module instead of a string.

### Instance and point outputs

> **Evidence:** E010 (AI4Boundaries): boundary + distance heads doubled PQ over extent +
> connected components (0.145 → 0.290), and per-head decoders (`decoders="separate"`) added
> +0.06 more, so they are the recommended preset for field delineation. E011 (OAM-TCD): MSE
> regression heatmaps beat focal-loss heatmaps. Calibrate the `find_peaks` threshold on
> validation data: E011b raised test point F1 from 0.40 (fixed 0.5) to 0.59 (val-chosen 0.15–0.2,
> since MSE peaks rarely reach 1). With calibrated peaks, counting peaks (MAE 20.6) is close to
> the density head (19.4). `density_head()` stays slightly better for counting and needs no
> threshold.


Dense instance-style outputs are built from ordinary pixel heads, optionally with per-head
task decoders. There are no proposal- or query-based methods. Presets return head configs:

```python
from manyseg.instances import instance_heads, instance_targets, instances_from_outputs
from manyseg.points import point_heads, density_head, find_peaks, count_from_density

# Field / paddock / crown / footprint delineation (ResUNet-a / FracTAL style)
model = create_model(backbone="convnext_tiny", in_channels=4,
                     heads=instance_heads(decoders={"boundary": "unet"}))
t = instance_targets(instance_ids, boundary_width=2)     # torch, runs on the GPU
labels = instances_from_outputs(out["extent"], out["boundary"], out["distance"], logits=True)

# Counting / locating trees, livestock, dams, sheds
model = create_model(backbone="resnet34", heads={**point_heads(offsets=True), **density_head()})
peaks = find_peaks(out["heatmap"], threshold=thr)        # thr chosen on val (E011b: ~0.15-0.2)
count = count_from_density(out["density"])
```

| Preset | Heads | Targets (`manyseg.*.targets`) | Post-processing |
| --- | --- | --- | --- |
| `instance_heads(num_classes=1, distance="normalized", decoders=None)` | `extent` (1 ch, or C+1 with background), `boundary` (1), `distance` (1, regression) | `instance_targets(ids, boundary_width, boundary_mode="inner"/"outer"/"both")`: exact EDT; distance is 1 + distance to the instance's edge, normalised per instance; tile borders are not boundaries | seeds = extent ∧ ¬boundary ∧ distance > t; every component gets at least one seed; growth in torch (default) or skimage watershed (`manyseg[instances]`) |
| `point_heads(heatmap="regression"/"segmentation", offsets=False)` | `heatmap` (1), `offset` (2: dy, dx) | `gaussian_heatmap(points, sigma)`, `heatmap_from_instances` (centroids rounded to pixels; fixed or size-scaled σ), `offset_targets` (to exact centroids) | `find_peaks` (max-pool NMS), `group_by_offsets` → instance ids, `panoptic_labels` |
| `density_head()` | `density` (1) | `density_map(points, sigma)`: each point's Gaussian sums to exactly 1 inside the image | `count_from_density` |

`decoders=None` shares the model decoder. `"separate"` gives each head its own `unet`. A name or
`{head: decoder}` mapping chooses decoders per head. Post-processing (peak picking, seeded
growth or watershed, grouping) is **not part of the exported graph**: export the model, then
run the post-processing functions on its outputs. Losses and object metrics (PQ, object F1,
over-/under-segmentation, point F1, count MAE) live in `manyseg-bench`
(`manyseg_bench.losses`, `manyseg_bench.evaluation.objects`, `manyseg_bench.tasks`).

> **Evidence (E009, Sen1Floods11, leave-events-out):** FiLM beat metadata-as-channels (+1.3
> water IoU, 2.9 SE, all folds); FiLM vs no metadata was +0.9 but under the 2 SE bar. Lat/lon added
> nothing on held-out events. **E013 (CloudSEN12+, unseen locations):** FiLM with lat/lon, day of
> year and solar angles was −0.08 val / −0.26 test vs no metadata, and lat/lon added nothing.
> **E016 (PASTIS, single-date crop types):** day-of-year by FiLM was +0.12 val mIoU (bar 0.99),
> the same as a shuffled-date control. Metadata conditioning has now shown no benefit on three
> datasets (flood events, unseen cloud scenes, crop phenology), so treat it as a tool to test on
> your data, not a default. The mechanism itself is verified end to end: a label-deciding metadata
> bit is learned to > 0.95 mIoU through the bench's real training recipe, and a shuffled bit stays
> at chance. The negative results are about the metadata, not the plumbing. Only use `coords` fields with geographic or event-level
> splits; otherwise the model can memorise locations and the evaluation will not show it.

## Inspecting a built model

`model.describe()` (or `manyseg.describe.describe(model)`) summarises the model stage by stage:
inputs and channel adapters, encoders (backbone, pretrained or random, feature levels), fusion
(type, op, fused strides, shared or independent encoders), metadata conditioning (fields, FiLM
targets, head bias), decoders (type, `out_stride`, width, levels read, `input_cross`) and heads
(type, channels, scope, scale, uncertainty, and the decoder that feeds each one), with
parameters per stage. It reads the constructed modules, not the `create_model` arguments, so it
shows what was actually built.

```python
print(model.describe())                       # indented text tree
print(model.describe(sample))                 # + real shapes at each stage boundary and FLOPs
mermaid = model.describe(format="mermaid")    # flowchart string
dot = model.describe(format="dot")            # Graphviz string

from manyseg.describe import model_graph, render_dot
render_dot(dot, "model.png")                  # needs the Graphviz `dot` binary on PATH
g = model_graph(model)                        # nodes and edges, for programmatic checks
```

With a `sample` (a tensor or dict, as for `model(sample)`), one `no_grad` forward pass in eval
mode records the padded encoder inputs, each encoder level, the fusion inputs (a `same_extent`
input shows up already resampled onto the primary grid), each decoder output, and each head's
raw and cropped output. Train/eval flags are restored afterwards. All three formats are plain
strings and need no extra dependencies. This model has a 4-channel raster, a FiLM metadata
vector and an FPN decoder of its own for one head:

```python
create_model(backbone="resnet18", inputs={
    "s2": {"channels": 4},
    "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}}},
    heads={"mask": {"type": "segmentation", "channels": 2},
           "boundary": {"type": "segmentation", "channels": 1, "decoder": "fpn"}})
```

```mermaid
flowchart LR
  in_s2(["<b>input: s2</b><br/>raster, 4 ch"])
  in_meta(["<b>input: meta</b><br/>vector, 1 col, film<br/>fields: doy (cyclic)<br/>encoding: width 2, MLP → 64<br/>4.35k params"])
  enc["<b>encoder</b><br/>resnet18 (random init)<br/>stem: 4 ch, 'repeat' weights from 3 ch<br/>levels: 64@/2, 64@/4, 128@/8, 256@/16, 512@/32<br/>11.18M params"]
  dec_decoder["<b>decoder</b><br/>unet, out_stride 1, width 16<br/>reads: /2, /4, /8, /16, /32<br/>input_cross: False<br/>FiLM: levels /2, /4, /8, /16, /32; output yes<br/>3.15M params"]
  dec_task_boundary["<b>decoder[boundary]</b><br/>fpn, out_stride 4, width 128<br/>reads: /4, /8, /16, /32<br/>FiLM: levels /4, /8, /16, /32; output yes<br/>1.87M params"]
  head_mask("<b>head: mask</b><br/>segmentation, pixel, 2 ch<br/>from: decoder<br/>scale: 1<br/>resample: /1 feature, none<br/>290 params")
  head_boundary("<b>head: boundary</b><br/>segmentation, pixel, 1 ch<br/>from: decoder[boundary]<br/>scale: 1<br/>resample: /4 feature, bilinear x4<br/>1.15k params")
  in_s2 --> enc
  enc -->|/2, /4, /8, /16, /32| dec_decoder
  enc -->|/4, /8, /16, /32| dec_task_boundary
  dec_decoder --> head_mask
  dec_task_boundary --> head_boundary
  in_meta -.->|FiLM| dec_decoder
  in_meta -.->|FiLM| dec_task_boundary
  classDef input fill:#e8f0fe,stroke:#5f6368
  classDef encoder fill:#e6f4ea,stroke:#5f6368
  classDef decoder fill:#fce8e6,stroke:#5f6368
  classDef head fill:#e0f2f1,stroke:#5f6368
  classDef vector fill:#f3e8fd,stroke:#5f6368
  class in_s2 input
  class in_meta vector
  class enc encoder
  class dec_decoder decoder
  class dec_task_boundary decoder
  class head_mask head
  class head_boundary head
```

## Export

```python
from manyseg.export import export_model
ep = export_model(model, example)            # dynamic H/W in [max_stride + 1, 4096]
ep = export_model(model, example, dynamic_batch=True)
ep = export_model(multi_input_model, {"optical": o, "sar": s})
ep = export_model(conditioned_model, {"s2": s2, "meta": meta})   # meta: (B, N), no H/W dims
```

`export_model` builds the `dynamic_shapes` spec itself and uses deferred runtime asserts. Some
timm blocks produce guards that are always true but that the export solver cannot prove.
Window-attention backbones (Swin, DaViT) cannot be exported with dynamic sizes. See
`backbone_compatibility.md`. For ONNX, pass the same spec to the dynamo exporter:
`torch.onnx.export(model, (x,), path, dynamo=True,
dynamic_shapes=manyseg.export.dynamic_shapes(model, x))` (see `examples/06_export.py`; the
parity check needs `onnxruntime`, which the `onnx` extra does not install).

### Distributed training

DDP needs nothing from ManySeg: wrap the model as usual. For FSDP2,
`manyseg.distributed.shard_model(model, mp_policy=..., mesh=...)` applies `fully_shard` per
stage (each encoder, the shared decoder, each task decoder, each head, the conditioner), then
the root. It is tested against DDP on CPU (gloo): parameters after a step match. BatchNorm
statistics stay per rank, as with DDP.

### Small batches

> **Evidence (E007, E007b; Pet and Sen1Floods11):** at batch 2, keeping the pretrained encoder's
> BatchNorm in eval mode (statistics frozen, affine weights still training) gave +0.8 (Pet)
> and +2.3 (Sen1Floods11) val mIoU at matched optimiser steps. Gradient accumulation alone did
> not help, because BN still sees micro-batches. Batch 2 × accumulate 8 with frozen encoder BN
> came within 0.2 mIoU of batch 16 on Pet and beat it on Sen1Floods11.

When memory forces batches of about 1–4, freeze the encoder's BN statistics after every
`model.train()`:

```python
model.train()
for m in model.backbone.modules():          # model.backbone.encoders for multi-input models
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()                            # running stats fixed; weights and biases still train
```

### Speed: compile and precision

> **Evidence (E014, idle RTX 4090, 512², 2 encoders × 5 decoders):** bf16 autocast was ≥ 1.3×
> faster than fp32 in 9 of 10 pairs (median 1.61×, inference batch 8). `torch.compile` on top
> gave a median 1.46× in bf16 (≥ 1.2× in 10/10) and only 1.22× in fp32. For training steps,
> compile gained only a median 1.17× and was sometimes slower. A cold compile cost 8–15 s per
> inference shape and 28–54 s per training shape.

Recommended for inference: `torch.autocast("cuda", torch.bfloat16)` +
`torch.compile(model, dynamic=True)` (one graph across input sizes; see the spatial policy).
At batch 1 in eager mode, fp32 can beat bf16 because cast overhead dominates small decoders;
compiling removes that. For training, use bf16 autocast; compile is optional.

## Quantisation

`quantize_pt2e` and `quantize_weights` need torchao (`pip install 'manyseg[quant]'`,
BSD-3-Clause). Without it, they raise an `ImportError` that names the extra. `model_size` and
`compare_outputs` work without torchao. In torch 2.14, PT2E moved out of `torch.ao` into
torchao, so both quantisation routes use it.

```python
from manyseg.quantize import compare_outputs, model_size, quantize_pt2e, quantize_weights

# Static int8 for CPU (PT2E): export -> prepare -> calibrate -> convert -> export.
ep = quantize_pt2e(model, example, calibration)         # calibration: iterable of inputs
ep = quantize_pt2e(multi_input_model, {"optical": o, "sar": s}, dict_batches)
ep = quantize_pt2e(swin_model, example, batches, dynamic=False)   # fixed size only
torch.export.save(ep, "model_int8.pt2")

torch._inductor.config.freezing = True
fast = torch.compile(ep.module())                       # oneDNN int8 kernels

# int8 weights on the eager module (CPU or GPU), no calibration.
q = quantize_weights(model)                             # "int8_weight_only" (default)
q = quantize_weights(model, "int8_dynamic")             # + int8 activations in Linear layers

model_size(ep), model_size(q)                           # serialized bytes
compare_outputs(model, ep, test_batches)
# {"max_abs_diff": ..., "mean_abs_diff": ..., "argmax_agreement": ...}
# argmax only for float (B, C) / (B, C, H, W); integer outputs get "exact_agreement";
# "dtype_changed" (and a warning) if the dtype differs; dict outputs: "seg/argmax_agreement"
```

`quantize_pt2e` returns an `ExportedProgram`, like `export_model`. H/W stay dynamic, and the
batch too with `dynamic_batch=True`. The default quantizer is torchao's
`X86InductorQuantizer`, with per-channel int8 weights and per-tensor uint8 activations. Pass
`quantizer=` to target another backend. Calling `ep.module()` eagerly runs the int8 graph as
quantise/dequantise pairs in float. That shows the accuracy cost, but it is about twice as
slow as float. The speedup only appears after `torch.compile` with inductor freezing.

> **Evidence (E008, trained ResNet-34 U-Nets, Pet and Sen1Floods11, 3 seeds):** bf16 and fp16
> cost nothing measurable. `quantize_weights` costs ≤ 0.05 mIoU. Compiled PT2E int8 costs
> 0.10–0.13 mIoU with ≥ 99.5% pixel agreement. On CPUs **without VNNI** (e.g. AMD Zen 3),
> inductor's u8 × s8 kernels saturate unless activations use a reduced range, and compiled int8
> silently lost 6.5 mIoU on Pet. `quantize_pt2e` therefore defaults to `reduce_range` when the
> *current* CPU lacks VNNI. If you quantise on one machine and deploy on another, pass
> `reduce_range=` for the deployment CPU (`True` is always safe; `False` is about 0.1 mIoU
> better only where VNNI exists). Always measure `compare_outputs` on the compiled module you
> deploy, not only on `ep.module()`.

A full-range program (`reduce_range=False`, the default on VNNI CPUs) warns when it is made, and
records the choice in the program (it survives `torch.export.save` / `load`). In the deployment
script, `manyseg.quantize.check_deployment(ep)` raises if this CPU cannot run that program
correctly. Window-attention backbones are rejected before export, because their
dynamic export hangs rather than fails.

`quantize_weights` stores Linear and Conv2d weights as int8, symmetric per output channel.
Depthwise convs stay float, because torchao rejects their weights and they are tiny anyway.
Conv weights are dequantised on the fly, so a CNN gets about 4x smaller but no faster. torchao
has no dynamic-activation conv, so `"int8_dynamic"` differs from weight-only only in Linear
layers. For a ResNet U-Net the two modes are identical. `conv=False` quantises Linear layers
only. The model is copied unless `inplace=True`.

Measured on CPU (4 threads, shared machine, so the timings are contended): a ResNet-34 U-Net
(ImageNet encoder, random decoder, 6 classes) at 256², with 16 calibration batches.

| | Size | Max / mean abs diff | Argmax agreement | Latency |
| --- | --- | --- | --- | --- |
| float | 99.5 MB (exported) | | | 56 ms eager, 37 ms compiled |
| `quantize_pt2e` | 26.5 MB (0.27x) | 1.94 / 0.12 | 0.961 (0.965 at 255×199) | 109 ms eager, 19 ms compiled |
| `quantize_weights` | 25.1 MB (0.26x of the state dict) | 0.61 / 0.06 | 0.980 | 66 ms eager |

A random decoder gives near-flat logits, so these agreement numbers are pessimistic. Check a
trained model on your own validation data.

## Registries

Decoders, heads, fusion ops, metadata field types and channel adapters register by name in a plain `dict`-backed
registry. Backbones do not need one, because timm's model registry is used as is.
Compatibility with a given timm model is discovered by probing (see
`architecture_review.md`), not from hand-maintained allow-lists.

## Non-goals

* Loss functions and training loops. These live in `manyseg-bench` or in user code.
* Data loading, augmentation, and georeferencing.
* A config-file DSL. Plain keyword arguments and dicts are enough, and they serialise to
  YAML or JSON if a user wants files.
