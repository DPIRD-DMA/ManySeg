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
  in front of one shared backbone. Inputs must share spatial size.
* **Intermediate fusion**: each input has its own backbone. Features with the same stride
  are fused by an op (`concat`, `add`, `gated`, `attention`, `cross_attention`) whose output
  has the primary input's channels. Strides that are not selected, or that exist only in the
  primary, pass the primary feature through unchanged. The first input in `inputs` is the
  primary.
* **Late fusion**: `fusion={"type": "late", "op": ...}`. Each input has its own backbone and
  decoder (same type and options). The decoder outputs are fused before the heads.
* Inputs without a `backbone` key use the top-level `backbone=` argument.
* **Shared encoder**: `{"backbone": "shared:optical"}` reuses another input's backbone weights.
  Each input keeps its own input adapter.

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
| `film` (default) | FiLM `x·(1+γ(e)) + β(e)` per channel on every encoder feature level the decoder reads (just before it enters the decoder) and on the decoder output. Applied around every decoder: the shared one, per-head task decoders and the late-fusion decoder. `{"on": ["output"]}` limits it to the output. | exact identity |
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
reads are found once at construction by a gradient probe (for example, DeepLabV3+ reads only
strides 4 and 32), so every FiLM parameter receives a gradient. All γ/β projections start at
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
* Backbones that cannot accept a size at all (for example, fixed-size ViTs without dynamic
  position embeddings) raise a clear error at construction or call time. They never silently
  resize.

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
peaks = find_peaks(out["heatmap"], threshold=0.5)        # [(points (K,2), scores)] per image
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
`backbone_compatibility.md`.

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
