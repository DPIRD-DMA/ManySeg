# ManySeg public API design

Status: draft v0.1 (Phase 1). Interfaces marked *(phase 4)* are designed but not yet implemented.

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

The head `type` sets the default activation (none: losses expect logits) and the
initialisation. `scope` is `"pixel"` (default) or `"image"`.

### Multiple inputs *(phase 4)*

```python
model = create_model(
    inputs={
        "optical": {"channels": 4, "backbone": "convnext_tiny", "pretrained": True},
        "sar":     {"channels": 2, "backbone": "resnet18",      "pretrained": True},
    },
    fusion={"type": "gated", "levels": [2, 3, 4]},   # pyramid levels to fuse (1 = stride 2)
    decoder="unet",
    heads={...},
)
out = model({"optical": o, "sar": s})
```

* **Early fusion**: `fusion={"type": "early"}`. Inputs are concatenated on the channel axis
  in front of one shared backbone. Inputs must share spatial size.
* **Intermediate fusion**: each input has its own backbone. The selected pyramid levels are
  fused by a `Fusion` op (`concat`, `add`, `gated`, `attention`, `cross_attention`). Levels that
  are not selected use the `primary` input only.
* **Late fusion**: `fusion={"type": "late", "op": ...}`. Each input has its own
  backbone and decoder, and the decoder outputs are fused before the heads.
* **Shared encoder**: `{"backbone": "shared:optical"}` reuses another input's backbone weights.
  Each input keeps its own input adapter.

#### Spatial alignment

Inputs with different pixel sizes are **not** assumed to cover the same ground extent. Each
input declares an alignment policy:

| Policy | Meaning |
| --- | --- |
| `"same_grid"` (default) | Inputs must have identical `(H, W)`. A mismatch raises an error. |
| `"same_extent"` | The input covers the same extent as the reference input at a different resolution. Features are resampled (bilinear, `align_corners=False`) onto the reference grid at each fused level. |

Inputs that are co-registered only partially must be cropped or resampled by the user's data
pipeline. Reprojecting or georeferencing data is outside the scope of a model library.

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

## Registries

Decoders, heads, fusion ops and channel adapters register by name in a plain `dict`-backed
registry. Backbones do not need one, because timm's model registry is used as is.
Compatibility with a given timm model is discovered by probing (see
`architecture_review.md`), not from hand-maintained allow-lists.

## Non-goals

* Loss functions and training loops. These live in `manyseg-bench` or in user code.
* Data loading, augmentation, and georeferencing.
* A config-file DSL. Plain keyword arguments and dicts are enough, and they serialise to
  YAML or JSON if a user wants files.
