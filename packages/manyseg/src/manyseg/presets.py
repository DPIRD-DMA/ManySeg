"""Named model presets: the configs on the measured accuracy/latency Pareto front of E018.

``create_model(**preset("balanced"), num_classes=2)``. Every preset is opt-in; the
``create_model`` defaults (ResNet-34 U-Net, SMP-parity widths) are unchanged. Any keyword can be
overridden by the caller (later keywords win in ``preset(...) | {...}``).

Numbers (``Preset.measured``) are from ``experiments/E018_pareto_presets``: val mIoU is the mean
of 3 seeds on Oxford-IIIT Pet and Sen1Floods11 with the bench ``fastai_like`` recipe; latency is
the compiled bf16 forward of a batch of 8 at 512x512 with 15 input channels on one (shared) RTX
4090. On the same scale the default ResNet-34 U-Net is 10.9 ms, 0.8231 Pet, 0.8901 Sen1Floods11.
Between presets, accuracy differences under ~0.3 mIoU are within seed noise; they are ordered
by latency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    kwargs: dict[str, Any]
    description: str
    measured: dict[str, Any] = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    "fast": Preset(
        {"backbone": "pvt_v2_b0.in1k", "decoder": "segformer"},
        "PVT-v2-B0 + SegFormer head: lowest GPU latency.",
        {"latency_ms": 4.5, "gflops": 17.5, "params_m": 3.8,
         "val_miou": {"oxford_pet": 0.8197, "sen1floods11": 0.9012}}),
    "compact": Preset(
        {"backbone": "pvt_v2_b0.in1k", "decoder": "unet",
         "decoder_kwargs": {"channels": [64, 32, 16, 8, 8]}},
        "PVT-v2-B0 + quarter-width U-Net: fewest FLOPs of the front.",
        {"latency_ms": 4.9, "gflops": 10.4, "params_m": 3.8,
         "val_miou": {"oxford_pet": 0.8229, "sen1floods11": 0.9009}}),
    "balanced": Preset(
        {"backbone": "pvt_v2_b0.in1k", "decoder": "unet",
         "decoder_kwargs": {"channels": [128, 64, 32, 16, 8]}},
        "PVT-v2-B0 + half-width U-Net: about 2x faster than the default ResNet-34 U-Net and "
        "more accurate on both E018 datasets.",
        {"latency_ms": 5.3, "gflops": 13.8, "params_m": 4.2,
         "val_miou": {"oxford_pet": 0.8254, "sen1floods11": 0.9024}}),
    "accurate": Preset(
        {"backbone": "pvt_v2_b0.in1k", "decoder": "unet"},
        "PVT-v2-B0 + full-width U-Net: best on Pet; on Sen1Floods11 tied with 'balanced'.",
        {"latency_ms": 7.3, "gflops": 27.2, "params_m": 5.7,
         "val_miou": {"oxford_pet": 0.8268, "sen1floods11": 0.9015}}),
}


def preset(name: str) -> dict[str, Any]:
    """``create_model`` keyword arguments for a named preset (a fresh copy, safe to modify)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name].kwargs)
