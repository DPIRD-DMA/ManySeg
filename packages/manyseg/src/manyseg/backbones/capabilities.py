"""Empirical capability probing of timm backbones (no hand-maintained allow-lists).

``probe_backbone`` builds a model with random weights and runs a few forwards to classify
what it supports. Expensive checks (export) are optional. The generated table lives in
``docs/backbone_compatibility.md`` (``scripts/backbone_compat.py``).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch

PROBE_SIZES = ((64, 64), (97, 131), (160, 96))


@dataclass
class BackboneCapabilities:
    name: str
    buildable: bool
    error: str = ""
    feature_strides: tuple[int, ...] = ()
    feature_channels: tuple[int, ...] = ()
    native_layout: str = ""
    plain_vit: bool = False
    size_support: str = ""  # "any" | "fixed" | "partial"
    sizes_ok: list = field(default_factory=list)
    sizes_failed: list = field(default_factory=list)
    params_m: float = 0.0
    export_dynamic: str = "not probed"
    probe_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def probe_backbone(name: str, sizes=PROBE_SIZES, export: bool = False,
                   **kwargs) -> BackboneCapabilities:
    from ..models import create_model

    t0 = time.perf_counter()
    try:
        model = create_model(backbone=name, pretrained=False, num_classes=2, **kwargs).eval()
    except Exception as e:  # noqa: BLE001 - classification, not control flow
        return BackboneCapabilities(name, False, error=f"{type(e).__name__}: {str(e)[:200]}")
    bb = model.backbone
    caps = BackboneCapabilities(
        name, True,
        feature_strides=tuple(s.stride for s in bb.feature_specs),
        feature_channels=tuple(s.channels for s in bb.feature_specs),
        native_layout="NHWC" if bb.nhwc else "NCHW",
        plain_vit=bool(bb.is_plain),
        params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    )
    for hw in sizes:
        try:
            out = model(torch.randn(1, bb.in_channels, *hw))
            ok = tuple(out.shape[-2:]) == tuple(hw)
        except Exception as e:  # noqa: BLE001
            ok = False
            caps.error = caps.error or f"{type(e).__name__}: {str(e)[:160]}"
        (caps.sizes_ok if ok else caps.sizes_failed).append(list(hw))
    caps.size_support = ("any" if not caps.sizes_failed else
                         "fixed" if not caps.sizes_ok else "partial")
    if export and caps.size_support == "any":
        from ..export import export_model

        try:
            ep = export_model(model, torch.randn(1, bb.in_channels, 64, 96), max_size=1024)
            x = torch.randn(1, bb.in_channels, 97, 131)
            same = torch.allclose(ep.module()(x), model(x), atol=1e-3, rtol=1e-3)
            caps.export_dynamic = "ok" if same else "mismatch"
        except Exception as e:  # noqa: BLE001
            caps.export_dynamic = f"fail: {type(e).__name__}"
    caps.probe_seconds = round(time.perf_counter() - t0, 1)
    return caps
