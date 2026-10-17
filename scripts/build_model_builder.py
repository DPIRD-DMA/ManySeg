"""Generate docs/builder/data.js, the data the model builder page reads.

For every buildable backbone in docs/backbone_compatibility.json (plus the default, the presets'
encoder and the remote-sensing weights' architectures) it records the feature levels and the
encoder's parameter count, and for every decoder configuration the page offers it builds the
real decoder class on those feature specs (meta device, no weights) to record its output spec and
parameter count. Registries (decoders, fusion ops, metadata field types, temporal aggregations,
remote-sensing weights, reserved names) and presets are copied from the package, so the page
offers exactly what the installed ManySeg accepts.

    uv run python scripts/build_model_builder.py            # rewrite docs/builder/data.js
    uv run python scripts/build_model_builder.py --check    # exit 1 if it is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from manyseg.backbones import create_backbone
from manyseg.conditioning import FIELD_TYPES
from manyseg.conditioning.inject import used_levels
from manyseg.decoders import DECODERS
from manyseg.features import FeatureSpec
from manyseg.fusion.ops import FUSION_OPS
from manyseg.presets import PRESETS
from manyseg.registry import reserved_names
from manyseg.temporal import AGGREGATIONS
from manyseg.weights import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "builder" / "data.js"
# Not in the compatibility table but offered by the page: the create_model default, the presets'
# encoder and the remote-sensing weights' architectures.
EXTRA = ["resnet34", "pvt_v2_b0.in1k", "resnet152"]
WIDTHS = {"full": None, "half": [128, 64, 32, 16, 8], "quarter": [64, 32, 16, 8, 8]}


def _params(module: torch.nn.Module) -> float:
    return round(sum(p.numel() for p in module.parameters()) / 1e6, 3)


def _encoder(name: str) -> tuple[list[int], list[int], float]:
    with torch.device("meta"):
        enc = create_backbone(name, pretrained=False, in_channels=3)
    specs = enc.feature_specs
    return [s.stride for s in specs], [s.channels for s in specs], round(_params(enc), 2)


def _decoder_configs(specs):
    deepest = max(s.stride for s in specs)
    for os_ in (1, 2, 4):
        nblocks = (deepest // os_).bit_length() - 1
        for w, ch in WIDTHS.items():
            kw = {"out_stride": os_}
            if ch is not None:
                kw["channels"] = ch[:nblocks]
            yield f"unet/{os_}/{w}", "unet", kw
        yield f"unetplusplus/{os_}", "unetplusplus", {"out_stride": os_}
        yield f"fastai_unet/{os_}", "fastai_unet", {"out_stride": os_}
    for d in ("fpn", "segformer", "deeplabv3plus", "upernet"):
        yield d, d, {}


def _decoders(strides, channels) -> dict:
    """{config: [out channels, out stride, params (M), strides it reads] or an error string}.
    The read strides come from ManySeg's own gradient probe (``used_levels``): create_model only
    fuses, and only requires, the levels some decoder reads."""
    specs = [FeatureSpec(c, s) for c, s in zip(channels, strides)]
    out = {}
    for key, name, kw in _decoder_configs(specs):
        cls = DECODERS[name]
        sp = [FeatureSpec(3, 1, "input"), *specs] if getattr(cls, "wants_input", False) else specs
        try:
            torch.manual_seed(0)
            d = cls(sp, **kw)
        except (ValueError, AssertionError) as e:  # recorded: the page shows it as unbuildable
            out[key] = str(e)[:160]
            continue
        read = sorted(specs[i].stride for i in used_levels(d, specs, 3))
        out[key] = [d.out_spec.channels, d.out_spec.stride, _params(d), read]
    return out


def build(only: set[str] | None = None) -> dict:
    """The page's data; ``only``: rebuild just these backbones (a quick partial check)."""
    compat = json.loads((ROOT / "docs" / "backbone_compatibility.json").read_text())
    rows = [(x["name"], x["size_support"], x["export_dynamic"].split(":")[0], x["plain_vit"])
            for x in compat if x["buildable"]]
    known = {r[0] for r in rows}
    rows = [(n, "any", "ok", False) for n in EXTRA if n not in known] + rows
    backbones = []
    for name, size, export, vit in rows:
        if only is not None and name not in only:
            continue
        strides, channels, params = _encoder(name)
        backbones.append({"name": name, "strides": strides, "channels": channels,
                          "params_m": params, "size": size, "export": export, "vit": vit,
                          "dec": _decoders(strides, channels)})
        print(name, file=sys.stderr)
    return {
        "backbones": backbones,
        "decoders": sorted(DECODERS),
        "fusion_ops": sorted(FUSION_OPS),
        "field_types": sorted(FIELD_TYPES),
        "temporal": list(AGGREGATIONS),
        "unet_widths": WIDTHS,
        # names ManySeg rejects for inputs, heads and fields (besides keywords, '.', '_x')
        "reserved_names": sorted(n for n in reserved_names() if not n.startswith("_")),
        "weights": {k: {"model": w.model, "tag": w.tag, "bands": len(w.bands),
                        "units": w.units, "license": w.license}
                    for k, w in sorted(WEIGHTS.items())},
        "presets": {k: {"kwargs": p.kwargs, "description": p.description,
                        "measured": p.measured} for k, p in PRESETS.items()},
    }


def render(data: dict) -> str:
    return ("// Generated by scripts/build_model_builder.py from the installed ManySeg; "
            "do not edit.\nconst DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if data.js is out of date")
    args = ap.parse_args()
    text = render(build())
    if args.check:
        if OUT.read_text() != text:
            print(f"{OUT} is out of date: run scripts/build_model_builder.py", file=sys.stderr)
            return 1
        print(f"{OUT} is up to date", file=sys.stderr)
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
