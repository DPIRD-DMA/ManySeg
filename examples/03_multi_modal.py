"""Multi-modal input: optical + half-resolution SAR + a metadata vector (the README example).

Each raster has its own encoder; features are fused per stride by a gated op. SAR covers the
same ground at half the resolution (`same_extent`), so its features are resampled onto the
optical grid. The metadata vector conditions the decoder through FiLM.
"""

import shutil
import tempfile
from pathlib import Path

import torch
from manyseg import create_model
from manyseg.describe import render_dot
from manyseg.instances import instance_heads


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        pretrained=False,  # True downloads ImageNet weights for every encoder
        inputs={
            "optical": {"in_channels": 4, "backbone": "convnext_atto"},  # README: convnext_tiny
            "sar": {"in_channels": 2, "backbone": "resnet18", "alignment": "same_extent"},
            "meta": {"kind": "vector", "fields": {  # non-raster conditioning, FiLM by default
                "doy": {"type": "cyclic", "period": 365.25},
                "sun_elev": {"type": "continuous", "mean": 45.0, "std": 15.0},
                "latlon": {"type": "coords"},
            }},
        },
        fusion={"type": "intermediate", "op": "gated"},  # first input is the primary grid
        decoder="unet",
        heads={
            **instance_heads(decoders={"boundary": "fpn"}),  # extent, boundary, distance
            "water_area": {"type": "regression", "out_channels": 1, "scope": "image"},
        },
    ).to(device)

    b, h, w = 2, 96, 128
    # One packed (B, N) float tensor per vector input; NaN marks a missing value.
    meta = model.pack_metadata("meta", {
        "doy": torch.tensor([200.0, 15.0]),
        "sun_elev": torch.tensor([55.0, float("nan")]),
        "latlon": torch.tensor([[-31.95, 115.86], [-33.86, 151.21]]),
    }).to(device)
    x = {"optical": torch.randn(b, 4, h, w, device=device),
         "sar": torch.randn(b, 2, h // 2, w // 2, device=device),  # half resolution
         "meta": meta}

    out = model(x)
    print({k: tuple(v.shape) for k, v in out.items()})
    # extent / boundary / distance: (B, 1, H, W) on the optical grid; water_area: (B, 1)

    sample = {k: v[:1] for k, v in x.items()}
    print(model.describe(sample))

    if shutil.which("dot") is None:
        print("Graphviz 'dot' not on PATH: skipping the PNG (the DOT string still works)")
        return
    png = Path(tempfile.mkdtemp()) / "multi_modal.png"
    render_dot(model.describe(format="dot"), str(png))
    print("diagram written to", png)


if __name__ == "__main__":
    main()
