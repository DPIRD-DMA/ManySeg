"""Conditioning a segmentation model on acquisition metadata (CPU, random weights).

A Sentinel-2-like 13-band raster plus a metadata vector (day of year, sun elevation, lat/lon,
sensor id). Shows the packed-vector convention, missing values, the three injection modes and
export with a vector input.
"""

import torch
from manyseg import create_model
from manyseg.export import export_model

FIELDS = {
    "doy": {"type": "cyclic", "period": 365.25},
    "sun_elev": {"type": "continuous", "mean": 45.0, "std": 15.0},
    "latlon": {"type": "coords"},
    "sensor": {"type": "categorical", "num": 3},
}


def build(conditioning="film"):
    return create_model(
        backbone="resnet18", pretrained=False,
        inputs={"s2": {"channels": 13},
                "meta": {"kind": "vector", "fields": FIELDS, "dropout": 0.1,
                         "conditioning": conditioning}},
        heads={"water": {"type": "segmentation", "channels": 2},
               "flooded_fraction": {"type": "regression", "channels": 1, "scope": "image"}},
    ).eval()


def main():
    model = build("film")
    enc = model.conditioner.encoders["meta"]
    b = 2
    meta = enc.pack({"doy": torch.tensor([200.0, 15.0]),
                     "sun_elev": torch.tensor([55.0, float("nan")]),  # NaN = missing
                     "latlon": torch.tensor([[-31.95, 115.86], [52.52, 13.40]]),
                     "sensor": torch.tensor([0, 2])})
    print(f"packed metadata {tuple(meta.shape)}: columns for fields {list(enc.field_names)}")
    s2 = torch.randn(b, 13, 97, 131)

    out = model({"s2": s2, "meta": meta})
    print({k: tuple(v.shape) for k, v in out.items()})

    # A fresh FiLM model equals the unconditioned model exactly, whatever the metadata is.
    no_meta = model({"s2": s2, "meta": torch.full_like(meta, float("nan"))})
    print("identity at init:", torch.equal(out["water"], no_meta["water"]))

    # Explicit mask (1 = present, one column per field): here only the date is known.
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(b, 4)
    model({"s2": s2, "meta": meta, "meta_mask": mask})

    for mode in ({"mode": "head", "pixel": True}, "channels"):
        m = build(mode)
        print(mode, {k: tuple(v.shape) for k, v in m({"s2": s2, "meta": meta}).items()},
              "s2 backbone in_channels:", m.backbone.in_channels)

    ep = export_model(model, {"s2": torch.randn(1, 13, 64, 96), "meta": meta[:1]})
    y = ep.module()({"s2": torch.randn(1, 13, 97, 131), "meta": meta[1:]})
    print("exported; dynamic H/W output:", tuple(y["water"].shape))


if __name__ == "__main__":
    main()
