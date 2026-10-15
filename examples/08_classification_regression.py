"""Image-level classification and regression: no decoder, timm-equivalent when simple.

Heads with ``"scope": "image"`` pool the deepest encoder feature; when every head is an image
head, no decoder is built. With one tensor input and one head this matches
``timm.create_model(backbone, num_classes=K, in_chans=C)`` (same encoder, weights and stem
adaptation; timm's ConvNeXt head adds a LayerNorm; E019 compares them). What timm
cannot express composes as for dense models:

1. a multi-label land-cover classifier on 12-band Sentinel-2 (sigmoid + BCE outside the model);
2. the same with Sentinel-2 at two native resolutions as two inputs, Sentinel-1 at 10 m, the
   acquisition day of year via FiLM, a crop-yield regression head with uncertainty, and a
   pixel head next to the image heads (the decoder is then built for the pixel head only).
"""

import math

import torch
import torch.nn.functional as F
from manyseg import create_model

CLASSES = 19


def train_steps(model, batch, loss_fn, steps=3):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        loss = loss_fn(model(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  step {step} loss {loss.item():.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. multi-label classifier (no decoder)")
    clf = create_model(backbone="resnet18", pretrained=False, in_channels=12,
                       heads={"labels": {"type": "classification", "channels": CLASSES,
                                         "scope": "image"}}).to(device)
    x = torch.randn(4, 12, 120, 120, device=device)
    y = (torch.rand(4, CLASSES, device=device) < 0.2).float()
    print("  decoder:", clf.decoder)
    train_steps(clf, x, lambda out: F.binary_cross_entropy_with_logits(out["labels"], y))

    print("2. multi-resolution, multi-sensor, metadata, mixed heads")
    model = create_model(
        inputs={
            "s2_10m": {"channels": 4, "backbone": "resnet18"},
            "s2_20m": {"channels": 6, "backbone": "resnet18", "alignment": "same_extent"},
            "s1": {"channels": 2, "backbone": "resnet18"},
            "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}},
        },
        pretrained=False,
        fusion={"type": "intermediate", "op": "concat"},
        heads={
            "labels": {"type": "classification", "channels": CLASSES, "scope": "image"},
            "harvest": {"type": "regression", "channels": 1, "scope": "image",
                        "uncertainty": True},          # [mean, log-variance]
            "water": {"type": "segmentation", "channels": 2},  # pixel head: builds a decoder
        },
    ).to(device)
    doy = torch.tensor([[45.0], [120.0], [200.0], [300.0]], device=device)
    batch = {"s2_10m": torch.randn(4, 4, 120, 120, device=device),
             "s2_20m": torch.randn(4, 6, 60, 60, device=device),   # native 20 m grid
             "s1": torch.randn(4, 2, 120, 120, device=device),
             "meta": model.conditioner.encoders["meta"].pack({"doy": doy})}
    harvest = torch.rand(4, 1, device=device) * 8
    water = torch.randint(0, 2, (4, 120, 120), device=device)

    def loss(out):
        mean, logvar = out["harvest"].chunk(2, dim=1)
        nll = 0.5 * (logvar + (harvest - mean) ** 2 / logvar.exp()).mean() + 0.5 * math.log(
            2 * math.pi)
        return (F.binary_cross_entropy_with_logits(out["labels"], y) + nll
                + F.cross_entropy(out["water"], water))

    train_steps(model, batch, loss)
    with torch.no_grad():
        out = model.eval()(batch)
    print("  outputs:", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
