"""13-band Sentinel-2-like input on an RGB-pretrained backbone.

`channel_adapter` decides how the pretrained 3-channel stem becomes a 13-channel one. It only
rewrites pretrained weights, so it matters with `pretrained=True`. `repeat` (default, timm's
rule) cycles the RGB filters and rescales by 3/C; in E004 (13-band S2) nothing beat it.

* "mean": every band gets the mean RGB filter.
* "rgb_plus_zero": bands 4+ start at zero, so the model starts identical to the RGB model.
  Useful when the first three bands are RGB and you fine-tune an RGB checkpoint.
* "rgb_plus_random": RGB filters kept, other bands small random.
* "projection": a learned 1x1 conv C -> 3 in front of the stem. Harmful in E004.
"""

import torch
import torch.nn.functional as F

from manyseg import create_model

BANDS = 13  # B1..B12 + B8A


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        backbone="resnet18",
        pretrained=False,          # True: ImageNet stem adapted to 13 bands by `channel_adapter`
        in_channels=BANDS,
        channel_adapter="repeat",  # the default; see the docstring for the alternatives
        num_classes=2,             # e.g. water / not water
    ).to(device)

    # Normalise per band (mean 0, std 1) before the model: padding uses 0.
    x = torch.rand(2, BANDS, 96, 120, device=device) * 3000
    x = (x - x.mean((0, 2, 3), keepdim=True)) / x.std((0, 2, 3), keepdim=True)
    y = torch.randint(0, 2, (2, 96, 120), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step} loss {loss.item():.3f}")

    print("backbone in_channels:", model.backbone.in_channels, "| output:", tuple(model(x).shape))


if __name__ == "__main__":
    main()
