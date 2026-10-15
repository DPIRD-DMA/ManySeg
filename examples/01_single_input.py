"""Single raster input: any timm backbone, any channel count, any H x W.

Builds a 4-band (RGB + NIR) U-Net, runs it on an odd, non-square tile, takes one training
step on random data and prints what was built.
"""

import torch
import torch.nn.functional as F
from manyseg import create_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        backbone="resnet18",  # any timm name: "convnext_tiny", "efficientnet_b0", "pvt_v2_b0", ...
        pretrained=False,     # True downloads ImageNet weights and adapts the stem to 4 bands
        in_channels=4,
        decoder="unet",       # "fpn", "deeplabv3plus", "upernet", "segformer", "fastai_unet", ...
        num_classes=3,
    ).to(device)

    x = torch.randn(2, 4, 97, 131, device=device)         # odd and non-square: padded and cropped
    y = torch.randint(0, 3, (2, 97, 131), device=device)  # class ids
    print("logits:", tuple(model(x).shape))                # (2, 3, 97, 131)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"one step, loss {loss.item():.3f}")

    print(model.describe(x[:1]))  # stage by stage, with real shapes and FLOPs for this sample


if __name__ == "__main__":
    main()
