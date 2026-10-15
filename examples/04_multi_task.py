"""Multi-task: a segmentation map, an image-level regression with uncertainty, and a pixel
regression with its own decoder. Passing `heads=` makes the model return a dict.
"""

import torch
import torch.nn.functional as F

from manyseg import create_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(
        backbone="resnet18",
        pretrained=False,  # True downloads ImageNet weights
        in_channels=4,
        decoder="unet",    # shared by every head without its own "decoder" key
        heads={
            "landcover": {"type": "segmentation", "channels": 5},
            # image scope: pools the deepest encoder feature -> (B, 2): [mean, log_variance]
            "water_area": {"type": "regression", "channels": 1, "scope": "image",
                           "uncertainty": True},
            # a task-specific decoder for this head only (any registered decoder, or a dict
            # such as {"type": "unet", "out_stride": 2})
            "height": {"type": "regression", "channels": 1, "decoder": "fpn"},
        },
    ).to(device)

    b, h, w = 2, 96, 112
    x = torch.randn(b, 4, h, w, device=device)
    y_seg = torch.randint(0, 5, (b, h, w), device=device)
    y_area = torch.rand(b, 1, device=device)
    y_height = torch.rand(b, 1, h, w, device=device) * 20

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        out = model(x)
        mean, log_var = out["water_area"].chunk(2, dim=1)
        loss = (F.cross_entropy(out["landcover"], y_seg)
                + F.gaussian_nll_loss(mean, y_area, log_var.exp())  # heteroscedastic
                + F.smooth_l1_loss(out["height"], y_height))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step} loss {loss.item():.3f}")

    print({k: tuple(v.shape) for k, v in out.items()})
    # landcover (2, 5, H, W), water_area (2, 2), height (2, 1, H, W)
    model.eval()
    with torch.no_grad():
        mean, log_var = model(x)["water_area"].chunk(2, dim=1)
    std = log_var.mul(0.5).exp()
    print("water_area mean:", mean.squeeze(1).tolist(), "std:", std.squeeze(1).tolist())
    print(model.describe())  # shows which decoder feeds each head


if __name__ == "__main__":
    main()
