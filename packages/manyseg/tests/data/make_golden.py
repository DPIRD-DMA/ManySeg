"""Write the golden saved-model file for test_saved_files_load.py (once per save format).

    uv run python packages/manyseg/tests/data/make_golden.py

The file is committed: it is what a user saved with this ManySeg version, and every later
version must keep loading it (migrating its config if keys were renamed) to the same outputs.
Do not regenerate it to make a failing test pass.
"""

from pathlib import Path

import manyseg
import torch
from manyseg import create_model
from safetensors.torch import save_file

HERE = Path(__file__).parent


def main():
    torch.manual_seed(0)
    model = create_model(
        backbone="test_resnet", pretrained=False,
        decoder="unet", decoder_kwargs={"channels": [16, 16, 8, 8, 8]},  # small file
        inputs={"img": {"in_channels": 4}, "doy": {"kind": "vector", "num_features": 1}},
        heads={"seg": {"type": "segmentation", "out_channels": 3},
               "density": {"type": "regression", "out_channels": 1, "activation": "softplus",
                           "hidden": [8], "dilation": 2, "bias_init": -4.0},
               "label": {"type": "classification", "out_channels": 2, "scope": "image"}},
    ).eval()
    with torch.no_grad():  # non-trivial BatchNorm statistics, so a mis-loaded buffer shows
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.8, 1.2)
    name = f"golden_format{manyseg.io.FORMAT}"  # one file per save format
    path = manyseg.save(model, HERE / f"{name}.safetensors",
                        normalize=[{"type": "scale", "value": 10000}],
                        activations={"seg": "softmax", "label": "softmax"},
                        classes={"seg": ["a", "b", "c"], "label": ["no", "yes"]})
    g = torch.Generator().manual_seed(1)
    x = {"img": torch.randn(2, 4, 45, 61, generator=g), "doy": torch.rand(2, 1, generator=g)}
    with torch.no_grad():
        out = model(x)
    save_file({"in_img": x["img"], "in_doy": x["doy"], **{f"out_{k}": v for k, v in out.items()}},
              str(HERE / f"{name}_io.safetensors"))
    print(path, path.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
