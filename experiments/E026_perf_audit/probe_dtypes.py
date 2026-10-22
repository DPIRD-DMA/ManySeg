"""E026 probe: output dtypes under CUDA bf16 autocast for ops the audit could not check on CPU."""
import torch
import torch.nn.functional as F
from torch import nn

from manyseg import create_model

dev = "cuda"
x = torch.randn(2, 8, 32, 32, device=dev, dtype=torch.bfloat16)
res = {}
with torch.autocast("cuda", dtype=torch.bfloat16):
    res["interpolate_nearest"] = F.interpolate(x, scale_factor=2, mode="nearest").dtype
    res["interpolate_bilinear"] = F.interpolate(x, size=(64, 64), mode="bilinear").dtype
    res["group_norm"] = nn.GroupNorm(4, 8).to(dev)(x).dtype
    res["batch_norm"] = nn.BatchNorm2d(8).to(dev)(x).dtype
    res["pixel_shuffle_cl_keeps_cl"] = F.pixel_shuffle(
        x.contiguous(memory_format=torch.channels_last), 2).is_contiguous(
        memory_format=torch.channels_last)
for k, v in res.items():
    print(f"{k}: {v}")

img = torch.randn(2, 3, 64, 64, device=dev)  # fp32 input, as the loader gives
for name, kw in [("fastai_unet", dict(decoder="fastai_unet")),
                 ("unet+input_cross", dict(decoder="unet", decoder_kwargs={"input_cross": True})),
                 ("unet", dict(decoder="unet")),
                 ("unet group norm", dict(decoder="unet", decoder_kwargs={"norm": "group"}))]:
    m = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2, **kw)
    m = m.to(dev)
    dec = m.decoder
    seen = {}
    dec.register_forward_hook(lambda mod, i, o: seen.update(dec=o.dtype) and None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(img)
    print(f"{name}: decoder output {seen.get('dec')}, model output {out.dtype}")
