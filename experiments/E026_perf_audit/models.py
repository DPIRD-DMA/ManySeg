"""E026 model configurations and matching random inputs."""

from __future__ import annotations

import torch

from manyseg import create_model

R34 = dict(backbone="resnet34")
MODELS = {
    "unet_r34": dict(R34, decoder="unet"),
    "unet_r34_cross": dict(R34, decoder="unet", decoder_kwargs={"input_cross": True}),
    "unet_r34_gn": dict(R34, decoder="unet", decoder_kwargs={"norm": "group"}),
    "fastai_r34": dict(R34, decoder="fastai_unet"),
    "unetpp_r34": dict(R34, decoder="unetplusplus"),
    "fpn_r34": dict(R34, decoder="fpn"),
    "deeplab_r34": dict(R34, decoder="deeplabv3plus"),
    "upernet_r34": dict(R34, decoder="upernet"),
    "segformer_r34": dict(R34, decoder="segformer"),
    "segformer_pvt": dict(backbone="pvt_v2_b0", decoder="segformer"),  # the "fast" preset
    "unet_pvt": dict(backbone="pvt_v2_b0", decoder="unet"),  # the "balanced" preset
    "unet_convnext": dict(backbone="convnext_tiny", decoder="unet"),
    "unet_edgenext": dict(backbone="edgenext_small", decoder="unet"),
    "fastai_edgenext": dict(backbone="edgenext_small", decoder="fastai_unet"),
    "unet_vit": dict(backbone="vit_small_patch16_224", decoder="unet",
                     backbone_kwargs={"img_size": 512}),
    # E027: fusion with a coarser same_extent input (resized, then concatenated)
    "late_concat_extent": dict(inputs={"a": {"in_channels": 4, "backbone": "resnet18"},
                                       "b": {"in_channels": 2, "backbone": "resnet18",
                                             "alignment": "same_extent"}},
                               fusion={"type": "late", "op": "concat"}, decoder="unet"),
    "inter_concat_extent": dict(inputs={"a": {"in_channels": 4, "backbone": "resnet18"},
                                        "b": {"in_channels": 2, "backbone": "resnet18",
                                              "alignment": "same_extent"}},
                                fusion={"type": "intermediate", "op": "concat"}, decoder="unet"),
    "unet_vit256": dict(backbone="vit_small_patch16_224", decoder="unet",
                        backbone_kwargs={"img_size": 256}),
    "unet_r34_bilinear": dict(R34, decoder="unet", decoder_kwargs={"upsample": "bilinear"}),
    "unet_r34_aligned": dict(R34, decoder="unet", decoder_kwargs={"upsample": "aligned"}),
    # time series: (B, T, C, H, W), mean over dates (E021's default)
    "unet_r34_ts": dict(inputs={"s2": {"kind": "timeseries", "in_channels": 13,
                                       "backbone": "resnet34", "temporal": "mean"}},
                        decoder="unet"),
    # two rasters fused by cross-attention
    "xattn_r18": dict(inputs={"a": {"in_channels": 4, "backbone": "resnet18"},
                              "b": {"in_channels": 2, "backbone": "resnet18"}},
                      fusion={"type": "intermediate", "op": "cross_attention",
                              "strides": [8, 16, 32]}, decoder="unet"),
}


def build(name: str, in_channels: int, num_classes: int):
    cfg = dict(MODELS[name])
    if "inputs" not in cfg:
        cfg["in_channels"] = in_channels
    return create_model(pretrained=False, num_classes=num_classes, **cfg)


def make_input(name: str, batch: int, size: int, in_channels: int, device, dtype=torch.float32,
               dates: int = 6, generator=None):
    def r(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    cfg = MODELS[name]
    if "inputs" not in cfg:
        return r(batch, in_channels, size, size)
    out = {}
    for k, c in cfg["inputs"].items():
        if c.get("kind") == "timeseries":
            out[k] = r(batch, dates, c["in_channels"], size, size)
        elif c.get("alignment") == "same_extent":  # a coarser grid over the same extent
            out[k] = r(batch, c["in_channels"], size // 2, size // 2)
        else:
            out[k] = r(batch, c["in_channels"], size, size)
    return out
