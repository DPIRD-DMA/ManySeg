"""E026 performance changes: each keeps the values of the code it replaced."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from manyseg import create_model
from manyseg.instances.postprocess import _neighbour_max, connected_components
from torch import nn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _segformer_reference(dec, features):
    """The concat-then-fuse SegFormer decoder, as before E026."""
    size = features[dec.levels[0]].shape[-2:]
    outs = []
    for i, lin in zip(dec.levels, dec.linear):
        y = lin(features[i])
        if y.shape[-2:] != size:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        outs.append(y)
    return dec.dropout(dec.fuse(torch.cat(outs, dim=1)))


@pytest.mark.parametrize("backbone", ["resnet18", "pvt_v2_b0"])
def test_segformer_per_level_fuse_equals_concat(backbone):
    torch.manual_seed(0)
    m = create_model(backbone=backbone, pretrained=False, in_channels=3, num_classes=2,
                     decoder="segformer").double().eval()
    feats = m.backbone(torch.randn(2, 3, 67, 61, dtype=torch.float64))
    got = m.decoder(feats)
    want = _segformer_reference(m.decoder, feats)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("backbone", ["resnet18", "pvt_v2_b0"])
def test_segformer_per_level_fuse_gradients_equal_concat(backbone):
    torch.manual_seed(0)
    m = create_model(backbone=backbone, pretrained=False, in_channels=3, num_classes=2,
                     decoder="segformer").double().train()
    m.decoder.dropout.p = 0.0
    feats = [f.detach() for f in m.backbone(torch.randn(2, 3, 64, 64, dtype=torch.float64))]
    grads = []
    for fn in (m.decoder, lambda f: _segformer_reference(m.decoder, f)):
        m.decoder.zero_grad()
        fn(feats).square().sum().backward()
        grads.append([p.grad.clone() for p in m.decoder.parameters()])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)


@cuda
@pytest.mark.parametrize("kwargs", [
    {"decoder": "fastai_unet"},
    {"decoder": "unet", "decoder_kwargs": {"input_cross": True}},
    {"decoder": "unet", "decoder_kwargs": {"norm": "group"}},
    {"decoder": "upernet"},
    {"decoder": "deeplabv3plus"},
    {"decoder": "segformer"},
], ids=lambda k: k["decoder"] + str(k.get("decoder_kwargs", "")))
def test_decoders_stay_bf16_under_cuda_autocast(kwargs):
    """CUDA autocast runs bilinear upsampling and group_norm in float32, and the input image
    is float32: none of that may leak into the decoder's activations (E026)."""
    m = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2,
                     **kwargs).cuda()
    seen = []
    for mod in m.decoder.modules():
        if isinstance(mod, nn.Conv2d | nn.ReLU):
            mod.register_forward_hook(lambda _m, _i, o: seen.append(o.dtype) and None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(torch.randn(2, 3, 64, 64, device="cuda"))
    assert seen and all(d == torch.bfloat16 for d in seen), set(seen)


@cuda
def test_fastai_pixel_shuffle_keeps_channels_last():
    m = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2,
                     decoder="fastai_unet").cuda().to(memory_format=torch.channels_last)
    shuf = m.decoder.blocks[0].shuf
    x = torch.randn(2, shuf.conv.in_channels, 8, 8, device="cuda").contiguous(
        memory_format=torch.channels_last)
    assert shuf(x).is_contiguous(memory_format=torch.channels_last)


def test_cross_attention_tokens_are_contiguous():
    m = create_model(pretrained=False, num_classes=2, decoder="unet",
                     inputs={"a": {"in_channels": 4, "backbone": "resnet18"},
                             "b": {"in_channels": 2, "backbone": "resnet18"}},
                     fusion={"type": "intermediate", "op": "cross_attention",
                             "strides": [16, 32]})
    op = next(mod for mod in m.modules() if type(mod).__name__ == "CrossAttentionFusion")
    assert op._tokens(torch.randn(2, 64, 5, 7)).is_contiguous()


def test_temporal_mean_bf16_matches_fp32_upcast():
    from manyseg.temporal import MeanAggregation

    torch.manual_seed(0)
    f = torch.randn(2, 5, 8, 6, 7).to(torch.bfloat16)
    f[0, 1, 0, 0, 0] = float("nan")  # a masked NaN must not leak
    mask = torch.tensor([[True, False, True, True, False], [False] * 5])
    got = MeanAggregation()(f, mask)
    m = mask[:, :, None, None, None]
    want = (torch.where(m, f.float(), 0.0).sum(1) / mask.sum(1).clamp(min=1)[:, None, None, None]
            ).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def _cc_reference(mask):
    """Label propagation without pointer jumping (before E026)."""
    b, h, w = mask.shape
    lab = torch.arange(1, h * w + 1).view(1, h, w).expand(b, h, w)
    lab = torch.where(mask, lab, 0)
    while True:
        nxt = torch.where(mask, torch.maximum(lab, _neighbour_max(lab)), 0)
        if torch.equal(nxt, lab):
            return lab
        lab = nxt


@pytest.mark.parametrize("kind", ["snake", "random", "empty", "full"])
def test_connected_components_pointer_jumping_same_labels(kind):
    g = torch.Generator().manual_seed(0)
    if kind == "snake":
        m = torch.zeros(2, 40, 40, dtype=torch.bool)
        m[:, ::4, :] = True
        for r in range(0, 36, 4):
            m[:, r:r + 5, 39 if (r // 4) % 2 == 0 else 0] = True
    elif kind == "random":
        m = torch.rand(3, 37, 41, generator=g) > 0.45
    elif kind == "empty":
        m = torch.zeros(1, 9, 9, dtype=torch.bool)
    else:
        m = torch.ones(1, 9, 11, dtype=torch.bool)
    assert torch.equal(connected_components(m), _cc_reference(m))


def _ckpt_model(backbone):
    torch.manual_seed(0)
    return create_model(backbone=backbone, pretrained=False, in_channels=3, num_classes=2,
                        decoder="unet").double().train()


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_atto", "efficientnet_b0"])
def test_grad_checkpointing_keeps_values_and_bn_statistics(backbone):
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    runs = []
    for ckpt in (False, True):
        m = _ckpt_model(backbone)
        if ckpt:
            m.set_grad_checkpointing()
        out = m(x)
        out.square().mean().backward()
        bn = {k: v.clone() for k, v in m.state_dict().items()
              if "running" in k or "num_batches" in k}
        runs.append((out.detach(), [p.grad.clone() for p in m.parameters()], bn))
    (o1, g1, s1), (o2, g2, s2) = runs
    torch.testing.assert_close(o2, o1, rtol=1e-12, atol=1e-12)
    for a, b in zip(g2, g1):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)
    assert s1.keys() == s2.keys()
    for k in s1:  # recomputation must not update the running statistics a second time
        torch.testing.assert_close(s2[k], s1[k], rtol=0, atol=0)


def test_grad_checkpointing_off_restores_inplace_activations():
    m = _ckpt_model("resnet18")
    before = {n: mod.inplace for n, mod in m.named_modules() if hasattr(mod, "inplace")}
    m.set_grad_checkpointing(True)
    assert not any(mod.inplace for n, mod in m.backbone.named_modules()
                   if hasattr(mod, "inplace"))
    m.set_grad_checkpointing(False)
    assert {n: mod.inplace for n, mod in m.named_modules() if hasattr(mod, "inplace")} == before
    hooks = [mod for mod in m.modules() if mod._forward_hooks or mod._forward_pre_hooks]
    assert hooks == []


def test_grad_checkpointing_shared_encoder_bn_statistics():
    """A Siamese input pair shares one encoder (and its BN with share_norm): the BN layers are
    recomputed twice in one backward pass, and must still end as without checkpointing."""
    x = {"t1": torch.randn(2, 3, 64, 64, dtype=torch.float64),
         "t2": torch.randn(2, 3, 64, 64, dtype=torch.float64)}
    runs = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        m = create_model(pretrained=False, num_classes=2, decoder="unet",
                         inputs={"t1": {"in_channels": 3, "backbone": "resnet18"},
                                 "t2": {"in_channels": 3, "backbone": "shared:t1",
                                        "share_norm": True}},
                         fusion={"type": "intermediate", "op": "diff"}).double().train()
        if ckpt:
            m.set_grad_checkpointing()
        m(x).square().mean().backward()
        runs.append({k: v.clone() for k, v in m.state_dict().items()
                     if "running" in k or "num_batches" in k})
    for k in runs[0]:
        torch.testing.assert_close(runs[1][k], runs[0][k], rtol=0, atol=0)


def test_segformer_hooked_fuse_conv_still_runs_its_hooks():
    """Codex review: the per-level path reads the fuse weight directly, so a hooked conv
    (spectral norm, observers) must keep the concat path and run its hooks."""
    torch.manual_seed(0)
    m = create_model(backbone="resnet18", pretrained=False, in_channels=3, num_classes=2,
                     decoder="segformer").eval()
    calls = []
    m.decoder.fuse[0].register_forward_pre_hook(lambda mod, a: calls.append(1) and None)
    feats = m.backbone(torch.randn(1, 3, 64, 64))
    torch.testing.assert_close(m.decoder(feats), _segformer_reference(m.decoder, feats))
    assert calls


def test_grad_checkpointing_is_all_or_nothing(monkeypatch):
    """Codex review: if one encoder cannot checkpoint, none may be changed."""
    from manyseg.backbones.timm_backbone import TimmBackbone

    m = create_model(pretrained=False, num_classes=2, decoder="unet",
                     inputs={"a": {"in_channels": 3, "backbone": "resnet18"},
                             "b": {"in_channels": 2, "backbone": "resnet18"}})
    encs = [mod for mod in m.modules() if isinstance(mod, TimmBackbone)]
    real = TimmBackbone._ckpt_target

    def target(self):
        if self is encs[1]:
            raise NotImplementedError("unsupported")
        return real(self)

    monkeypatch.setattr(TimmBackbone, "_ckpt_target", target)
    with pytest.raises(NotImplementedError):
        m.set_grad_checkpointing()
    assert all(mod.inplace for mod in encs[0].modules() if isinstance(mod, nn.ReLU))
    assert not any(mod._forward_pre_hooks for mod in m.modules())
