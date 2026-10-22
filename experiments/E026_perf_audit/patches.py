"""E026: candidate performance changes, applied at runtime (the library source is not edited).

Each patch is a function that monkeypatches manyseg; ``apply(names)`` runs them before the
model is built. Every patch is meant to give the same values as the code it replaces (at most
float rounding); ``check.py`` verifies that.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import manyseg.decoders.deeplab as deeplab_mod
import manyseg.decoders.fastai_unet as fastai_mod
import manyseg.decoders.fpn as fpn_mod
import manyseg.decoders.segformer as segformer_mod
import manyseg.decoders.unet as unet_mod
import manyseg.decoders.unetplusplus as unetpp_mod
import manyseg.decoders.upernet as upernet_mod
import manyseg.fusion.ops as fusion_ops_mod
import manyseg.layers as layers_mod
import manyseg.models.dense as dense_mod
import manyseg.temporal as temporal_mod

PATCHES = {}


def patch(fn):
    PATCHES[fn.__name__] = fn
    return fn


def apply(names) -> list[str]:
    names = [n for n in (names or []) if n]
    for n in names:
        PATCHES[n]()
    return names


def _ac_dtype(t: torch.Tensor):
    dev = t.device.type
    return torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else None


@patch
def tail_dtype():
    """fastai_unet / unet input_cross: cast the fp32 input image to the autocast dtype before
    it is concatenated, so the full-resolution tail runs in bf16 (it runs in fp32 now)."""
    for cls in (unet_mod.UNetDecoder, fastai_mod.FastaiUNetDecoder):
        orig = cls.forward

        def forward(self, features, _orig=orig):
            if getattr(self, "wants_input", False):
                dt = _ac_dtype(features[0])
                if dt is not None:
                    features = [features[0].to(dt), *features[1:]]
            return _orig(self, features)

        cls.forward = forward


@patch
def fastai_layout():
    """fastai_unet: keep channels_last through pixel_shuffle (CUDA's drops it), and in-place
    ReLUs where the input is not saved for backward."""

    def shuf_forward(self, x):
        y = self.shuffle(F.relu(self.conv(x), inplace=True))
        if x.is_contiguous(memory_format=torch.channels_last) and not \
                y.is_contiguous(memory_format=torch.channels_last):
            y = y.contiguous(memory_format=torch.channels_last)
        return y

    def block_forward(self, x, skip):
        up = fastai_mod.fit_nearest(self.shuf(x), skip.shape[-2:], self.exact)
        return self.conv2(self.conv1(F.relu(torch.cat([up, self.bn(skip)], dim=1),
                                            inplace=True)))

    def res_forward(self, x):
        return F.relu(self.convs(x) + x, inplace=True)

    fastai_mod.PixelShuffleICNR.forward = shuf_forward
    fastai_mod._Block.forward = block_forward
    fastai_mod._ResBlock.forward = res_forward


@patch
def cross_exact():
    """unet input_cross: the image the decoder gets is the padded input, so the last step is an
    exact 2x too; drop the exclusion (compile then uses the cheap upsample, as E025)."""

    def _set_input_divisor(self, divisor):
        exact = divisor % self._deepest_stride == 0
        for block in self.blocks:
            block.exact_2x = exact

    unet_mod.UNetDecoder._set_input_divisor = _set_input_divisor


@patch
def segformer_fuse():
    """SegFormer: the fuse is a bias-free 1x1 conv after bilinear upsampling (linear, weights
    sum to 1), so apply each level's weight slice at its own resolution, then upsample and
    sum. Same parameters; no 4C-channel concat at stride 4."""

    def forward(self, features):
        size = features[self.levels[0]].shape[-2:]
        conv = self.fuse[0]
        assert conv.bias is None and conv.kernel_size == (1, 1)
        c = conv.weight.shape[1] // len(self.levels)
        acc = None
        for j, (i, lin) in enumerate(zip(self.levels, self.linear)):
            y = lin(features[i])
            z = F.conv2d(y, conv.weight[:, j * c:(j + 1) * c])
            if z.shape[-2:] != size:
                z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
            z = z.to(torch.promote_types(z.dtype, torch.float32))  # sum in >= fp32, round once
            acc = z if acc is None else acc + z
        return self.dropout(self.fuse[2](self.fuse[1](acc.to(y.dtype))))

    segformer_mod.SegFormerDecoder.forward = forward


def _interp_keep_dtype(x, *args, **kwargs):
    return F.interpolate(x, *args, **kwargs).to(x.dtype)


@patch
def interp_dtype():
    """UPerNet / DeepLab / FPN / SegFormer: autocast runs bilinear interpolate in fp32; cast the
    result back to the input dtype (the next conv casts to bf16 anyway)."""
    shim = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    shim.interpolate = _interp_keep_dtype
    for mod in (upernet_mod, deeplab_mod, fpn_mod, segformer_mod):
        mod.F = shim


class _GroupNormKeepDtype(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x).to(x.dtype)


@patch
def gn_dtype():
    """GroupNorm decoders: autocast runs group_norm in fp32 (and ReLU after it); cast back."""
    orig = layers_mod.make_norm

    def make_norm(norm, channels):
        if norm == "group":
            return _GroupNormKeepDtype(min(32, channels), channels)
        return orig(norm, channels)

    layers_mod.make_norm = make_norm


def _split_conv(cna, parts):
    """ConvNormAct over cat(parts) as a sum of per-part convs with weight slices."""
    conv = cna[0]
    y, start, dt = None, 0, None
    for p in parts:
        n = p.shape[1]
        z = F.conv2d(p, conv.weight[:, start:start + n], None, conv.stride, conv.padding)
        dt = z.dtype
        z = z.to(torch.promote_types(z.dtype, torch.float32))  # sum in >= fp32, round once
        y = z if y is None else y + z
        start += n
    if conv.bias is not None:
        y = y + conv.bias.view(1, -1, 1, 1).to(y.dtype)
    return cna[2](cna[1](y.to(dt)))


@patch
def split_conv():
    """U-Net / UNet++: conv(cat(up, skips)) as a sum of convs over weight slices, so the
    concat buffer is not stored for backward."""

    def block_forward(self, x, skip, size=None):
        exact = self.exact_2x or (skip is None and size is None)
        if size is None:
            size = skip.shape[-2:] if skip is not None else (2 * x.shape[-2], 2 * x.shape[-1])
        x = unet_mod.upsample2x(x, size, self.upsample, exact)
        if skip is None:
            return self.conv2(self.conv1(x))
        return self.conv2(_split_conv(self.conv1, [x, skip]))

    def node_forward(self, below, same):
        ref = same[0]
        up = unet_mod.upsample2x(below, ref.shape[-2:], self.upsample, self.exact_2x)
        return self.block.conv2(_split_conv(self.block.conv1, [up, *same]))

    unet_mod.UNetBlock.forward = block_forward
    unetpp_mod._Node.forward = node_forward


@patch
def pad_fast():
    """Skip the zero-pad copy when the input is already a multiple of the divisor (eager)."""
    orig = dense_mod.pad_to_multiple

    def pad_to_multiple(x, divisor, value=0.0):
        if not torch.compiler.is_compiling() and divisor > 1 and \
                x.shape[-2] % divisor == 0 and x.shape[-1] % divisor == 0:
            return x
        return orig(x, divisor, value)

    dense_mod.pad_to_multiple = pad_to_multiple


@patch
def mean_agg():
    """Temporal mean: select and sum with an fp32 accumulator instead of upcasting first."""

    def forward(self, f, mask, dates=None):
        m = mask[:, :, None, None, None]
        s = torch.where(m, f, torch.zeros((), device=f.device, dtype=f.dtype)).sum(
            1, dtype=torch.float32)
        return (s / mask.sum(1).clamp(min=1)[:, None, None, None]).to(f.dtype)

    temporal_mod.MeanAggregation.forward = forward


@patch
def xattn_contig():
    """Cross-attention fusion: give SDPA tokens with a unit last-dim stride (flash /
    memory-efficient kernels need it)."""
    orig = fusion_ops_mod.CrossAttentionFusion._tokens

    def _tokens(self, x):
        return orig(self, x).contiguous()

    fusion_ops_mod.CrossAttentionFusion._tokens = _tokens


# --- memory-for-time opt-ins (not value-identical for BN running statistics, see check.py) ---

def enable_encoder_checkpointing(model: nn.Module) -> int:
    """timm's own gradient checkpointing on every timm encoder in the model."""
    from manyseg.backbones.timm_backbone import TimmBackbone

    if hasattr(model, "set_grad_checkpointing"):  # the library method (after E026)
        model.set_grad_checkpointing()
        return sum(isinstance(m, TimmBackbone) for m in model.modules())
    n = 0
    for m in model.modules():
        if isinstance(m, TimmBackbone):
            # timm's in-place ReLUs modify tensors that checkpointing saved (ResNet fails with
            # "modified by an inplace operation"): make them out of place
            for a in m.modules():
                if isinstance(a, nn.ReLU):
                    a.inplace = False
            tm = m.model
            target = tm if hasattr(tm, "set_grad_checkpointing") else getattr(tm, "model", None)
            target.set_grad_checkpointing(True)
            n += 1
    return n


def enable_decoder_checkpointing(model: nn.Module, finest: int = 2) -> int:
    """Checkpoint the ``finest`` (highest-resolution) U-Net blocks."""
    dec = model.decoder
    blocks = list(getattr(dec, "blocks", []))[-finest:]
    for b in blocks:
        orig = b.forward

        def fwd(x, skip, size=None, _orig=orig):
            if not torch.is_grad_enabled():
                return _orig(x, skip, size)
            return checkpoint(_orig, x, skip, size, use_reentrant=False)

        b.forward = fwd
    return len(blocks)


# --- round 2 (changes numerics beyond rounding: the residual stream rounds to bf16) ---

@patch
def ln_dtype():
    """timm LayerNorms (and torch's): output in the input's dtype. CUDA autocast runs
    layer_norm in fp32; the stem / patch-embed norm then makes the encoder's whole residual
    stream fp32 (ConvNeXt, EdgeNeXt, PVTv2, ViT, Swin)."""
    import timm.layers.norm as tn

    for cls in (nn.LayerNorm, tn.LayerNorm, tn.LayerNorm2d):
        orig = cls.forward

        def forward(self, x, _orig=orig):
            return _orig(self, x).to(x.dtype)

        cls.forward = forward
