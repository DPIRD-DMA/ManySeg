"""timm models as multi-scale feature extractors with a uniform NCHW contract."""

from __future__ import annotations

import difflib
import inspect
import math
from collections.abc import Sequence

import timm
import torch
from torch import Tensor, nn

from ..adapters import CHANNEL_ADAPTERS, ProjectionAdapter, adapt_conv_weight
from ..features import FeatureSpec
from .necks import ViTPyramidNeck, default_pyramid_channels

PYRAMID_STRIDES = (4, 8, 16, 32)


def _timm_create(name: str, pretrained: bool = False, **kwargs) -> nn.Module:
    # Indirection point so tests can substitute weights without network access.
    return timm.create_model(name, pretrained=pretrained, **kwargs)


def _check_name(name: str) -> None:
    arch = name.split(".", 1)[0]
    if arch in timm.list_models():
        return
    close = difflib.get_close_matches(arch, timm.list_models(), n=5)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise ValueError(f"unknown timm backbone {name!r}{hint}")


def _is_int(v) -> bool:
    import numbers

    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def introspect(name: str) -> dict:
    """Cheap architecture introspection: build on the meta device (no memory, no weights) and
    read the constructor signature and the full feature_info."""
    try:
        with torch.device("meta"):
            m = timm.create_model(name.split(".", 1)[0], pretrained=False)
    except Exception:  # noqa: BLE001 - fall back to a real (random-init) build
        m = timm.create_model(name.split(".", 1)[0], pretrained=False)
    params = inspect.signature(type(m).__init__).parameters
    info = list(getattr(m, "feature_info", []) or [])
    reductions = [f["reduction"] for f in info]
    return {
        "init_params": set(params),
        "feature_info": info,
        "plain": len(info) > 1 and len(set(reductions)) == 1,
        "embed_dim": getattr(m, "embed_dim", None) or getattr(m, "num_features", None),
    }


_FIXED_SIZE_CACHE: dict[str, tuple[int, int] | None] = {}


def _probe_fixed_size(name: str, kwargs: dict, divisor: int, run, in_channels: int,
                      adapter: str) -> tuple[int, int] | None:
    # The meta probe depends on the architecture and its kwargs; the real fallback also on
    # the (adapted) input channels, so those are part of the key.
    key = repr((name.split(".", 1)[0], sorted(kwargs.items(), key=str), divisor, in_channels,
                adapter))
    if key not in _FIXED_SIZE_CACHE:
        found = _probe_fixed_size_uncached(name, kwargs, divisor)
        if found == "inconclusive":  # meta-device ops missing (older torch): probe for real
            found = _probe_real(run, in_channels, divisor)
        _FIXED_SIZE_CACHE[key] = found
    return _FIXED_SIZE_CACHE[key]


def _probe_sizes(cfg: dict, divisor: int):
    """(channels, native size, other sizes): fixed means only the native size runs. Two
    other sizes, an awkward one and native + one stride, so an any-size model with an extra
    divisibility rule is not mistaken for a fixed-size one."""
    c, h, w = (tuple(cfg.get("input_size") or (3, 224, 224)) + (224, 224))[:3]
    native = (int(h), int(w))
    odd = tuple(-(-n // divisor) * divisor for n in (97, 131))
    others = [s for s in (odd, (native[0] + divisor, native[1] + divisor)) if s != native]
    return c, native, others


def _fixed_from(try_size, native, others):
    """``try_size(hw) -> bool``. Fixed iff native runs and every other size fails; None if
    native fails too (nothing can be concluded)."""
    if not try_size(native):
        return "inconclusive"
    return native if others and not any(try_size(s) for s in others) else None


def _probe_real(run, in_channels: int, divisor: int) -> tuple[int, int] | None:
    """The same test on the built CPU backbone (through its input adapter), only reached when
    the meta probe cannot run. Skipped for very large models (two full CPU forwards)."""
    model = run.__self__ if hasattr(run, "__self__") else None
    params = sum(p.numel() for p in model.parameters()) if model is not None else 0
    if params > 150_000_000:
        return None
    cfg = getattr(getattr(model, "model", None), "pretrained_cfg", {}) or {}
    _, native, others = _probe_sizes(cfg, divisor)
    was = model.training if model is not None else False
    if model is not None:
        model.eval()
    dev = next(model.parameters()).device if model is not None else torch.device("cpu")

    def try_size(hw):
        try:
            with torch.no_grad():
                run(torch.zeros(1, in_channels, *hw, device=dev))
            return True
        except Exception:  # noqa: BLE001
            return False

    try:
        found = _fixed_from(try_size, native, others)
        return None if found == "inconclusive" else found
    finally:
        if model is not None:
            model.train(was)


def _probe_fixed_size_uncached(name: str, kwargs: dict, divisor: int):
    """Native (H, W) if the architecture only runs at its native size, else None.

    Some timm models (MaxViT, CoAtNet, LeViT, EfficientFormerV2, Hiera, ...) bake the input size
    into window partitions or positional tables and fail deep inside timm at other sizes. A
    meta-device forward (shapes only, no memory or compute) at the native size and at two other
    sizes tells them apart. If even the native probe fails (an op without meta support, e.g. on
    older torch), "inconclusive" sends the caller to ``_probe_real``."""
    try:
        with torch.device("meta"):
            m = timm.create_model(name.split(".", 1)[0], pretrained=False, **kwargs)
    except Exception:  # noqa: BLE001
        return "inconclusive"
    c, native, others = _probe_sizes(getattr(m, "pretrained_cfg", {}) or {}, divisor)

    def try_size(hw):
        try:
            with torch.no_grad():
                m(torch.empty(1, c, *hw, device="meta"))
            return True
        except Exception:  # noqa: BLE001
            return False

    return _fixed_from(try_size, native, others)


def _size_kwargs(init_params: set[str]) -> dict:
    """Enable timm's own dynamic-size support where the architecture offers it."""
    kw = {}
    if "dynamic_img_size" in init_params:
        kw["dynamic_img_size"] = True
    if "dynamic_img_pad" in init_params:
        kw["dynamic_img_pad"] = True
    if "strict_img_size" in init_params:
        kw["strict_img_size"] = False
    return kw


def _norm(name: str) -> str:
    return name.replace(".", "_")


def find_stem_convs(model: nn.Module, first_conv=None,
                    channels: int = 3) -> list[tuple[str, nn.Conv2d]]:
    """Locate the convolution(s) that consume the image.

    ``pretrained_cfg['first_conv']`` may be a name or a list of names (multi-branch stems such
    as MobileOne); feature wrappers may rename ``a.b`` to ``a_b``. Without it, the first
    ``Conv2d`` with ``channels`` (3, or a multispectral checkpoint's band count) input channels
    in module order is used.
    """
    candidates = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, nn.Conv2d) and m.in_channels == channels]
    if not candidates:
        raise ValueError(f"could not find a {channels}-channel stem convolution to adapt")
    wanted = [first_conv] if isinstance(first_conv, str) else list(first_conv or [])
    found = []
    for want in wanted:
        for n, m in candidates:
            if _norm(n) == _norm(want) or _norm(n).endswith(_norm(want)):
                found.append((n, m))
                break
    return found or [candidates[0]]


def find_stem_conv(model: nn.Module, first_conv=None) -> tuple[str, nn.Conv2d]:
    return find_stem_convs(model, first_conv)[0]


class TimmBackbone(nn.Module):
    """A timm model wrapped to return ``list[Tensor]`` (NCHW) with static ``feature_specs``
    (a plain ViT's neck levels that the model never reads come out as ``None``).

    Args:
        name: timm model name, optionally with a pretrained tag (``"resnet50.a1_in1k"``).
        pretrained: Load timm pretrained weights (``True``), or a remote-sensing weights tag from
            :mod:`manyseg.weights` (e.g. ``"ssl4eo_s2_moco"``; ``in_channels`` must equal its
            band count).
        in_channels: Number of input channels.
        channel_adapter: How to adapt the RGB stem when ``in_channels != 3``; see
            :func:`manyseg.adapters.adapt_conv_weight` and :class:`ProjectionAdapter`.
        out_indices: Feature levels to return (timm ``out_indices``). Default: all.
        rgb_indices: Input channels carrying R, G, B for ``rgb_plus_*`` / ``projection``.
        **timm_kwargs: Forwarded to ``timm.create_model`` (e.g. ``drop_path_rate``).
    """

    # set by manyseg.load: the pretrained weights this backbone started from (see
    # manyseg.weights.attribution), as the rebuilt backbone was not given them
    restored_attribution: dict[str, str] | None = None

    def __init__(
        self,
        name: str,
        pretrained: bool | str = True,
        in_channels: int = 3,
        channel_adapter: str = "repeat",
        out_indices: Sequence[int] | None = None,
        rgb_indices: Sequence[int] | None = None,
        **timm_kwargs,
    ) -> None:
        super().__init__()
        _check_name(name)
        if channel_adapter not in CHANNEL_ADAPTERS:
            raise ValueError(
                f"unknown channel_adapter {channel_adapter!r}; choose from {CHANNEL_ADAPTERS}"
            )
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if rgb_indices is not None and channel_adapter in ("repeat", "mean"):
            raise ValueError(f"rgb_indices applies to the rgb_plus_* and projection adapters, "
                             f"not {channel_adapter!r}")
        arch = introspect(name)
        self.is_plain = arch["plain"]
        # deepest stride the network computes; some timm feature models (e.g. MobileNetV3) run
        # every block whatever out_indices returns (export's minimum size depends on it)
        self.network_stride = max((int(f["reduction"]) for f in arch["feature_info"]),
                                  default=1)
        kwargs = dict(features_only=True, **_size_kwargs(arch["init_params"]))
        kwargs.update(timm_kwargs)
        self.fixed_size: tuple[int, int] | None = None
        # A user img_size fixes the size unless timm is told to accept others: no dynamic
        # size support, or the dynamic/strict flags (ours or the user's) leave it strict.
        dynamic = bool(kwargs.get("dynamic_img_size")) or kwargs.get("strict_img_size") is False
        if "img_size" in timm_kwargs and not dynamic:
            size = timm_kwargs["img_size"]
            self.fixed_size = (size, size) if isinstance(size, int) else tuple(size)
        if self.is_plain and out_indices is None:
            n = len(arch["feature_info"])
            out_indices = tuple(n * (i + 1) // 4 - 1 for i in range(4))
        if out_indices is not None:
            n = len(arch["feature_info"])
            idx = list(out_indices)
            if not idx or any(not _is_int(i) or not -n <= i < n for i in idx) or \
                    len({i % n for i in idx}) != len(idx):
                kind = "blocks" if self.is_plain else "feature levels"
                raise ValueError(f"{name}: out_indices must be distinct integer indices into "
                                 f"its {n} {kind} (0..{n - 1}, or negative from the end), got "
                                 f"{list(out_indices)}")
            out_indices = tuple(int(i) for i in idx)
            kwargs["out_indices"] = out_indices
        self.block_indices = tuple(out_indices) if self.is_plain else None
        self.weights: str | None = None
        if isinstance(pretrained, str):  # remote-sensing weights (manyseg.weights)
            from ..weights import get_weights, load_into

            w = get_weights(name, pretrained)
            if in_channels != w.in_channels:
                raise ValueError(f"{w.key} expects {w.in_channels} bands "
                                 f"({', '.join(w.bands)}), in that order; got "
                                 f"in_channels={in_channels}")
            self.model = _timm_create(name, pretrained=False,
                                      **{**kwargs, "in_chans": w.in_channels})
            load_into(self.model, w)
            self.weights = w.key
        else:
            self.model = _timm_create(name, pretrained=pretrained, **kwargs)
        self._drop_unreturned_blocks()
        self.name = name
        self.pretrained = bool(pretrained)  # recorded for describe(); weights are already set
        self.in_channels = in_channels
        self.channel_adapter = channel_adapter
        self.rgb_indices = None if rgb_indices is None else tuple(int(i) for i in rgb_indices)

        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        native = in_channels if self.weights is not None else 3
        stems = find_stem_convs(self.model, cfg.get("first_conv"), native)
        self.stem_conv_names = [n for n, _ in stems]
        self._stem_name = self.stem_conv_names[0]
        self.input_adapter: nn.Module = nn.Identity()
        if rgb_indices is not None and in_channels == stems[0][1].in_channels:
            raise ValueError(f"rgb_indices has no effect: a {in_channels}-channel input needs no "
                             "stem adaptation")
        if in_channels != stems[0][1].in_channels:
            if channel_adapter == "projection":
                self.input_adapter = ProjectionAdapter(in_channels, stems[0][1].in_channels,
                                                       rgb_indices=rgb_indices)
            else:
                for _, stem in stems:
                    if stem.groups != 1:
                        raise ValueError("cannot adapt a grouped stem convolution")
                    new_w = adapt_conv_weight(stem.weight, in_channels, channel_adapter,
                                              rgb_indices=rgb_indices)
                    stem.weight = nn.Parameter(new_w.contiguous())
                    stem.in_channels = in_channels

        info = self.model.feature_info
        self.nhwc = str(getattr(self.model, "output_fmt", "NCHW")).upper().endswith("NHWC")
        specs = tuple(
            FeatureSpec(c, r, n)
            for c, r, n in zip(info.channels(), info.reduction(), info.module_name())
        )
        self.neck: ViTPyramidNeck | None = None
        if self.is_plain:
            k = len(specs)
            strides = PYRAMID_STRIDES[len(PYRAMID_STRIDES) - k:] if k <= 4 else None
            if strides is None:
                raise ValueError(f"plain ViT pyramid supports up to 4 blocks, got {k}")
            channels = default_pyramid_channels(specs[0].channels, k)
            self.neck = ViTPyramidNeck(specs[0].channels, strides, channels, specs[0].stride)
            specs = self.neck.feature_specs
        self.feature_specs: tuple[FeatureSpec, ...] = specs
        if self.fixed_size is None and not self.is_plain and "img_size" not in timm_kwargs:
            # DenseModel pads smaller inputs up to a fixed size and rejects larger ones clearly.
            self.fixed_size = _probe_fixed_size(name, kwargs, self.input_divisor,
                                                self._run_adapted, in_channels,
                                                channel_adapter)
        self.mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    def _drop_unreturned_blocks(self) -> None:
        """timm's EfficientNet / MobileNetV3 feature models run every block whatever
        ``out_indices`` returns; drop the blocks after the deepest returned level (their
        parameters would never get a gradient, which DDP rejects by default)."""
        from timm.models.efficientnet import EfficientNetFeatures
        from timm.models.mobilenetv3 import MobileNetV3Features

        if not isinstance(self.model, (EfficientNetFeatures, MobileNetV3Features)):
            return  # only the classes whose block / stage layout is known
        stages = getattr(self.model, "_stage_out_idx", None)
        blocks = getattr(self.model, "blocks", None)
        if not stages or not isinstance(blocks, nn.Sequential) or \
                getattr(self.model, "feature_hooks", None) is not None:
            return
        last = max(stages)  # block i feeds stage i + 1; stage 0 is the stem
        if last < len(blocks):
            self.model.blocks = blocks[:last]
            self.network_stride = max(self.model.feature_info.reduction())

    def _restrict_levels(self, levels) -> None:
        """Only ``levels`` are read by the model: a plain ViT's neck builds no projection for
        the others (they come out as ``None``)."""
        if self.neck is not None:
            self.neck._restrict(levels)

    @property
    def input_divisor(self) -> int:
        """Inputs whose H and W are multiples of this give exactly aligned feature maps."""
        return math.lcm(*(s.stride for s in self.feature_specs))

    def _run_adapted(self, x: Tensor):
        return self.model(self.input_adapter(x))

    def stem_conv(self) -> nn.Conv2d:
        return self.model.get_submodule(self._stem_name)

    def _ckpt_target(self) -> nn.Module:
        target = self.model if hasattr(self.model, "set_grad_checkpointing") else \
            getattr(self.model, "model", None)  # FeatureGetterNet wraps the timm model
        if not hasattr(target, "set_grad_checkpointing"):
            raise NotImplementedError(
                f"timm backbone {self.name!r} does not support gradient checkpointing")
        return target

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Recompute the encoder's activations in the backward pass instead of storing them
        (timm's gradient checkpointing): less training memory for more compute (E026: -22%
        peak memory for +5% step time on a ResNet-34 U-Net, -61% for +18% on ConvNeXt-T).

        Values are unchanged. In-place activations are switched off while it is on (they would
        overwrite tensors the checkpoint saved), and BatchNorm running statistics are restored
        after the recomputation, so they are updated once per step, as without checkpointing.
        """
        self._ckpt_target().set_grad_checkpointing(enable)
        for h in getattr(self, "_ckpt_hooks", ()):
            h.remove()
        self._ckpt_hooks = []
        saved = getattr(self, "_ckpt_inplace", {})
        if enable:
            for m in self.model.modules():
                if isinstance(getattr(m, "inplace", None), bool) and m not in saved:
                    saved[m], m.inplace = m.inplace, False
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
                    self._ckpt_hooks.append(m.register_forward_pre_hook(_bn_stash))
        else:
            for m, inplace in saved.items():
                m.inplace = inplace
            saved = {}
        self._ckpt_inplace = saved

    def forward(self, x: Tensor) -> list[Tensor]:
        h, w = x.shape[-2], x.shape[-1]
        if self.fixed_size is not None and (h, w) != self.fixed_size:
            raise ValueError(
                f"{self.name} was built for a fixed input size {self.fixed_size}; got {(h, w)}"
            )
        feats = self.model(self.input_adapter(x))
        if isinstance(feats, dict):
            feats = list(feats.values())
        feats = list(feats)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        if self.neck is not None:
            feats = self.neck(feats, h, w)
        return feats


def _in_backward() -> bool:
    return torch._C._current_graph_task_id() != -1


def _bn_stash(m: nn.Module, args) -> None:
    """A BatchNorm forward inside the backward pass is a checkpoint's recomputation: put its
    running statistics back when the backward pass ends. (Not in a forward hook: checkpointing
    stops a recomputation early, inside the forward, once it has every tensor it needs.)"""
    if not (m.training and _in_backward()) or m.__dict__.get("_ckpt_pending"):
        return  # the first recomputation in a backward pass saves (shared encoders run twice)
    bufs = (m.running_mean, m.running_var, m.num_batches_tracked)
    saved = [t.clone() for t in bufs]
    m.__dict__["_ckpt_pending"] = True

    def restore():
        with torch.no_grad():
            for t, s in zip(bufs, saved):
                t.copy_(s)
        m.__dict__["_ckpt_pending"] = False

    torch.autograd.Variable._execution_engine.queue_callback(restore)


def create_backbone(name: str, pretrained: bool | str = True, **kwargs) -> TimmBackbone:
    return TimmBackbone(name, pretrained=pretrained, **kwargs)


__all__ = ["TimmBackbone", "create_backbone", "find_stem_conv"]
