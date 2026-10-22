"""E027: round-2 candidates (from E026's leads), applied at runtime; the library is not edited.

Each patch should give the values of the code it replaces (at most float rounding).
"""

from __future__ import annotations

import types

import manyseg.backbones.necks as necks_mod
import manyseg.decoders.unet as unet_mod
import manyseg.fusion.encoders as fenc_mod
import manyseg.fusion.late as late_mod
import manyseg_bench.evaluation.metrics as metrics_mod
import manyseg_bench.evaluation.objects as objects_mod
import manyseg_bench.tasks as tasks_mod
import torch
import torch.nn.functional as F

PATCHES = {}


def patch(fn):
    PATCHES[fn.__name__] = fn
    return fn


def apply(names) -> list[str]:
    names = [n for n in (names or []) if n]
    for n in names:
        PATCHES[n]()
    return names


@patch
def eval_nosync():
    """Semantic evaluation without host syncs: loss / boundary-F1 sums stay on the device,
    and the confusion matrix is a fixed-size index_add (boolean masking and bincount both
    wait for the GPU)."""

    def cm_update(self, pred, target):
        pred, target = pred.reshape(-1), target.reshape(-1)
        keep = (target >= 0) & (target < self.k)
        if self.ignore_index is not None:
            keep &= target != self.ignore_index
        dump = self.k * self.k  # ignored pixels go to an extra bin
        idx = torch.where(keep, target.long() * self.k + pred.long(), dump).to(self.mat.device)
        if not hasattr(self, "_flat"):
            self._flat = torch.zeros(dump + 1, dtype=torch.int64, device=self.mat.device)
        self._flat.index_add_(0, idx, torch.ones_like(idx))
        self.mat = self._flat[:dump].view(self.k, self.k)

    metrics_mod.ConfusionMatrix.update = torch.no_grad()(cm_update)

    def to_dev(v, like):
        return v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float64,
                                                                  device=like.device)

    def update(self, out, y):
        f = tasks_mod.head_factor(out, y)
        if self.factor is None:
            self.factor = f
        if self.t.native_metrics:
            if self.native is None:
                self.native = tasks_mod.ConfusionMatrix(self.t.num_classes, self.t.ignore_index,
                                                        device=y.device)
            ys, ps = self.t.head_targets(out, y), out.float().argmax(1)
            self.native.update(ps, ys)
            self.native_bf1 = to_dev(self.native_bf1, y) + tasks_mod.boundary_f1(
                ps, ys, ignore_index=self.t.ignore_index).sum().double()
        logits = tasks_mod.to_label_grid(out, y).float()
        if self.cm is None:
            self.cm = tasks_mod.ConfusionMatrix(self.t.num_classes, self.t.ignore_index,
                                                device=y.device)
        ignore = self.t.ignore_index if self.t.ignore_index is not None else -100
        self.loss = to_dev(self.loss, y) + F.cross_entropy(
            logits, y, ignore_index=ignore, reduction="sum").double()
        pred = logits.argmax(1)
        self.cm.update(pred, y)
        self.bf1 = to_dev(self.bf1, y) + tasks_mod.boundary_f1(
            pred, y, ignore_index=self.t.ignore_index).sum().double()
        self.n += y.shape[0]
        self.px += y.numel()

    orig_compute = tasks_mod._SemanticEval.compute

    def compute(self):
        for k in ("bf1", "loss", "native_bf1"):
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                setattr(self, k, float(v))
        return orig_compute(self)

    tasks_mod._SemanticEval.update = update
    tasks_mod._SemanticEval.compute = compute


def _resize_keep(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False).to(x.dtype)


def _shim(**overrides):
    ns = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _interp_keep_dtype(x, *args, **kwargs):
    return F.interpolate(x, *args, **kwargs).to(x.dtype)


@patch
def fusion_dtype():
    """Late fusion and same_extent intermediate fusion: bilinear resizes keep the input dtype
    (CUDA autocast returns float32, which a concat fusion then spreads to the whole buffer)."""
    shim = _shim(interpolate=_interp_keep_dtype)
    late_mod.F = shim
    fenc_mod.F = shim


@patch
def neck_dtype():
    """Plain-ViT pyramid neck: the resized projection keeps its dtype."""
    necks_mod.F = _shim(interpolate=_interp_keep_dtype)


@patch
def unet_bilinear_dtype():
    """U-Net ``upsample="bilinear"`` / ``"aligned"``: keep the dtype before the skip concat."""
    orig = unet_mod.upsample2x

    def upsample2x(x, size, mode, exact_2x=False):
        return orig(x, size, mode, exact_2x).to(x.dtype)

    unet_mod.upsample2x = upsample2x


@patch
def objects_cpu():
    """ObjectMetrics: one device-to-host copy per batch instead of two per image."""
    orig = objects_mod.ObjectMetrics.update

    def update(self, pred, gt):
        return orig(self, pred.cpu(), gt.cpu())

    objects_mod.ObjectMetrics.update = update
