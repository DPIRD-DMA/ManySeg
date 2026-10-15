"""Quantisation entry points (deployment is kept separate from model construction).

Two routes, both through torchao (``pip install 'manyseg[quant]'``):

* ``quantize_pt2e``: static int8 for CPU deployment. ``torch.export`` (with
  ``export_model``'s dynamic H/W) → ``prepare_pt2e`` → calibration → ``convert_pt2e`` →
  ``torch.export`` again, so the result is an ``ExportedProgram`` like ``export_model``'s.
  Weights and activations are int8; ``torch.compile`` lowers the graph to oneDNN int8 kernels.
* ``quantize_weights``: int8 weights (optionally int8 dynamic activations for Linear layers)
  on the eager ``nn.Module``, for CPU or GPU, no calibration and no export.

``model_size`` and ``compare_outputs`` report what the conversion saved and what it cost.

This is a module of its own rather than part of ``manyseg.export`` because it carries an
optional dependency and because ``quantize_weights`` does not export at all. The PT2E path
reuses ``manyseg.export.dynamic_shapes`` rather than duplicating it.
"""

from __future__ import annotations

import copy
import io
import warnings
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn

from .export import dynamic_shapes

WEIGHT_MODES = ("int8_weight_only", "int8_dynamic")


def _require_torchao() -> None:
    try:
        import torchao  # noqa: F401
    except ImportError as e:
        raise ImportError("quantisation needs torchao: pip install 'manyseg[quant]'") from e


# --- PT2E: static int8, exported -----------------------------------------------------------

def _has_vnni() -> bool:
    """Whether this CPU has VNNI; unknown counts as no (``reduce_range`` is then the safe
    choice)."""
    probe = getattr(torch.cpu, "_is_vnni_supported", None)
    try:
        return bool(probe()) if probe is not None else False
    except Exception:
        return False


def _x86_quantizer(reduce_range: bool | None = None):
    """torchao's x86 quantizer. ``reduce_range`` (7-bit activations) defaults to "this CPU has
    no VNNI": there, inductor's u8 x s8 kernels saturate their 16-bit intermediate sums, and
    in E008 the compiled int8 model lost ~6 mIoU on Oxford Pet (AMD Zen 3; the eager
    reference was fine). torchao's own docs give the same advice."""
    from torchao.quantization.pt2e.quantizer import x86_inductor_quantizer as xiq

    rr = (not _has_vnni()) if reduce_range is None else bool(reduce_range)
    cfg = xiq.get_default_x86_inductor_quantization_config(reduce_range=rr)
    return xiq.X86InductorQuantizer().set_global(cfg)


# timm's window-attention classes, by (module, class): their dynamic-size export hangs rather
# than fails (``architecture_review.md``). Matched exactly so that an unrelated module with a
# ``window_size`` attribute is not rejected.
_WINDOW_ATTENTION = {
    ("timm.models.swin_transformer", "WindowAttention"),
    ("timm.models.swin_transformer_v2", "WindowAttention"),
    ("timm.models.swin_transformer_v2_cr", "WindowMultiHeadAttention"),
    ("timm.models.davit", "WindowAttention"),
}


def _windowed(model: nn.Module) -> str | None:
    """Class name of the first timm window-attention module (Swin, Swin V2, DaViT), if any."""
    return next((type(m).__name__ for m in model.modules()
                 if (type(m).__module__, type(m).__name__) in _WINDOW_ATTENTION), None)


def _shapes(model: nn.Module, example, dynamic: bool, max_size: int, dynamic_batch: bool):
    """``export_model``'s dynamic-shape spec, or None (fixed size). Computed once from the float
    model: the quantised graph is re-exported with the same spec."""
    if not dynamic:
        return None
    if (block := _windowed(model)) is not None:
        raise ValueError(f"window-attention backbones ({block}) cannot be exported with dynamic "
                         "H/W; pass dynamic=False to quantise at the example's fixed size")
    return dynamic_shapes(model, example, max_size, dynamic_batch)


def _export(module: nn.Module, example, shapes) -> torch.export.ExportedProgram:
    """``torch.export`` as ``export_model`` does it, for the float model and the int8 graph."""
    try:
        return torch.export.export(module, (example,), dynamic_shapes=shapes,
                                   prefer_deferred_runtime_asserts_over_guards=True)
    except Exception as e:  # torch.export raises several types (UserError, RuntimeError, ...)
        if shapes is None:
            raise
        raise ValueError(
            "could not export this model with dynamic H/W; pass dynamic=False to quantise at "
            f"the example's fixed size. Export error: {type(e).__name__}: {e}") from e


def _drop_unused_attrs(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """``convert_pt2e`` adds int8 copies of the weights but leaves the float originals
    registered (unused); drop every parameter and buffer the graph no longer reads. An attribute
    is read if a ``get_attr`` or ``call_module`` target is it or a dotted prefix of it (e.g.
    ``body`` for ``body.weight``)."""
    targets = {n.target for n in gm.graph.nodes if n.op in ("get_attr", "call_module")}

    def used(name: str) -> bool:
        parts = name.split(".")
        return any(".".join(parts[:i]) in targets for i in range(1, len(parts) + 1))

    for name in [n for n, _ in [*gm.named_parameters(), *gm.named_buffers()] if not used(n)]:
        owner, _, leaf = name.rpartition(".")
        mod = gm.get_submodule(owner)
        mod._parameters.pop(leaf, None)
        mod._buffers.pop(leaf, None)
    gm.delete_all_unused_submodules()
    return gm


def quantize_pt2e(model: nn.Module, example, calibration: Iterable, *, quantizer=None,
                  dynamic: bool = True, max_size: int = 4096,
                  dynamic_batch: bool = False,
                  reduce_range: bool | None = None) -> torch.export.ExportedProgram:
    """Static int8 (weights and activations) by PT2E post-training quantisation.

    ``example`` is a Tensor or, for multi-input models, a ``dict[str, Tensor]``, exactly as the
    model takes it. ``calibration`` is an iterable of inputs of the same form (e.g. a few dozen
    representative batches); activation ranges are observed on it. ``quantizer`` defaults to
    torchao's ``X86InductorQuantizer`` (per-channel weights, per-tensor activations); pass any
    other PT2E quantizer (e.g. ``ArmInductorQuantizer``) to target another backend.
    ``reduce_range`` (default: on when *this* CPU lacks VNNI) uses 7-bit activations, which
    is needed for correct compiled int8 on such CPUs; set it for the CPU you deploy on.

    H/W stay dynamic as in ``export_model`` (and the batch too with ``dynamic_batch``).
    Window-attention backbones (Swin, DaViT) cannot export dynamically and need
    ``dynamic=False``: the result then only accepts the example's size.

    Returns an ``ExportedProgram``, like ``export_model``: ``torch.export.save`` it, or run
    ``ep.module()``. Eager calls run int8 as quantise/dequantise pairs in float, which shows the
    accuracy cost but is not faster; for speed, compile it with inductor freezing::

        torch._inductor.config.freezing = True
        fast = torch.compile(ep.module())
    """
    _require_torchao()
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    model.eval()
    shapes = _shapes(model, example, dynamic, max_size, dynamic_batch)
    gm = _export(model, example, shapes).module()
    prepared = prepare_pt2e(gm, quantizer if quantizer is not None
                            else _x86_quantizer(reduce_range))
    n, bs = 0, _batch_of(example)
    with torch.no_grad():
        for x in calibration:
            if not dynamic_batch and _batch_of(x) != bs:
                raise ValueError(f"calibration batch size {_batch_of(x)} differs from the "
                                 f"example's {bs}; the export fixes the batch dimension unless "
                                 "dynamic_batch=True")
            prepared(x)
            n += 1
    if n == 0:
        raise ValueError("calibration must yield at least one input")
    return _export(_drop_unused_attrs(convert_pt2e(prepared)), example, shapes)


def _batch_of(x) -> int:
    if isinstance(x, Mapping):
        x = next(v for v in x.values() if torch.is_tensor(v))
    return int(x.shape[0])


# --- torchao: weight-only / dynamic, eager -------------------------------------------------

def _is_linear(mod: nn.Module, fqn: str) -> bool:
    return isinstance(mod, nn.Linear)


def _is_dense_conv(mod: nn.Module, fqn: str) -> bool:
    """Conv2d with more than one input channel per group. Depthwise weights (C, 1, k, k) are
    tiny and torchao's int8 conv weights reject them, so they stay float."""
    return isinstance(mod, nn.Conv2d) and mod.weight.shape[1] > 1


def quantize_weights(model: nn.Module, mode: str = "int8_weight_only", *, conv: bool = True,
                     inplace: bool = False) -> nn.Module:
    """int8 weights through torchao, for eager or ``torch.compile`` inference on CPU or GPU.

    ``mode``:

    * ``"int8_weight_only"``: Linear and Conv2d weights stored int8 (symmetric, per output
      channel), activations stay float. About 4x smaller; on GPU torchao's int8 matmul makes
      Linear layers faster under ``torch.compile``.
    * ``"int8_dynamic"``: Linear layers also quantise their activations to int8 at run time.
      torchao has no dynamic-activation conv, so Conv2d weights are int8 weight-only.

    Conv2d weights are dequantised on the fly, so for CNNs this saves memory and disk, not
    time. ``conv=False`` leaves convs in float (Linear only). Depthwise convs always stay float.
    The model is copied first unless ``inplace``.
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode must be one of {WEIGHT_MODES}, got {mode!r}")
    _require_torchao()
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        IntxWeightOnlyConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerAxis

    if not inplace:
        model = copy.deepcopy(model)
    if conv:
        quantize_(model, IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
                  filter_fn=_is_dense_conv)
    linear = (Int8WeightOnlyConfig() if mode == "int8_weight_only"
              else Int8DynamicActivationInt8WeightConfig())
    quantize_(model, linear, filter_fn=_is_linear)
    return model


# --- reports -------------------------------------------------------------------------------

def model_size(model: nn.Module | torch.export.ExportedProgram) -> int:
    """Serialized size in bytes: ``torch.export.save`` for an ``ExportedProgram`` (graph and
    weights, as deployed), ``torch.save(model.state_dict())`` for a module (weights only)."""
    buf = io.BytesIO()
    if isinstance(model, torch.export.ExportedProgram):
        torch.export.save(model, buf)
    else:
        torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


class _Diff:
    """Running agreement statistics for one output."""

    def __init__(self):
        self.max, self.abs_sum, self.n, self.agree, self.count = 0.0, 0.0, 0, 0, 0
        self.kind: str | None = None  # "argmax" (float logits), "exact" (integer/bool) or None
        self.dtype_changed = False

    def add(self, ref: Tensor, out: Tensor) -> None:
        out = out.to(ref.device)
        self.dtype_changed |= ref.dtype != out.dtype
        d = (ref.double() - out.double()).abs()
        self.max = max(self.max, d.max().item())
        self.abs_sum += d.sum().item()
        self.n += d.numel()
        if not ref.is_floating_point():  # label maps, counts: compare exactly
            self.kind, agree = "exact", ref == out
        elif ref.ndim in (2, 4) and ref.shape[1] > 1:  # (B, C) or (B, C, H, W) logits
            self.kind, agree = "argmax", ref.argmax(1) == out.argmax(1)
        else:
            return
        self.agree += agree.sum().item()
        self.count += agree.numel()

    def report(self, prefix: str = "") -> dict[str, float]:
        r = {f"{prefix}max_abs_diff": self.max, f"{prefix}mean_abs_diff": self.abs_sum / self.n}
        if self.kind is not None:
            r[f"{prefix}{self.kind}_agreement"] = self.agree / self.count
        if self.dtype_changed:
            r[f"{prefix}dtype_changed"] = 1.0
        return r


def compare_outputs(reference, candidate, inputs: Iterable) -> dict[str, float]:
    """Numerical agreement of ``candidate`` (e.g. a quantised model) with ``reference`` over
    ``inputs`` (an iterable of model inputs, Tensor or dict). Either model may be an
    ``nn.Module`` or an ``ExportedProgram``.

    Returns ``max_abs_diff`` and ``mean_abs_diff`` of the raw outputs, plus:

    * ``argmax_agreement`` for floating outputs shaped like logits, ``(B, C)`` or
      ``(B, C, H, W)`` with ``C > 1``: the fraction of images or pixels whose argmax over dim 1
      is unchanged;
    * ``exact_agreement`` for integer or bool outputs (e.g. label maps): the fraction of equal
      elements;
    * ``dtype_changed`` = 1.0, with a warning, if the candidate's dtype differs.

    For dict outputs every key is prefixed with the head name, e.g.
    ``"segmentation/argmax_agreement"``.
    """
    reference, candidate = (m.module() if isinstance(m, torch.export.ExportedProgram) else m
                            for m in (reference, candidate))
    diffs: dict[str, _Diff] = {}
    with torch.no_grad():
        for x in inputs:
            ref, out = reference(x), candidate(x)
            if isinstance(ref, Mapping):
                for k, v in ref.items():
                    diffs.setdefault(f"{k}/", _Diff()).add(v, out[k])
            else:
                diffs.setdefault("", _Diff()).add(ref, out)
    if not diffs:
        raise ValueError("compare_outputs needs at least one input")
    if changed := [p.rstrip("/") or "output" for p, d in diffs.items() if d.dtype_changed]:
        warnings.warn(f"candidate output dtype differs from the reference for: {changed}",
                      stacklevel=2)
    return {k: v for prefix, d in diffs.items() for k, v in d.report(prefix).items()}
