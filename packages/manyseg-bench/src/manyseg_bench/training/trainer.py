"""Provider-neutral supervised training loop for semantic segmentation."""

from __future__ import annotations

import functools
import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from .recipe import Recipe

_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encoder_bn(model: nn.Module, enc_params) -> list[nn.Module]:
    """BatchNorm modules inside the encoder: matched by module path (a sibling of, or the owner
    of, an encoder parameter), so BN layers without affine weights are found too."""
    ids = {id(p) for p in enc_params}
    owners = {n.rsplit(".", 1)[0] for n, p in model.named_parameters() if id(p) in ids}
    parents = {o.rsplit(".", 1)[0] if "." in o else "" for o in owners}
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            if name in owners or parent in parents:
                out.append(m)
    return out


def autocast(device: torch.device, precision: str):
    dt = _DTYPES[precision]
    return nullcontext() if dt is None else torch.autocast(device.type, dtype=dt)


def _channels_last(x):
    if isinstance(x, dict):  # vector (metadata) inputs are (B, N): left as they are
        return {k: v.contiguous(memory_format=torch.channels_last) if v.ndim == 4 else v
                for k, v in x.items()}
    return x.contiguous(memory_format=torch.channels_last)


def _batch_size(x) -> int:
    return next(v for v in x.values() if v.ndim >= 4).shape[0] if isinstance(x, dict) \
        else x.shape[0]


def _is_no_decay(p: nn.Parameter) -> bool:
    return p.ndim <= 1  # norm weights/biases and conv/linear biases


def build_optimizer(enc, dec, recipe: Recipe) -> torch.optim.Optimizer:
    groups = []
    for params, mult, tag in ((enc, recipe.encoder_lr_mult, "encoder"), (dec, 1.0, "decoder")):
        if recipe.no_decay_norm_bias:
            decay = [p for p in params if not _is_no_decay(p)]
            no_decay = [p for p in params if _is_no_decay(p)]
            parts = [(decay, recipe.weight_decay), (no_decay, 0.0)]
        else:
            parts = [(list(params), recipe.weight_decay)]
        for ps, wd in parts:
            if ps:
                groups.append({"params": ps, "lr": recipe.lr * mult, "base_lr": recipe.lr * mult,
                               "weight_decay": wd, "tag": tag})
    # fused (one kernel for the whole step) when every parameter is on CUDA: same update up to
    # rounding, and it can skip a step on a device-side non-finite flag (see train)
    fused = bool(groups) and all(p.is_cuda for g in groups for p in g["params"])
    return torch.optim.AdamW(groups, lr=recipe.lr, betas=recipe.betas, eps=recipe.eps,
                             fused=fused or None)


def lr_factor(step: int, total: int, warmup: int, schedule: str) -> float:
    if schedule == "constant":
        return min(1.0, (step + 1) / warmup) if warmup else 1.0
    if schedule == "cosine":
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    if schedule == "onecycle":  # LR/25 up to LR over 30%, cosine down to ~0
        pct = 0.3
        up = max(1, int(total * pct))
        if step < up:
            t = step / up
            return 1 / 25 + (1 - 1 / 25) * (1 - math.cos(math.pi * t)) / 2
        t = (step - up) / max(1, total - up)
        return 1e-5 + (1 - 1e-5) * (1 + math.cos(math.pi * t)) / 2
    raise ValueError(f"unknown schedule {schedule!r}")


def _pad_batch(x, n: int):
    """``x`` (a tensor or dict of tensors with the batch first) padded to ``n`` samples by
    repeating its first sample."""
    def pad(t):
        if not isinstance(t, torch.Tensor) or t.ndim == 0 or t.shape[0] >= n:
            return t
        return torch.cat([t, t[:1].expand(n - t.shape[0], *t.shape[1:])])
    return {k: pad(v) for k, v in x.items()} if isinstance(x, dict) else pad(x)


def _take(out, n: int):
    """The first ``n`` samples of a model output (a tensor or dict of tensors)."""
    return {k: v[:n] for k, v in out.items()} if isinstance(out, dict) else out[:n]


@torch.no_grad()
def evaluate(model: nn.Module, loader, num_classes: int, device: torch.device,
             precision: str = "bf16", ignore_index: int | None = None, task=None,
             pad_last: bool = False) -> dict:
    """``pad_last``: run a short final batch padded to the first batch's size (the padding is
    dropped before the metrics), so a static compiled model sees one shape."""
    from ..tasks import SemanticTask

    task = task or SemanticTask(num_classes, ignore_index)
    model.eval()
    ev = task.evaluator()
    full = None
    for x, y in loader:
        n = _batch_size(x)
        full = full or n
        with autocast(device, precision):
            out = model(_pad_batch(x, full)) if pad_last and n < full else model(x)
        ev.update(_take(out, n) if pad_last and n < full else out, y)
    return ev.compute()


def train(
    model: nn.Module,
    enc_params: list,
    dec_params: list,
    train_loader,
    val_loader,
    *,
    num_classes: int,
    recipe: Recipe,
    device: torch.device,
    ignore_index: int | None = None,
    log=None,
    task=None,
) -> dict:
    """Train ``model`` and return history. Selection metric: last epoch (no early stopping,
    so every provider gets the same budget); best-epoch val is reported for information."""
    from ..tasks import SemanticTask

    log = log or functools.partial(print, flush=True)
    task = task or SemanticTask(num_classes, ignore_index, recipe.label_smoothing)
    model = model.to(device)
    if recipe.channels_last:
        model = model.to(memory_format=torch.channels_last)
    run = torch.compile(model) if recipe.compile else model
    # Evaluation gets its own static compile, fed one shape (the short last batch padded): the
    # shared callable recompiles for the eval batch size and then runs dynamic-shape kernels
    # (E027: +13-25% evaluation throughput for the PVTv2-b0 presets).
    eval_run = torch.compile(model, dynamic=False) if recipe.compile else model
    opt = build_optimizer(enc_params, dec_params, recipe)
    accum = max(1, int(recipe.grad_accum_steps))
    micro_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)  # optimiser steps
    total = steps_per_epoch * recipe.epochs
    warmup = int(recipe.warmup_epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=recipe.precision == "fp16")
    # No host sync per step (E026: -24% step time on small models): with one micro-batch per
    # step and a fused optimiser, a non-finite loss skips the update on the device (the fused
    # kernel's found_inf, as GradScaler uses), and losses are summed on the device. Otherwise
    # (accumulation, fp16 scaling, other optimisers) the per-micro-batch check below stays.
    device_skip = (accum == 1 and not scaler.is_enabled()
                   and bool(getattr(opt, "defaults", {}).get("fused")))

    enc_bn = _encoder_bn(model, enc_params) if recipe.freeze_encoder_bn else []
    history, step = [], 0
    t_start = time.perf_counter()
    for epoch in range(recipe.epochs):
        frozen = epoch < recipe.freeze_encoder_epochs
        for p in enc_params:  # fastai-style freeze keeps norm params trainable
            p.requires_grad_(not frozen or _is_no_decay(p))
        model.train()
        for m in enc_bn:  # running statistics stay at their pretrained values
            m.eval()
        t0 = time.perf_counter()
        loss_sum, nb, nonfinite, n_img = 0.0, 0, 0, 0
        loss_dev = torch.zeros((), device=device)  # device_skip: summed without a sync
        nonfinite_dev = torch.zeros((), device=device, dtype=torch.long)
        skipped_img_dev = torch.zeros((), device=device, dtype=torch.long)
        for i, (x, y) in enumerate(train_loader):
            n_img += _batch_size(x)
            if recipe.channels_last:
                x = _channels_last(x)
            first, start = i % accum == 0, i - i % accum
            group = min(accum, micro_per_epoch - start)  # the last group may be shorter
            last = i - start == group - 1
            if first:
                f = lr_factor(step, total, warmup, recipe.schedule)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                opt.zero_grad(set_to_none=True)
                backward, group_n = False, 0
            with autocast(device, recipe.precision):
                out = run(x)
            loss = task.loss(out, task.targets(y))
            if device_skip:
                n = _batch_size(x)
                bad = ~torch.isfinite(loss.detach())
                scaler.scale(loss * n).backward()
                backward, group_n = True, n
                loss_dev += torch.where(bad, 0.0, loss.detach().float()) * n
                nonfinite_dev += bad.long()
                skipped_img_dev += bad.long() * n
                nb += n  # skipped images are subtracted at the end of the epoch
                opt.found_inf = bad.float()  # the fused step leaves everything as it was
            elif not torch.isfinite(loss):
                nonfinite += 1
            else:
                # Sum of per-image-weighted losses; divided by the group's image count below,
                # so a short micro-batch weighs as many images as it has.
                n = _batch_size(x)
                scaler.scale(loss * n).backward()
                backward, group_n = True, group_n + n
                loss_sum += float(loss.detach()) * n  # per-image mean, as optimised
                nb += n
            if not last:
                continue
            if not backward:  # every micro-batch was non-finite: skip the update
                step += 1
                continue
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                torch._foreach_div_(grads, group_n)
            if recipe.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if device_skip:
            nonfinite += int(nonfinite_dev)
            loss_sum += float(loss_dev)
            nb -= int(skipped_img_dev)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, nb),
               "nonfinite_steps": nonfinite, "epoch_time_s": time.perf_counter() - t0,
               "train_images": n_img, "encoder_frozen": frozen}
        if (epoch + 1) % recipe.eval_every == 0 or epoch + 1 == recipe.epochs:
            rec["val"] = evaluate(eval_run, val_loader, num_classes, device, recipe.precision,
                                  ignore_index, task=task, pad_last=recipe.compile)
            pm = task.primary_metric
            name = "mIoU" if pm == "miou" else pm
            log(f"epoch {epoch + 1}/{recipe.epochs} loss {rec['train_loss']:.4f} "
                f"val {name} {rec['val'][pm]:.4f} ({rec['epoch_time_s']:.1f}s)")
        history.append(rec)
    for p in enc_params:
        p.requires_grad_(True)
    evals = [h for h in history if "val" in h]
    pm, sign = task.primary_metric, (1 if task.higher_is_better else -1)
    best = max(evals, key=lambda h: sign * h["val"][pm])
    # Training throughput from the timed epochs (data loading + step; excludes evaluation).
    # "steady" skips the first epoch (compilation, cudnn autotuning, allocator warm-up).
    steady = history[1:] or history
    return {
        "train_images_per_s": sum(h["train_images"] for h in history)
        / max(1e-9, sum(h["epoch_time_s"] for h in history)),
        "train_images_per_s_steady": sum(h["train_images"] for h in steady)
        / max(1e-9, sum(h["epoch_time_s"] for h in steady)),
        "history": history,
        "task": task.name,
        "primary_metric": pm,
        "final_val": evals[-1]["val"],
        f"best_val_{pm}": best["val"][pm],
        "best_epoch": best["epoch"],
        "train_time_s": time.perf_counter() - t_start,
        "peak_train_mem_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                               if device.type == "cuda" else None),
    }
