"""Provider-neutral supervised training loop for semantic segmentation."""

from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..evaluation import ConfusionMatrix, boundary_f1
from .recipe import Recipe

_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device: torch.device, precision: str):
    dt = _DTYPES[precision]
    return nullcontext() if dt is None else torch.autocast(device.type, dtype=dt)


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
    return torch.optim.AdamW(groups, lr=recipe.lr, betas=recipe.betas, eps=recipe.eps)


def lr_factor(step: int, total: int, warmup: int, schedule: str) -> float:
    if schedule == "constant":
        return min(1.0, (step + 1) / warmup) if warmup else 1.0
    if schedule == "cosine":
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    if schedule == "onecycle":  # fastai: 25%->100% over 30%, cosine down to ~0
        pct = 0.3
        up = max(1, int(total * pct))
        if step < up:
            t = step / up
            return 1 / 25 + (1 - 1 / 25) * (1 - math.cos(math.pi * t)) / 2
        t = (step - up) / max(1, total - up)
        return 1e-5 + (1 - 1e-5) * (1 + math.cos(math.pi * t)) / 2
    raise ValueError(f"unknown schedule {schedule!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, num_classes: int, device: torch.device,
             precision: str = "bf16", ignore_index: int | None = None) -> dict:
    model.eval()
    cm = ConfusionMatrix(num_classes, ignore_index, device=device)
    bf1_sum, n_img, loss_sum = 0.0, 0, 0.0
    for x, y in loader:
        with autocast(device, precision):
            logits = model(x)
        logits = logits.float()
        loss_sum += float(F.cross_entropy(logits, y, ignore_index=ignore_index
                                          if ignore_index is not None else -100, reduction="sum"))
        pred = logits.argmax(1)
        cm.update(pred, y)
        bf1_sum += float(boundary_f1(pred, y, ignore_index=ignore_index).sum())
        n_img += x.shape[0]
    out = cm.compute()
    out["boundary_f1"] = bf1_sum / max(1, n_img)
    out["loss"] = loss_sum / max(1, n_img * y.shape[-1] * y.shape[-2])
    return out


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
    log=print,
) -> dict:
    """Train ``model`` and return history. Selection metric: last epoch (no early stopping,
    so every provider gets the same budget); best-epoch val is reported for information."""
    model = model.to(device)
    if recipe.channels_last:
        model = model.to(memory_format=torch.channels_last)
    run = torch.compile(model) if recipe.compile else model
    opt = build_optimizer(enc_params, dec_params, recipe)
    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * recipe.epochs
    warmup = int(recipe.warmup_epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=recipe.precision == "fp16")
    ce_ignore = ignore_index if ignore_index is not None else -100

    history, step = [], 0
    t_start = time.perf_counter()
    for epoch in range(recipe.epochs):
        frozen = epoch < recipe.freeze_encoder_epochs
        for p in enc_params:  # fastai-style freeze keeps norm params trainable
            p.requires_grad_(not frozen or _is_no_decay(p))
        model.train()
        t0 = time.perf_counter()
        loss_sum, nb, nonfinite = 0.0, 0, 0
        for x, y in train_loader:
            if recipe.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            f = lr_factor(step, total, warmup, recipe.schedule)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.zero_grad(set_to_none=True)
            with autocast(device, recipe.precision):
                logits = run(x)
            loss = F.cross_entropy(logits.float(), y, ignore_index=ce_ignore,
                                   label_smoothing=recipe.label_smoothing)
            if not torch.isfinite(loss):
                nonfinite += 1
                step += 1
                continue
            scaler.scale(loss).backward()
            if recipe.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach())
            nb += 1
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, nb),
               "nonfinite_steps": nonfinite, "epoch_time_s": time.perf_counter() - t0,
               "encoder_frozen": frozen}
        if (epoch + 1) % recipe.eval_every == 0 or epoch + 1 == recipe.epochs:
            rec["val"] = evaluate(run, val_loader, num_classes, device, recipe.precision,
                                  ignore_index)
            log(f"epoch {epoch + 1}/{recipe.epochs} loss {rec['train_loss']:.4f} "
                f"val mIoU {rec['val']['miou']:.4f} ({rec['epoch_time_s']:.1f}s)")
        history.append(rec)
    for p in enc_params:
        p.requires_grad_(True)
    evals = [h for h in history if "val" in h]
    best = max(evals, key=lambda h: h["val"]["miou"])
    return {
        "history": history,
        "final_val": evals[-1]["val"],
        "best_val_miou": best["val"]["miou"],
        "best_epoch": best["epoch"],
        "train_time_s": time.perf_counter() - t_start,
        "peak_train_mem_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                               if device.type == "cuda" else None),
    }
