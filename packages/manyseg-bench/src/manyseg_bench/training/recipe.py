from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Recipe:
    """Optimisation protocol. Applied identically to every provider's model.

    The only provider-dependent input is the encoder/decoder parameter split, which every
    provider must supply; ``encoder_lr_mult`` applies the same multiplier to all of them.
    """

    name: str = "standard"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 5e-4
    encoder_lr_mult: float = 1.0
    weight_decay: float = 1e-4
    no_decay_norm_bias: bool = True
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_epochs: float = 1.0
    schedule: str = "cosine"  # cosine | onecycle | constant
    freeze_encoder_epochs: int = 0
    # Keep the encoder's BatchNorm layers in eval mode (running statistics fixed at their
    # pretrained values; affine weights still train). For small batches (E007).
    freeze_encoder_bn: bool = False
    # Optimiser step every k micro-batches of ``batch_size`` (effective batch k*batch_size;
    # BN still sees micro-batches). The lr schedule counts optimiser steps (E007b).
    grad_accum_steps: int = 1
    precision: str = "bf16"  # fp32 | bf16 | fp16
    grad_clip: float | None = None
    compile: bool = False
    channels_last: bool = False
    label_smoothing: float = 0.0
    eval_every: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> Recipe:
        return replace(self, **kw)


RECIPES = {
    # A conventional modern recipe (AdamW + warmup + cosine), single LR for all parameters.
    "standard": Recipe(),
    # Mirrors fastai unet_learner + fine_tune defaults (see docs/architecture_review.md):
    # AdamW(0.9, 0.99, eps 1e-5, wd 1e-2, no wd on norm/bias), 1 frozen epoch, then
    # discriminative LRs (encoder 0.1x), one-cycle.
    "fastai_like": Recipe(
        name="fastai_like", lr=1e-3, encoder_lr_mult=0.1, weight_decay=1e-2,
        betas=(0.9, 0.99), eps=1e-5, schedule="onecycle", freeze_encoder_epochs=1,
        warmup_epochs=0.0,
    ),
    # Tiny smoke-test protocol for CI.
    "smoke": Recipe(name="smoke", epochs=1, batch_size=4, warmup_epochs=0.0),
}


def get_recipe(name: str, **overrides) -> Recipe:
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name!r}; available: {sorted(RECIPES)}")
    return RECIPES[name].with_(**overrides) if overrides else RECIPES[name]
