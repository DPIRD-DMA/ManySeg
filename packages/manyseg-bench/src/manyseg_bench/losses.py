"""The losses the benchmark trains with, from :mod:`manyseg.losses` (where they moved before
ManySeg 0.1.0); re-exported so ``from manyseg_bench import losses`` keeps working.

The re-export is lazy: the benchmark core (tasks, runner) imports this module, and must import
without ManySeg installed (``manyseg-bench[smp]`` trains semantic tasks with SMP alone). A loss
is looked up in :mod:`manyseg.losses` on first access, which then needs ManySeg.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and linters; at runtime see __getattr__
    from manyseg.losses import (
        _as_bhw,
        density_loss,
        dice_bce,
        distance_map,
        distance_weighted_bce,
        heatmap_focal,
        heatmap_peaks,
        masked_regression,
        multi_head_loss,
        multiclass_dice,
        soft_dice,
    )

__all__ = [
    "_as_bhw",
    "density_loss",
    "dice_bce",
    "distance_map",
    "distance_weighted_bce",
    "heatmap_focal",
    "heatmap_peaks",
    "masked_regression",
    "multi_head_loss",
    "multiclass_dice",
    "soft_dice",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import manyseg.losses as _losses
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"manyseg_bench.losses.{name} comes from ManySeg: "
                          "pip install 'manyseg-bench[manyseg]'") from e
    value = getattr(_losses, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
