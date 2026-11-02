"""The losses the benchmark trains with, from :mod:`manyseg.losses` (where they moved before
ManySeg 0.1.0); re-exported so ``from manyseg_bench import losses`` keeps working."""

from manyseg.losses import (  # noqa: F401
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
