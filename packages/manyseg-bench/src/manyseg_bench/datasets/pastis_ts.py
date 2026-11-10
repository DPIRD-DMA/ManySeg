"""PASTIS as a time series: the 8 dates per patch of the ``pastis`` cache (see ``pastis.py`` for
the licence, the date selection and the splits) regrouped into ``(T=8, H, W, C)`` samples.

``prepare`` reads the prepared ``pastis`` cache (run ``manyseg-bench prepare pastis`` first) and
writes ``(N, 8, 128, 128, 10)`` float16 series ordered by date, the patch label (identical for
all dates of a patch; checked) and ``<split>_dates.npy``: ``(N, 8)`` day of year within the
season window (days since 2018-11-01, so values increase through the season).

Variants (one cache, ``pastis_ts``):

* ``pastis_ts``: the series, input ``s2`` with ``s2_dates``.
* ``pastis_ts_single``: one date per patch, the one nearest the season's sixth eighth
  (early July, peak growing season; index ``SINGLE_INDEX``), as an ordinary raster: the
  single-date baseline on exactly the same patches and labels.

Unlike U-TAE (all 38-61 acquisitions), only 8 dates are used, and no cloud masks exist: about
15% of dates are cloudy (``bright_blue_frac`` in the ``pastis`` cache's ``meta.json``).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from . import pastis
from .base import DatasetInfo, check_native_size, load_split

CACHE_NAME = "pastis_ts"
T = pastis.K_DATES
SINGLE_INDEX = 5
SEASON_START = datetime.date(2018, 11, 1)
TASK = "semantic"
LOADER_KWARGS = {"device_resident": False, "vflip": True}
EVAL_BATCH_SIZE = 16

INFO = DatasetInfo(**{**pastis.INFO.to_dict(), "name": "pastis_ts", "timeseries": {"s2": T},
                      "notes": ("8 dates per patch from the pastis cache", *pastis.INFO.notes)})
INFO_SINGLE = DatasetInfo(**{**pastis.INFO.to_dict(), "name": "pastis_ts_single",
                             "notes": (f"date index {SINGLE_INDEX} of 8 (early July)",)})
VARIANTS = (INFO_SINGLE,)


def season_day(name: str) -> float:
    return float((pastis.parse_name(name)[2] - SEASON_START).days)


def group_by_patch(names: list[str]) -> list[tuple[str, list[int]]]:
    """``[(patch name, sample indices ordered by date)]``, patches in first-seen order."""
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        patch, tile, _ = pastis.parse_name(n)
        groups.setdefault(f"{patch}_{tile}", []).append(i)
    return [(p, sorted(idx, key=lambda i: season_day(names[i]))) for p, idx in groups.items()]


def prepare(raw: Path, cache: Path, size: int | None = None) -> dict:
    """Group the prepared ``pastis`` cache's dates into series. The tile size is that cache's
    (``prepare pastis --size``); another ``size`` raises."""
    from numpy.lib.format import open_memmap

    cache = Path(cache)
    src = cache.parent / pastis.CACHE_NAME
    if not (src / "train_names.json").exists():
        raise FileNotFoundError(f"{src} is not prepared; run `manyseg-bench prepare pastis`")
    check_native_size(size, int(load_split(src, "train").images["s2"].shape[1]), "pastis_ts")
    cache.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        s = load_split(src, split)
        groups = group_by_patch(s.names)
        if any(len(idx) != T for _, idx in groups):
            raise RuntimeError(f"{split}: every patch needs {T} dates")
        img = s.images["s2"]
        out = open_memmap(cache / f"{split}_images.s2.npy", "w+", img.dtype,
                          (len(groups), T, *img.shape[1:]))
        masks = np.empty((len(groups), *s.masks.shape[1:]), s.masks.dtype)
        dates = np.empty((len(groups), T), np.float32)
        for g, (_, idx) in enumerate(groups):
            out[g] = img[idx]
            labels = np.asarray(s.masks[idx])
            if not (labels == labels[0]).all():
                raise RuntimeError(f"{split}: labels differ between dates of {groups[g][0]}")
            masks[g] = labels[0]
            dates[g] = [season_day(s.names[i]) for i in idx]
        out.flush()
        np.save(cache / f"{split}_masks.npy", masks)
        np.save(cache / f"{split}_dates.npy", dates)
        (cache / f"{split}_names.json").write_text(json.dumps([p for p, _ in groups]))
        summary[split] = {"patches": len(groups), "date_range": [float(dates.min()),
                                                                   float(dates.max())]}
        del out
    meta = json.loads((src / "meta.json").read_text())
    (cache / "meta.json").write_text(json.dumps(
        {"mean": meta["mean"], "std": meta["std"], "summary": summary}, indent=1))
    return summary


def series_extras(cache: Path, split_name: str) -> dict[str, np.ndarray]:
    """``{"s2_dates": (N, T)}`` for a split (days since 2018-11-01)."""
    return {"s2_dates": np.load(Path(cache) / f"{split_name}_dates.npy")}


def single_date(split):
    """``pastis_ts_single``: keep date ``SINGLE_INDEX`` as an ordinary raster."""
    split.images = {k: np.asarray(v[:, SINGLE_INDEX]) for k, v in split.images.items()}
    return split


VARIANT_OVERRIDES = {"pastis_ts_single": {"series_extras": None, "transform_split": single_date}}
