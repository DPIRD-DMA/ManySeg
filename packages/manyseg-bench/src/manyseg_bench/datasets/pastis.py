"""PASTIS (Garnot & Landrieu 2021) as single-date crop-type semantic segmentation.

Licence: CC BY 4.0 (Zenodo record 5012942, doi:10.5281/zenodo.5012942). Please cite
V. Sainte Fare Garnot and L. Landrieu, "Panoptic Segmentation of Satellite Image Time Series
with Convolutional Temporal Attention Networks", ICCV 2021. Imagery: "Value-added data
processed by the CNES for the Theia www.theia.land.fr data cluster using Copernicus data";
annotations from the French LPIS (IGN).

The release has 2,433 patches of 128x128 px at 10 m from four Sentinel-2 tiles in France
(T30UXV, T31TFJ, T31TFM, T32ULU). Each patch is a time series ``DATA_S2/S2_<id>.npy``
``(T, 10, 128, 128)`` int16 (THEIA L2A surface reflectance x 1e4; bands B2, B3, B4, B5, B6,
B7, B8, B8A, B11, B12), T = 38-61 acquisitions between September 2018 and November 2019.
Only acquisitions with < 90% tile cloud cover were processed by THEIA; no further cloud
filtering, so some dates are partly or fully cloudy. ``ANNOTATIONS/TARGET_<id>.npy`` is
``(3, 128, 128)`` uint8 whose first channel is the semantic label: 0 background
(non-agricultural land), 1-18 crop classes, 19 void (parcels of out-of-nomenclature crops or
mostly outside the patch). As in the paper (and U-TAE's code: void gets zero loss weight and is
removed from the confusion matrix), **background is a class** and **void -> ignore (255)**,
so there are 19 classes. ``DATA_S2`` also holds 35 series that have neither a metadata entry
nor an annotation; they are skipped (ids recorded in ``meta.json``).

**Single-date samples.** The label is one per season (2019 LPIS declarations), but each image
is one acquisition: the same field looks very different across the season, so the acquisition
day of year should help tell crops apart. Every acquisition date of a tile is shared by all of
its patches (one date list per tile). Dates are taken from the 365-day **season window**
2018-11-01 .. 2019-10-31 (so a cyclic day of year is one-to-one within the data: September
2018, the previous season's stubble, is excluded); the window is cut into ``K_DATES`` = 8
equal parts and, for each part's centre, the nearest unused acquisition is chosen
(``select_dates``). Each (patch, date) is one sample named ``<patch>_<tile>_<YYYYMMDD>``.
19,464 samples, 6.3 GiB float16 (reflectance / 1e4, stored as input ``s2``); the cache stays
memory-mapped (``device_resident: False``).

**Splits.** The official fold I: train = folds 1-3, val = fold 4, test = fold 5. All dates of
a patch stay in its fold; the release keeps adjacent patches in the same fold. ``prepare``
re-checks that no patch id crosses splits. Normalisation stats come from train only.

**Variants** (one shared cache, ``pastis``):

* ``pastis``: S2 only (10 bands).
* ``pastis_meta``: S2 + vector input ``meta`` with field ``doy`` (cyclic, period 365.25).
* ``pastis_meta_shuffled``: control, field ``doy_shuffled``: the doy column permuted across
  all samples of the split (fixed seed). Keeps the marginal date distribution and breaks the
  date <-> image link.
* ``pastis_meta_tileshuffled``: control, field ``doy_tileshuffled``: permuted only among
  samples of the same Sentinel-2 tile. Because each tile has its own date list, ``doy``
  alone identifies the tile (and so the region's crop mix); this control keeps that
  information and removes only the seasonal (phenology) link.

The metadata fields come from the sample names only (no label-derived field can be used):
``N_Parcel`` and ``Parcel_Cover`` of ``metadata.geojson`` describe the annotations and are
refused by ``load_vectors``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from .base import DatasetInfo

URL = "https://zenodo.org/records/5012942/files/PASTIS.zip?download=1"
MD5 = "cfc441bf18137ff0bbf4fad58828fb98"
ZIP_NAME = "PASTIS.zip"
CACHE_NAME = "pastis"
BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
SIZE = 128
VOID = 19
IGNORE = 255
K_DATES = 8
SEASON = ("2018-11-01", "2019-10-31")  # 365 days: cyclic day of year is one-to-one
FOLDS = {"train": (1, 2, 3), "val": (4,), "test": (5,)}
EXPECTED_PATCHES = {"train": 1455, "val": 482, "test": 496}
SHUFFLE_SEED = 0
INPUTS = {"s2": 10}  # one raster input (dict-mode runs need it named)
# metadata.geojson properties that describe the annotations: never model inputs.
LABEL_DERIVED = ("N_Parcel", "Parcel_Cover")
CLASS_NAMES = ("background", "meadow", "soft_winter_wheat", "corn", "winter_barley",
               "winter_rapeseed", "spring_barley", "sunflower", "grapevine", "beet",
               "winter_triticale", "winter_durum_wheat", "fruits_vegetables_flowers",
               "potatoes", "leguminous_fodder", "soybeans", "orchard", "mixed_cereal",
               "sorghum")
LICENSE = ("CC BY 4.0; cite Sainte Fare Garnot & Landrieu, ICCV 2021, "
           "doi:10.5281/zenodo.5012942. Imagery: value-added data processed by CNES for the "
           "Theia data cluster using Copernicus data; annotations: French LPIS (IGN)")

INFO = DatasetInfo(
    name="pastis",
    num_classes=len(CLASS_NAMES),
    in_channels=len(BANDS),
    class_names=CLASS_NAMES,
    mean={}, std={},  # filled from meta.json
    ignore_index=IGNORE,
    license=LICENSE,
    source="https://zenodo.org/records/5012942",
    splits=("train", "val", "test"),
    test_splits=("test",),
    inputs=INPUTS,
    notes=("single-date samples: 8 dates per patch spread over 2018-11-01..2019-10-31",
           "official fold I: train folds 1-3, val fold 4, test fold 5",
           "void (19) ignored; background kept as a class",
           "every patch of a Sentinel-2 tile shares its date list: doy identifies the tile"),
)
DOY = {"type": "cyclic", "period": 365.25}
INFO_META = DatasetInfo(**{**INFO.to_dict(), "name": "pastis_meta",
                           "vectors": {"meta": {"fields": {"doy": DOY}}}})
INFO_META_SHUFFLED = DatasetInfo(**{**INFO.to_dict(), "name": "pastis_meta_shuffled",
                                    "vectors": {"meta": {"fields": {"doy_shuffled": DOY}}}})
INFO_META_TILESHUFFLED = DatasetInfo(**{
    **INFO.to_dict(), "name": "pastis_meta_tileshuffled",
    "vectors": {"meta": {"fields": {"doy_tileshuffled": DOY}}}})
VARIANTS = (INFO_META, INFO_META_SHUFFLED, INFO_META_TILESHUFFLED)
TASK = "semantic"
LOADER_KWARGS = {"device_resident": False, "vflip": True}
EVAL_BATCH_SIZE = 32


# --- names and dates ----------------------------------------------------------------------------

def _date(v) -> datetime.date:
    s = str(v).replace("-", "")
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def sample_name(patch: int | str, tile: str, date: int | str) -> str:
    return f"{patch}_{tile}_{_date(date):%Y%m%d}"


def parse_name(name: str) -> tuple[str, str, datetime.date]:
    """``(patch id, tile, acquisition date)`` of a cached sample name."""
    patch, tile, date = name.split("_")
    return patch, tile, _date(date)


def doy_of(name: str) -> float:
    return float(parse_name(name)[2].timetuple().tm_yday)


def select_dates(dates, k: int = K_DATES, season: tuple[str, str] = SEASON) -> list[int]:
    """Indices (into ``dates``, ``YYYYMMDD``) of ``k`` acquisitions spread over ``season``.

    The season window is cut into ``k`` equal parts; from each part, the acquisition nearest the
    part's centre is taken (ties -> the earlier date), so every date comes from its own part.
    A part without acquisitions contributes nothing (fewer than ``k`` dates)."""
    d = [_date(v) for v in dates]
    lo, hi = _date(season[0]), _date(season[1])
    span = (hi - lo).days + 1
    chosen: list[int] = []
    for j in range(k):
        a = lo + datetime.timedelta(days=j * span / k)
        b = lo + datetime.timedelta(days=(j + 1) * span / k)
        part = [i for i, x in enumerate(d) if a <= x < b and lo <= x <= hi]
        if part:
            centre = lo + datetime.timedelta(days=(j + 0.5) * span / k)
            chosen.append(min(part, key=lambda i: (abs((d[i] - centre).days), d[i])))
    return sorted(chosen, key=lambda i: d[i])


# --- raw layout ---------------------------------------------------------------------------------

def _root(raw: Path) -> Path:
    for cand in (Path(raw) / "PASTIS", Path(raw)):
        if (cand / "metadata.geojson").exists():
            return cand
    raise FileNotFoundError(f"no PASTIS/metadata.geojson under {raw}; download {URL} into "
                            f"{raw} (md5 {MD5}), unzip it there and write 'MD5_OK' to "
                            f"{Path(raw) / 'download.log'}")


def patch_table(root: Path) -> list[dict]:
    """One dict per patch (``metadata.geojson`` order): id, fold, tile and S2 dates in time
    index order. Label-derived properties are not read."""
    feats = json.loads((Path(root) / "metadata.geojson").read_text())["features"]
    out = []
    for f in feats:
        p = f["properties"]
        ds = p["dates-S2"]
        if isinstance(ds, str):
            ds = json.loads(ds.replace("'", '"'))
        dates = [int(ds[str(i)]) for i in range(len(ds))]
        out.append({"id": int(p["ID_PATCH"]), "fold": int(p["Fold"]),
                    "tile": str(p["TILE"]).lower(), "dates": dates})
    return out


def check_splits(patches: list[dict], expected: dict | None = EXPECTED_PATCHES
                 ) -> dict[str, list[dict]]:
    """Group patches by split (official fold I); check counts and that no id crosses splits."""
    fold_split = {f: s for s, fs in FOLDS.items() for f in fs}
    by_split: dict = {s: [] for s in FOLDS}
    seen: dict = {}
    for p in patches:
        if p["fold"] not in fold_split:
            raise RuntimeError(f"integrity: patch {p['id']} has fold {p['fold']}")
        s = fold_split[p["fold"]]
        if p["id"] in seen:
            what = "crosses splits" if seen[p["id"]] != s else "occurs twice"
            raise RuntimeError(f"integrity: patch {p['id']} {what}")
        seen[p["id"]] = s
        by_split[s].append(p)
    if expected is not None:
        got = {k: len(v) for k, v in by_split.items()}
        if got != expected:
            raise RuntimeError(f"integrity: split sizes {got}, expected {expected}")
    return by_split


def check_download(raw: Path) -> bool:
    log = Path(raw) / "download.log"
    return log.exists() and any(ln.strip() == "MD5_OK" for ln in log.read_text().splitlines())


def read_patch(job):
    """``(names, (k, H, W, 10) float16 reflectance, (H, W) uint8 mask, blue means)``."""
    root, p, k = job
    idx = select_dates(p["dates"], k)
    s2 = np.load(Path(root) / "DATA_S2" / f"S2_{p['id']}.npy", mmap_mode="r")
    if s2.ndim != 4 or s2.shape[0] != len(p["dates"]) or s2.shape[1] != len(BANDS):
        raise RuntimeError(f"integrity: S2_{p['id']} has shape {s2.shape}, "
                           f"{len(p['dates'])} dates in the metadata")
    img = np.moveaxis(np.asarray(s2[idx], dtype=np.float32) / 1e4, 1, -1).astype(np.float16)
    lab = np.load(Path(root) / "ANNOTATIONS" / f"TARGET_{p['id']}.npy")[0]
    if lab.shape != s2.shape[2:]:
        raise RuntimeError(f"integrity: TARGET_{p['id']} is {lab.shape}, S2 {s2.shape}")
    bad = set(np.unique(lab).tolist()) - set(range(VOID + 1))
    if bad:
        raise RuntimeError(f"integrity: TARGET_{p['id']} has label values {sorted(bad)}")
    mask = np.where(lab == VOID, IGNORE, lab).astype(np.uint8)
    names = [sample_name(p["id"], p["tile"], p["dates"][i]) for i in idx]
    blue = img[..., 0].astype(np.float32).mean(axis=(1, 2))
    return names, img, mask, blue


def prepare(raw: Path, cache: Path, size: int = SIZE, workers: int = 4, k: int = K_DATES,
            expected: dict | None = EXPECTED_PATCHES, require_checksum: bool = True,
            log=lambda m: print(m, flush=True)) -> dict:
    """Select ``k`` dates per patch and cache single-date samples (see the module docstring).
    Arrays are written through memory maps; RAM stays bounded."""
    import contextlib
    from concurrent.futures import ProcessPoolExecutor

    raw, cache = Path(raw), Path(cache)
    root = _root(raw)
    if require_checksum and not check_download(raw):
        raise RuntimeError(f"integrity: {raw / 'download.log'} has no MD5_OK line "
                           f"(verify {ZIP_NAME} against md5 {MD5})")
    by_split = check_splits(patch_table(root), expected)
    cache.mkdir(parents=True, exist_ok=True)
    summary, stats = {}, {}
    nc = len(CLASS_NAMES)
    with (ProcessPoolExecutor(workers) if workers > 1 else contextlib.nullcontext()) as ex:
        for split, patches in by_split.items():
            n = sum(len(select_dates(p["dates"], k)) for p in patches)
            images = np.lib.format.open_memmap(cache / f"{split}_images.s2.npy", "w+", np.float16,
                                               (n, size, size, len(BANDS)))
            masks = np.lib.format.open_memmap(cache / f"{split}_masks.npy", "w+", np.uint8,
                                              (n, size, size))
            acc = np.zeros((3, len(BANDS)))  # sum, sum of squares, count
            classes = np.zeros(256, np.int64)
            names: list[str] = []
            blues: list[float] = []
            jobs = [(root, p, k) for p in patches]

            def results(jobs=jobs):
                # Bounded window: at most 4 x workers patches in flight, so finished results
                # cannot pile up behind a slow early one (RAM stays bounded).
                step = 4 * workers
                for s0 in range(0, len(jobs), step):
                    batch = jobs[s0:s0 + step]
                    yield from (ex.map(read_patch, batch) if ex is not None
                                else map(read_patch, batch))

            for b, (nm, img, m, blue) in enumerate(results()):
                if img.shape[1:3] != (size, size):
                    raise RuntimeError(f"integrity: {nm[0]} is {img.shape[1:3]}, expected {size}")
                j = len(names)
                images[j:j + len(nm)] = img
                masks[j:j + len(nm)] = m
                x = img.reshape(-1, len(BANDS)).astype(np.float64)
                ok = np.isfinite(x)
                x = np.where(ok, x, 0.0)
                acc += (x.sum(0), (x * x).sum(0), ok.sum(0))
                classes += np.bincount(m.ravel(), minlength=256) * len(nm)
                names += nm
                blues += blue.tolist()
                if b % 250 == 0:
                    log(f"{split}: {b + 1}/{len(patches)} patches, {len(names)}/{n} samples")
            if len(names) != n:
                raise RuntimeError(f"integrity: {split} wrote {len(names)} samples, expected {n}")
            images.flush()
            masks.flush()
            del images, masks
            (cache / f"{split}_names.json").write_text(json.dumps(names))
            if split == "train":
                s_, ss, cnt = acc
                mean = s_ / np.maximum(cnt, 1)
                std = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 0.0))
                stats = {"mean": {"s2": mean.tolist()}, "std": {"s2": (std + 1e-6).tolist()}}
            labelled = classes[:nc].sum()
            doys = [doy_of(x) for x in names]
            months: dict = {}
            for x in names:
                key = f"{parse_name(x)[2]:%Y-%m}"
                months[key] = months.get(key, 0) + 1
            blues_a = np.asarray(blues)
            summary[split] = {
                "patches": len(patches), "samples": n,
                "tiles": {t: sum(p["tile"] == t for p in patches)
                          for t in sorted({p["tile"] for p in patches})},
                "samples_per_patch": sorted({len(select_dates(p["dates"], k)) for p in patches}),
                "class_frac": dict(zip(CLASS_NAMES, (classes[:nc] / max(labelled, 1))
                                       .round(5).tolist())),
                "void_frac": float(classes[IGNORE] / max(classes.sum(), 1)),
                "doy_min_max": [min(doys), max(doys)] if doys else None,
                "samples_per_month": dict(sorted(months.items())),
                # Blue (B2) reflectance > 0.2 averaged over the patch: mostly cloud or snow.
                "bright_blue_frac": float((blues_a > 0.2).mean()) if blues else 0.0,
                "blue_mean_quantiles": (np.quantile(blues_a, [0.1, 0.5, 0.9]).round(4).tolist()
                                        if blues else None)}
            log(f"{split}: {json.dumps(summary[split])}")
    # The release ships some S2 series with no metadata entry and no annotation: never used.
    known = {p["id"] for ps in by_split.values() for p in ps}
    unlisted = sorted(int(f.stem[3:]) for f in (root / "DATA_S2").glob("S2_*.npy")
                      if f.stem[3:].isdigit() and int(f.stem[3:]) not in known)
    summary["unlisted_s2_series"] = len(unlisted)
    meta = {**stats, "inputs": INPUTS, "bands": BANDS, "k_dates": k, "season": SEASON,
            "summary": summary, "unlisted_s2_ids": unlisted}
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    return summary


# --- metadata vectors ---------------------------------------------------------------------------

def shuffled_index(groups: list[str], seed: int = SHUFFLE_SEED) -> np.ndarray:
    """A permutation ``perm`` of ``range(len(groups))`` that only moves samples within their
    group (``groups[perm[i]] == groups[i]``); one group = a global shuffle."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(groups))
    for g in sorted(set(groups)):
        idx = np.flatnonzero(np.asarray(groups) == g)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def load_vectors(raw: Path, names: list[str], vectors: dict) -> dict[str, np.ndarray]:
    """``{vector input: (N, columns) float32}`` for cached samples ``names``, from the names
    alone. Fields: ``doy`` (acquisition day of year), ``doy_shuffled`` (doy permuted across
    the given samples, seed ``SHUFFLE_SEED``), ``doy_tileshuffled`` (permuted within each
    Sentinel-2 tile). Label-derived fields are refused."""
    doy = np.asarray([doy_of(n) for n in names], dtype=np.float32)
    out = {}
    for vname, cfg in vectors.items():
        fields = list(cfg["fields"])
        leaked = [f for f in fields if f in LABEL_DERIVED or "parcel" in f.lower()]
        if leaked:
            raise ValueError(f"label-derived metadata fields are not allowed: {leaked}")
        cols = []
        for f in fields:
            if f == "doy":
                cols.append(doy)
            elif f == "doy_shuffled":
                cols.append(doy[shuffled_index(["all"] * len(names))])
            elif f == "doy_tileshuffled":
                cols.append(doy[shuffled_index([parse_name(n)[1] for n in names])])
            else:
                raise ValueError(f"no extractor for PASTIS metadata field {f!r}")
        out[vname] = np.stack(cols, 1).astype(np.float32)
    return out
