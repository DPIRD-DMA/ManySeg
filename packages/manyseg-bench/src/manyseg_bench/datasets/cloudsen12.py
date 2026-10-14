"""CloudSEN12+ expert-labelled ("high") subset (Aybar et al. 2024).

Licence: CC0-1.0 for the labels and curation; the imagery is subject to the Copernicus
Sentinel data notice, so derived data must state "Contains modified Copernicus Sentinel data
2017-2022" (``SENTINEL_NOTICE``, recorded in every run's dataset licence). The auxiliary
layers add S1GBM (CC BY 4.0, Bauer-Marschallinger et al. 2021, doi:10.1038/s41597-021-01059-7)
and the Copernicus DEM GLO-30 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
2014-2018, provided under COPERNICUS by the European Union and ESA).

Cloud and cloud-shadow segmentation on Sentinel-2 L1C top-of-atmosphere reflectance, 13 bands
at 10 m, four classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow. Please cite
Aybar et al., "CloudSEN12+: The largest dataset of expert-labeled pixels for cloud and cloud
shadow detection in Sentinel-2", Data in Brief 2024, doi:10.1016/j.dib.2024.110852.

Sources (both under ``raw/cloudsen12plus_high``):

* ``isp-uv-es/cloudsen12plus-l1c-high-taco`` (TACO v3): two stored (uncompressed) zips whose
  ``METADATA/{sample,children}.parquet`` give each GeoTIFF's byte offset and size, so chips are
  read straight out of the zip. DN = 10000 x TOA reflectance (the baseline-04.00 offset is
  already removed); 65535 = no data. 10,000 samples are 512x512 (509x509 valid + 3 padding
  rows/columns) and 847 are 2048x2048 (2000x2000 valid + 48 padding).
* ``tacofoundation/cloudsen12`` ``cloudsen12-extra.*.part.taco`` (TACO v1 "tortilla" blobs):
  per-sample auxiliary rasters on the same grid. Only the high samples' blobs are downloaded
  (HTTP range requests, ``download_extra``). We use **only** ``vv``/``vh`` and ``elevation``:

  - ``vv``/``vh`` come from the **Sentinel-1 Global Backscatter Model (S1GBM)**, a *static*
    multi-year normalised land backscatter mosaic. It is **not** a Sentinel-1 acquisition
    coincident with the S2 image and cannot see the clouds; it only describes the surface
    (stored as dB x 10; converted to dB).
  - ``elevation`` is the Copernicus DEM GLO-30 (metres), stored relative to its per-chip
    median (relief), as for Sen1Floods11.
  - The ``cloudmask_*`` layers (QA60, Sen2Cor, s2cloudless, Cloud Score+, UNetMobV2 v1/v2,
    SEnSeI v2) are other cloud detectors' outputs and are **never** read: UNetMobV2 was
    trained on CloudSEN12 itself, so using it would leak labels. ``lc10`` is not used either.

Preprocessing: S2 reflectance = DN / 1e4 (float16), 65535 -> NaN per band. Padding pixels
(all 13 bands 0) become NaN in every input and **ignore (255)** in the mask: the release labels
them "clear". 2048x2048 samples are cut into 16 non-overlapping 512x512 tiles named
``<sample_id>__t<row>_<col>`` that inherit the parent's split and ROI. Per-channel mean/std
come from the training split only (``meta.json``).

Splits: the official ones (train 9,175 / validation 612 / test 1,060 samples). They are
**ROI-disjoint** (2,847 ROIs, none crosses a split; ``prepare`` re-checks this), i.e.
geographic, so conditioning on latitude/longitude is legitimate to evaluate here: test ROIs
are unseen locations. Label-derived columns of the sample table (``*_percentage``,
``cloud_frac``) and the extra index's socioeconomic ``rai:*`` columns are never inputs.

Variants (one shared cache, ``cloudsen12plus_high``): ``cloudsen12`` (S2 only),
``cloudsen12_s2_aux`` (dict inputs ``s2`` + ``s1gbm`` + ``dem``) and ``cloudsen12_meta``
(S2 + a metadata vector: sample-centre lat/lon, day of year, solar zenith and azimuth).
"""

from __future__ import annotations

import io
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

from .base import DatasetInfo

ZIPS = ("cloudsen12plus-l1c-high.0000.zip", "cloudsen12plus-l1c-high.0001.zip")
EXTRA_URL = "https://huggingface.co/datasets/tacofoundation/cloudsen12/resolve/main/"
SPLIT_NAMES = {"train": "train", "validation": "val", "val": "val", "test": "test"}
EXPECTED = {"train": 9175, "val": 612, "test": 1060}
TILE = 512
NODATA_DN = 65535
IGNORE = 255
INPUTS = {"s2": 13}
ALL_INPUTS = {"s2": 13, "s1gbm": 2, "dem": 1}
AUX_LAYERS = ("vv", "vh", "elevation")  # the only extra layers ever read
FORBIDDEN_PREFIX = "cloudmask_"
# Columns of the sample/extra tables that must never become model inputs.
LABEL_DERIVED = ("ml:thick_percentage", "ml:thin_percentage", "ml:cloud_shadow_percentage",
                 "ml:clear_percentage", "ml:cloud_frac", "ml:labelled_frac", "thick_percentage",
                 "thin_percentage", "cloud_shadow_percentage", "clear_percentage")

SENTINEL_NOTICE = "Contains modified Copernicus Sentinel data 2017-2022"
LICENSE = (f"CC0-1.0 labels/curation (cite Aybar et al. 2024, doi:10.1016/j.dib.2024.110852); "
           f"imagery: {SENTINEL_NOTICE}")
LICENSE_AUX = (f"{LICENSE}; S1GBM: CC BY 4.0 (Bauer-Marschallinger et al. 2021); Copernicus "
               "DEM GLO-30: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH "
               "2014-2018, provided under COPERNICUS by the European Union and ESA")

INFO = DatasetInfo(
    name="cloudsen12",
    num_classes=4,
    in_channels=13,
    class_names=("clear", "thick_cloud", "thin_cloud", "cloud_shadow"),
    mean={}, std={},  # filled from meta.json
    ignore_index=IGNORE,
    license=LICENSE,
    source="https://huggingface.co/datasets/isp-uv-es/cloudsen12plus-l1c-high-taco",
    splits=("train", "val", "test"),
    test_splits=("test",),
    inputs=INPUTS,
    notes=("official splits are ROI-disjoint (geographic)",
           "2048x2048 samples tiled to 512x512; tiles inherit the parent's split",
           "padding pixels (release labels them clear) are ignored"),
)
INFO_AUX = DatasetInfo(**{
    **INFO.to_dict(), "name": "cloudsen12_s2_aux", "inputs": ALL_INPUTS,
    "in_channels": sum(ALL_INPUTS.values()), "license": LICENSE_AUX,
    "notes": INFO.notes + ("s1gbm = static S1 Global Backscatter Model, not coincident S1",)})
META_FIELDS = {"latlon": {"type": "coords"}, "doy": {"type": "cyclic", "period": 365.25},
               "sza": {"type": "continuous", "mean": 45.0, "std": 15.0},
               "saa": {"type": "cyclic", "period": 360.0}}
INFO_META = DatasetInfo(**{**INFO.to_dict(), "name": "cloudsen12_meta",
                           "vectors": {"meta": {"fields": META_FIELDS}}})
VARIANTS = (INFO_AUX, INFO_META)
TASK = "semantic"
CACHE_NAME = "cloudsen12plus_high"  # raw/ and cache/ directory shared by all variants
LOADER_KWARGS = {"device_resident": False, "vflip": True}
EVAL_BATCH_SIZE = 8


# --- names ------------------------------------------------------------------------------------

def tile_name(sample_id: str, row: int, col: int) -> str:
    return f"{sample_id}__t{row}_{col}"


def parent_of(name: str) -> str:
    """Sample id of a cached chip name (a 512x512 sample, or a tile of a 2048x2048 one)."""
    head, sep, tail = name.rpartition("__t")
    return head if sep and tail.replace("_", "").isdigit() else name


def roi_of(name: str) -> str:
    return name.split("__", 1)[0]


def tile_offsets(h: int, w: int, size: int = TILE) -> list[tuple[int, int]]:
    """Top-left (y, x) of the non-overlapping ``size`` tiles of an ``h`` x ``w`` sample."""
    if h % size or w % size:
        raise ValueError(f"sample {h}x{w} is not a multiple of the {size} tile size")
    return [(y, x) for y in range(0, h, size) for x in range(0, w, size)]


# --- TACO v3 zip (S2 + labels) ----------------------------------------------------------------

def sample_table(raw: Path) -> list[dict]:
    """One dict per sample (official parquet order) with its split, ids, acquisition metadata
    and the zip members of its ``s2`` and ``label`` GeoTIFFs (offset, size, shape)."""
    import pyarrow.parquet as pq

    raw = Path(raw)
    samples = pq.read_table(raw / "sample.parquet").to_pylist()
    children = pq.read_table(raw / "children.parquet").to_pylist()
    kids: dict = {}
    for c in children:  # ids restart in every zip: key by (zip, parent id)
        role = c["internal:relative_path"].rsplit("/", 1)[-1].removesuffix(".tif")
        kids.setdefault((c["internal:source_file"], int(c["internal:parent_id"])), {})[role] = {
            "offset": int(c["internal:offset"]), "size": int(c["internal:size"]),
            "shape": [int(v) for v in c["ml:tensor_shape"]], "crs": c["ml:crs"]}
    out = []
    for s in samples:
        k = kids.get((s["internal:source_file"], int(s["internal:current_id"])), {})
        if set(k) != {"s2", "label"}:
            raise RuntimeError(f"integrity: {s['ml:sample_id']} has children {sorted(k)}")
        out.append({"id": s["ml:sample_id"], "roi": s["ml:roi_id"],
                    "split": SPLIT_NAMES[s["ml:split"]], "zip": s["internal:source_file"],
                    "datetime": s["ml:datetime"], "sza": s["ml:solar_zenith_deg"],
                    "saa": s["ml:solar_azimuth_deg"], "s2": k["s2"], "label": k["label"]})
    return out


def read_member(fh, offset: int, size: int) -> bytes:
    fh.seek(offset)
    b = fh.read(size)
    if len(b) != size or b[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        raise RuntimeError(f"integrity: no GeoTIFF at offset {offset} (size {size})")
    return b


def decode_tif(b: bytes) -> np.ndarray:
    import tifffile

    return tifffile.imread(io.BytesIO(b))


def georef(b: bytes) -> tuple[float, float, float, float]:
    """``(x0, y0, dx, dy)`` of a north-up GeoTIFF: top-left corner and pixel size."""
    import tifffile

    with tifffile.TiffFile(io.BytesIO(b)) as t:
        tags = t.pages[0].tags
        sx, sy = tags["ModelPixelScaleTag"].value[:2]
        tie = tags["ModelTiepointTag"].value
    return float(tie[3] - tie[0] * sx), float(tie[4] + tie[1] * sy), float(sx), float(sy)


# --- TACO v1 tortilla blobs (auxiliary rasters) -------------------------------------------------

TORTILLA_MAGIC = b"#y"


def tortilla_index(blob: bytes) -> dict[str, tuple[int, int]]:
    """``{layer: (offset, length)}`` of a legacy TACO v1 tortilla blob.

    Layout: ``b"#y"``, footer offset and footer length (uint64 little-endian, from the blob
    start), then padding and the layers; the footer is a parquet table with ``tortilla:id``,
    ``tortilla:offset`` and ``tortilla:length`` (offsets relative to the blob start)."""
    import pyarrow.parquet as pq

    if len(blob) < 18 or blob[:2] != TORTILLA_MAGIC:
        raise ValueError("not a tortilla blob (bad magic)")
    off, length = struct.unpack("<QQ", blob[2:18])
    if off + length > len(blob):
        raise ValueError(f"tortilla footer [{off}, {off + length}) beyond blob ({len(blob)} B)")
    cols = ["tortilla:id", "tortilla:offset", "tortilla:length"]
    t = pq.read_table(io.BytesIO(blob[off:off + length]), columns=cols).to_pydict()
    idx = {}
    for name, o, n in zip(*(t[c] for c in cols)):
        if o + n > off:
            raise ValueError(f"tortilla layer {name!r} overlaps the footer")
        idx[name] = (int(o), int(n))
    return idx


def tortilla_layer(blob: bytes, name: str, index: dict | None = None) -> bytes:
    """GeoTIFF bytes of an allowed auxiliary layer (``AUX_LAYERS``); cloud masks are refused."""
    if name not in AUX_LAYERS or name.startswith(FORBIDDEN_PREFIX):
        raise ValueError(f"layer {name!r} is not an allowed input (allowed: {AUX_LAYERS})")
    o, n = (index or tortilla_index(blob))[name]
    return blob[o:o + n]


def extra_table(raw: Path, ids: set[str] | None = None) -> dict[str, dict]:
    """``{sample id: {"part", "offset", "length"}}`` from ``extra_index.parquet`` (the outer
    byte range of each sample's tortilla blob within its ``cloudsen12-extra`` part)."""
    import pyarrow.parquet as pq

    cols = ["tortilla:id", "tortilla:offset", "tortilla:length", "internal:subfile"]
    t = pq.read_table(Path(raw) / "extra_index.parquet", columns=cols).to_pylist()
    out = {}
    for r in t:
        if ids is not None and r["tortilla:id"] not in ids:
            continue
        part = r["internal:subfile"].rsplit("/", 1)[-1]
        out[r["tortilla:id"]] = {"part": part, "offset": int(r["tortilla:offset"]),
                                 "length": int(r["tortilla:length"])}
    return out


def extra_path(raw: Path, sample_id: str) -> Path:
    return Path(raw) / "extra" / f"{sample_id}.tortilla"


def _fetch_range(url: str, offset: int, length: int, dest: Path, retries: int = 6) -> str:
    import urllib.request

    if dest.exists() and dest.stat().st_size == length:
        return "skip"
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Range": f"bytes={offset}-{offset + length - 1}",
                "User-Agent": "manyseg-bench (cloudsen12 extra subset)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {r.status}")
                b = r.read()
            if len(b) != length or b[:2] != TORTILLA_MAGIC:
                raise RuntimeError(f"bad blob ({len(b)} B, expected {length})")
            tortilla_index(b)  # parses: footer intact
            tmp.write_bytes(b)
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001 - retried, then reported
            err = e
            time.sleep(min(60, 2 ** attempt))
    return f"failed: {err}"


def download_extra(raw: Path, workers: int = 12, base_url: str = EXTRA_URL,
                   log=print) -> dict:
    """Download the high samples' auxiliary tortilla blobs (resumable, length-checked)."""
    from concurrent.futures import ThreadPoolExecutor

    raw = Path(raw)
    ids = {s["id"] for s in sample_table(raw)}
    table = extra_table(raw, ids)
    (raw / "extra").mkdir(parents=True, exist_ok=True)
    jobs = [(base_url + e["part"], e["offset"], e["length"], extra_path(raw, i))
            for i, e in sorted(table.items())]
    counts: dict = {"missing_in_index": len(ids - set(table))}
    with ThreadPoolExecutor(workers) as ex:
        for k, status in enumerate(ex.map(lambda j: _fetch_range(*j), jobs)):
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed" or (k + 1) % 500 == 0:
                log(f"[{k + 1}/{len(jobs)}] {status} {counts}")
    return counts


# --- coordinates --------------------------------------------------------------------------------

def utm_to_latlon(epsg: int, x: float, y: float) -> tuple[float, float]:
    """WGS 84 / UTM (EPSG 326zz north, 327zz south) easting/northing -> (lat, lon) degrees
    (Krüger series, sub-millimetre within a zone; no pyproj needed)."""
    if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError(f"EPSG:{epsg} is not WGS 84 / UTM")
    zone, south = epsg % 100, epsg >= 32700
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    n = f / (2 - f)
    big_a = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    beta = (n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96,
            n ** 2 / 48 + n ** 3 / 15,
            17 * n ** 3 / 480)
    delta = (2 * n - 2 * n ** 2 / 3 - 2 * n ** 3,
             7 * n ** 2 / 3 - 8 * n ** 3 / 5,
             56 * n ** 3 / 15)
    xi = (y - (10_000_000.0 if south else 0.0)) / (k0 * big_a)
    eta = (x - 500_000.0) / (k0 * big_a)
    xi_p = xi - sum(b * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
                    for j, b in enumerate(beta, 1))
    eta_p = eta - sum(b * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
                      for j, b in enumerate(beta, 1))
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    lat = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(delta, 1))
    lon0 = math.radians(zone * 6 - 183)
    lon = lon0 + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(lat), (math.degrees(lon) + 180) % 360 - 180


# --- preparation --------------------------------------------------------------------------------

def unverified_zips(raw: Path, zips) -> list[str]:
    """Zips the download job has not verified: ``download.log`` must contain ``SHA256_OK`` and
    a ``<zip>: OK`` line (or a line naming the zip with ``SHA256_OK``) for each of them."""
    log = Path(raw) / "download.log"
    lines = log.read_text().splitlines() if log.exists() else []
    all_ok = any(ln.strip() == "SHA256_OK" for ln in lines)
    return [z for z in zips if not any(
        (z in ln and "SHA256_OK" in ln) or (all_ok and ln.strip() == f"{z}: OK")
        for ln in lines)]


def _read_aux(raw: Path, sample_id: str, ref: tuple, hw: tuple[int, int]):
    """``(3, H, W)`` float32 [vv dB, vh dB, elevation m] (NaN = missing) and a status:
    ``ok``, ``missing`` (no blob) or ``grid_mismatch`` (a layer not on the S2 grid)."""
    out = np.full((3, *hw), np.nan, np.float32)
    p = extra_path(raw, sample_id)
    if not p.exists():
        return out, "missing"
    blob = p.read_bytes()
    idx = tortilla_index(blob)
    layers = {}
    for name in AUX_LAYERS:
        b = tortilla_layer(blob, name, idx)
        a = decode_tif(b)
        if a.shape != tuple(hw) or not np.allclose(georef(b), ref, atol=1e-3):
            return out, "grid_mismatch"
        layers[name] = a
    for i, name in enumerate(("vv", "vh")):
        v = layers[name].astype(np.float32)
        out[i] = np.where(v == 0, np.nan, v / 10.0)  # S1GBM: dB x 10, 0 = no data
    out[2] = layers["elevation"].astype(np.float32)
    return out, "ok"


def _hwc(a: np.ndarray) -> np.ndarray:
    return np.moveaxis(a, 0, -1).astype(np.float16)


def read_sample(job) -> tuple[list, str]:
    """Decode one sample into ``size`` tiles: ``([(name, {input: HWC float16}, mask)], aux
    status)``."""
    raw, s, size, aux = job
    with open(Path(raw) / s["zip"], "rb") as fh:
        b2 = read_member(fh, s["s2"]["offset"], s["s2"]["size"])
        bl = read_member(fh, s["label"]["offset"], s["label"]["size"])
    dn, lab = decode_tif(b2), decode_tif(bl)
    lab = lab[0] if lab.ndim == 3 else lab
    if list(dn.shape) != s["s2"]["shape"] or [1, *lab.shape] != s["label"]["shape"] \
            or lab.shape != dn.shape[1:]:
        raise RuntimeError(f"integrity: {s['id']} shapes s2 {dn.shape}, label {lab.shape}")
    bad = set(np.unique(lab).tolist()) - {0, 1, 2, 3, IGNORE}
    if bad:
        raise RuntimeError(f"integrity: {s['id']} has label values {sorted(bad)}")
    pad = (dn == 0).all(0)  # 509x509 / 2000x2000 valid; the release labels padding "clear"
    s2 = dn.astype(np.float32) / 1e4
    s2[dn == NODATA_DN] = np.nan
    s2[:, pad] = np.nan
    lab = np.where(pad, IGNORE, lab).astype(np.uint8)
    hw = dn.shape[1:]
    del dn
    if aux:
        extra, status = _read_aux(raw, s["id"], georef(b2), hw)
        extra[:, pad] = np.nan
    else:
        extra, status = None, "skipped"
    tiles = []
    for y, x in tile_offsets(*hw, size):
        sl = (slice(y, y + size), slice(x, x + size))
        name = s["id"] if hw == (size, size) else tile_name(s["id"], y // size, x // size)
        arrays = {"s2": _hwc(s2[(slice(None), *sl)])}
        if extra is not None:
            dem = extra[2][sl]
            if np.isfinite(dem).any():
                dem = dem - np.nanmedian(dem)  # relative relief, as for Sen1Floods11
            arrays["s1gbm"] = _hwc(extra[:2][(slice(None), *sl)])
            arrays["dem"] = dem[..., None].astype(np.float16)
        tiles.append((name, arrays, lab[sl]))
    return tiles, status


def check_splits(samples: list[dict], expected: dict | None) -> dict[str, list[dict]]:
    """Group samples by split; check split sizes and that no ROI crosses a split."""
    by_split: dict = {}
    for s in samples:
        by_split.setdefault(s["split"], []).append(s)
    if expected is not None:
        got = {k: len(v) for k, v in by_split.items()}
        if got != expected:
            raise RuntimeError(f"integrity: split sizes {got}, expected {expected}")
    rois: dict = {}
    for s in samples:
        rois.setdefault(s["roi"], set()).add(s["split"])
    crossing = sorted(r for r, sp in rois.items() if len(sp) > 1)
    if crossing:
        raise RuntimeError(f"integrity: {len(crossing)} ROIs cross splits, e.g. {crossing[0]}")
    return by_split


def prepare(raw: Path, cache: Path, size: int = TILE, workers: int = 4,
            expected: dict | None = EXPECTED, require_checksums: bool = True,
            aux: bool = True, log=lambda m: print(m, flush=True)) -> dict:
    """Decode, tile and cache CloudSEN12+ high (see the module docstring); ``aux=False``
    builds an S2-only cache. Arrays are written through memory maps; RAM stays bounded."""
    import contextlib
    from concurrent.futures import ProcessPoolExecutor

    raw, cache = Path(raw), Path(cache)
    samples = sample_table(raw)
    zips = sorted({s["zip"] for s in samples})
    unverified = unverified_zips(raw, zips)
    if require_checksums and unverified:
        raise RuntimeError(f"integrity: download.log has no SHA256_OK for {unverified}")
    by_split = check_splits(samples, expected)
    inputs = ALL_INPUTS if aux else INPUTS
    cache.mkdir(parents=True, exist_ok=True)
    summary, stats = {}, {"mean": {}, "std": {}}
    with (ProcessPoolExecutor(workers) if workers > 1 else contextlib.nullcontext()) as ex:
        run = ex.map if ex is not None else map
        for split in ("train", "val", "test"):
            items = by_split.get(split, [])
            n = sum(s["s2"]["shape"][1] * s["s2"]["shape"][2] // size ** 2 for s in items)
            arrays = {k: np.lib.format.open_memmap(cache / f"{split}_images.{k}.npy", "w+",
                                                   np.float16, (n, size, size, c))
                      for k, c in inputs.items()}
            masks = np.lib.format.open_memmap(cache / f"{split}_masks.npy", "w+", np.uint8,
                                              (n, size, size))
            acc = {k: np.zeros((3, c)) for k, c in inputs.items()}  # sum, sum sq, count
            classes = np.zeros(256, np.int64)
            status: dict = {}
            names: list[str] = []
            batch = max(1, workers * 2)  # bounded number of decoded samples in flight
            for b0 in range(0, len(items), batch):
                jobs = [(raw, s, size, aux) for s in items[b0:b0 + batch]]
                for tiles, st in run(read_sample, jobs):
                    status[st] = status.get(st, 0) + 1
                    for name, arrs, m in tiles:
                        j = len(names)
                        for k, c in inputs.items():
                            arrays[k][j] = arrs[k]
                            x = arrs[k].reshape(-1, c).astype(np.float64)
                            ok = np.isfinite(x)
                            x = np.where(ok, x, 0.0)
                            acc[k] += (x.sum(0), (x * x).sum(0), ok.sum(0))
                        masks[j] = m
                        classes += np.bincount(m.ravel(), minlength=256)
                        names.append(name)
                if (b0 // batch) % 100 == 0:
                    log(f"{split}: {b0 + len(jobs)}/{len(items)} samples, "
                        f"{len(names)}/{n} tiles")
            if len(names) != n:
                raise RuntimeError(f"integrity: {split} wrote {len(names)} tiles, expected {n}")
            for a in (*arrays.values(), masks):
                a.flush()
            del arrays, masks
            (cache / f"{split}_names.json").write_text(json.dumps(names))
            if split == "train":
                for k, (s_, ss, cnt) in acc.items():
                    mean = s_ / np.maximum(cnt, 1)
                    std = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 0.0))
                    stats["mean"][k], stats["std"][k] = mean.tolist(), (std + 1e-6).tolist()
            labelled = classes[:4].sum()
            summary[split] = {
                "samples": len(items),
                "samples_2048": sum(s["s2"]["shape"][1] > size for s in items),
                "tiles": n, "rois": len({s["roi"] for s in items}),
                "class_frac": dict(zip(INFO.class_names,
                                       (classes[:4] / max(labelled, 1)).round(5).tolist())),
                "ignored_frac": float(classes[IGNORE] / max(classes.sum(), 1)),
                "aux_status": status,
                "nan_frac": {k: float(1 - (a[2] / max(n * size * size, 1)).mean())
                             for k, a in acc.items()}}
            log(f"{split}: {json.dumps(summary[split])}")
    meta = {**stats, "inputs": inputs, "tile": size, "unverified_zips": unverified,
            "summary": summary}
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    return summary


# --- metadata vectors ---------------------------------------------------------------------------

def georef_index(raw: Path, samples: list[dict]) -> dict[str, list]:
    """``{sample id: [epsg, x0, y0, dx, dy]}`` of each S2 GeoTIFF, read from the zip headers
    and cached in ``raw/s2_georef.json``."""
    path = Path(raw) / "s2_georef.json"
    idx = json.loads(path.read_text()) if path.exists() else {}
    todo = [s for s in samples if s["id"] not in idx]
    if not todo:
        return idx
    import contextlib

    with contextlib.ExitStack() as stack:
        handles: dict = {}
        for s in todo:
            if s["zip"] not in handles:
                handles[s["zip"]] = stack.enter_context(open(Path(raw) / s["zip"], "rb"))
            fh = handles[s["zip"]]
            fh.seek(s["s2"]["offset"])
            try:  # the tags sit in the first IFD; avoid reading whole 2048 chips
                ref = georef(fh.read(min(s["s2"]["size"], 1 << 16)))
            except Exception:  # noqa: BLE001
                ref = georef(read_member(fh, s["s2"]["offset"], s["s2"]["size"]))
            idx[s["id"]] = [int(s["s2"]["crs"].split(":")[1]), *ref]
    with contextlib.suppress(OSError):  # read-only raw directory: recompute next time
        path.write_text(json.dumps(idx))
    return idx


def chip_latlon(name: str, ref: list, shape, size: int | None = None) -> tuple[float, float]:
    """(lat, lon) of a cached chip's centre: the sample's centre (``shape`` = its ``[C, H,
    W]``), or for a tile of a large sample the centre of that ``size`` tile."""
    epsg, x0, y0, dx, dy = ref
    head, sep, tail = name.rpartition("__t")
    if sep and parent_of(name) == head:
        size = size or TILE
        r, c = map(int, tail.split("_"))
        px, py = (c + 0.5) * size, (r + 0.5) * size
    else:
        px, py = shape[-1] / 2, shape[-2] / 2
    return utm_to_latlon(int(epsg), x0 + px * dx, y0 - py * dy)


def _doy(iso: str) -> float:
    import datetime

    return float(datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timetuple().tm_yday)


def load_vectors(raw: Path, names: list[str], vectors: dict) -> dict[str, np.ndarray]:
    """``{vector input: (N, columns) float32}`` for cached chips ``names``. Fields: ``latlon``
    (chip centre), ``doy`` (acquisition day of year, UTC), ``sza``/``saa`` (mean solar zenith
    and azimuth, degrees). Label-derived columns (cloud percentages) are refused."""
    samples = {s["id"]: s for s in sample_table(raw)}
    refs = None
    out = {}
    for vname, cfg in vectors.items():
        fields = list(cfg["fields"])
        leaked = [f for f in fields if f in LABEL_DERIVED or "percentage" in f or "frac" in f]
        if leaked:
            raise ValueError(f"label-derived metadata fields are not allowed: {leaked}")
        if "latlon" in fields and refs is None:
            parents = dict.fromkeys(parent_of(n) for n in names)
            refs = georef_index(raw, [samples[p] for p in parents])
        rows = []
        for n in names:
            s = samples[parent_of(n)]
            row: list[float] = []
            for f in fields:
                if f == "latlon":
                    row += chip_latlon(n, refs[s["id"]], s["s2"]["shape"], TILE)
                elif f == "doy":
                    row.append(_doy(s["datetime"]))
                elif f in ("sza", "saa"):
                    row.append(float("nan") if s[f] is None else float(s[f]))
                else:
                    raise ValueError(f"no extractor for CloudSEN12 metadata field {f!r}")
            rows.append(row)
        out[vname] = np.asarray(rows, dtype=np.float32)
    return out
