"""CloudSEN12+ high: parsers and preparation on a tiny synthetic copy of the real layout (stored
zips indexed by sample/children parquets, TACO v1 tortilla blobs for the auxiliary rasters)."""

import io
import json
import struct
import zipfile

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from manyseg_bench.datasets import cloudsen12 as cs  # noqa: E402
from manyseg_bench.datasets import get_dataset  # noqa: E402
from manyseg_bench.datasets.base import load_split  # noqa: E402

T = 32          # tile size used by the synthetic data
BIG = 4 * T     # "2048" samples
PAD = 1         # padding rows/columns ("509 of 512")
CLOUDMASK = 77  # value of every cloudmask_* layer: must never reach the cache
EPSG, X0, Y0 = 32719, 440030.0, 4573910.0


def _tif(a, x0=X0, y0=Y0, res=10.0):
    b = io.BytesIO()
    tifffile.imwrite(b, a, compression="zlib", extratags=[
        (33550, "d", 3, (res, res, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, x0, y0, 0.0))])
    return b.getvalue()


def tortilla(layers: dict[str, bytes]) -> bytes:
    """A legacy TACO v1 tortilla blob: magic, footer offset/length, padding to 200, layers,
    parquet footer."""
    body, rows, off = b"", [], 200
    for name, b in layers.items():
        rows.append({"tortilla:id": name, "tortilla:offset": off, "tortilla:length": len(b),
                     "tortilla:file_format": "GTiff"})
        body += b
        off += len(b)
    foot = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), foot)
    head = b"#y" + struct.pack("<QQQ", off, len(foot.getvalue()), 1)
    return head.ljust(200, b"\0") + body + foot.getvalue()


def _sample(i, split, roi, size):
    """Sample i: S2 DN = 1000 + 100 i in band 0 (band 1 = 65535 at one pixel), label = class
    (i % 4) with a column of 3s, padding last PAD rows/columns (DN 0, label 0 as released)."""
    dn = np.full((13, size, size), 1000 + 100 * i, np.uint16)
    dn[1, 0, 0] = 65535
    lab = np.full((size, size), i % 4, np.uint8)
    lab[:, 2] = 3
    dn[:, -PAD:], dn[:, :, -PAD:] = 0, 0
    lab[-PAD:], lab[:, -PAD:] = 0, 0
    sid = f"{roi}__2019010{i % 9 + 1}T000000_x_T19FDF"
    x0 = X0 + 100_000 * i
    elev = np.full((size, size), 100 + i, np.uint16)
    elev[0, 0] = 200 + i
    vv = np.full((size, size), -140, np.int16)
    vv[1, 1] = 0  # S1GBM no data
    layers = {"elevation": _tif(elev, x0), "lc10": _tif(np.full((size, size), 10, np.uint16), x0),
              "vv": _tif(vv, x0), "vh": _tif(np.full((size, size), -220, np.int16), x0)}
    for m in ("qa60", "sen2cor", "s2cloudless", "unetmobv2_v1", "unetmobv2_v2"):
        layers[f"cloudmask_{m}"] = _tif(np.full((size, size), CLOUDMASK, np.uint8), x0)
    return {"id": sid, "roi": roi, "split": split, "s2": _tif(dn, x0), "label": _tif(lab[None], x0),
            "size": size, "blob": tortilla(layers), "x0": x0}


# (split as released, roi, size); two zips, ids restarting per zip as in the real release
LAYOUT = [[("train", "ROI_00001", T), ("train", "ROI_00001", T), ("train", "ROI_00002", BIG),
           ("validation", "ROI_00003", T)],
          [("test", "ROI_00004", T), ("test", "ROI_00005", BIG), ("train", "ROI_00006", T)]]


def make_raw(root, extra_missing=("ROI_00006",), log="ok"):
    raw = root / "raw" / cs.CACHE_NAME
    (raw / "extra").mkdir(parents=True)
    srows, crows, made, i = [], [], [], 0
    for zi, group in enumerate(LAYOUT):
        zname = cs.ZIPS[zi]
        with zipfile.ZipFile(raw / zname, "w", zipfile.ZIP_STORED) as z:
            z.writestr("__cozip__", b"\0" * 128)
            for local, (split, roi, size) in enumerate(group):
                s = _sample(i, split, roi, size)
                made.append(s)
                z.writestr(f"DATA/{local}/s2.tif", s["s2"])
                z.writestr(f"DATA/{local}/label.tif", s["label"])
                srows.append({"internal:current_id": local, "ml:split": split,
                              "ml:sample_id": s["id"], "ml:roi_id": roi,
                              "ml:datetime": f"2019-03-0{i + 1}T14:20:31Z",
                              "ml:thick_percentage": 50.0, "ml:cloud_frac": 0.5,
                              "ml:solar_zenith_deg": 30.0 + i, "ml:solar_azimuth_deg": 100.0 + i,
                              "internal:source_file": zname})
                if roi not in extra_missing:
                    (raw / "extra" / f"{s['id']}.tortilla").write_bytes(s["blob"])
                i += 1
        with zipfile.ZipFile(raw / zname) as z, open(raw / zname, "rb") as fh:
            for info in z.infolist()[1:]:
                fh.seek(info.header_offset + 26)
                n, m = struct.unpack("<HH", fh.read(4))
                local, role = info.filename.split("/")[1:]
                size = group[int(local)][2]
                crows.append({"internal:parent_id": int(local), "internal:relative_path":
                              f"{local}/{role}", "internal:offset": info.header_offset + 30 + n + m,
                              "internal:size": info.file_size, "ml:crs": f"EPSG:{EPSG}",
                              "ml:tensor_shape": [13 if role == "s2.tif" else 1, size, size],
                              "ml:role": "input" if role == "s2.tif" else "target",
                              "internal:source_file": zname})
    pq.write_table(pa.Table.from_pylist(srows), raw / "sample.parquet")
    pq.write_table(pa.Table.from_pylist(crows), raw / "children.parquet")
    lines = [f"DOWNLOADED {raw / z}" for z in cs.ZIPS]
    if log == "ok":
        lines += [f"{z}: OK" for z in cs.ZIPS] + ["SHA256_OK"]
    (raw / "download.log").write_text("\n".join(lines) + "\n")
    return raw, made


EXPECTED = {"train": 4, "val": 1, "test": 2}


@pytest.fixture
def raw(tmp_path):
    return make_raw(tmp_path)


def _prepare(raw, tmp_path, **kw):
    return cs.prepare(raw, tmp_path / "cache" / cs.CACHE_NAME, size=T, workers=1,
                      expected=EXPECTED, log=lambda m: None, **kw)


# --- parsers -----------------------------------------------------------------------------------

def test_tortilla_index_and_layer_guard():
    blob = tortilla({"elevation": b"II*\0aa", "vv": b"II*\0b", "cloudmask_qa60": b"c"})
    idx = cs.tortilla_index(blob)
    assert list(idx) == ["elevation", "vv", "cloudmask_qa60"]
    assert cs.tortilla_layer(blob, "vv", idx) == b"II*\0b"
    with pytest.raises(ValueError, match="not an allowed input"):
        cs.tortilla_layer(blob, "cloudmask_qa60", idx)
    with pytest.raises(ValueError, match="not an allowed input"):
        cs.tortilla_layer(blob, "lc10", idx)
    with pytest.raises(ValueError, match="magic"):
        cs.tortilla_index(b"PK" + blob[2:])
    with pytest.raises(ValueError, match="footer"):
        cs.tortilla_index(blob[:-10])


def test_sample_table_joins_children_per_zip(raw):
    raw, made = raw
    rows = cs.sample_table(raw)
    assert [r["id"] for r in rows] == [s["id"] for s in made]
    assert [r["split"] for r in rows] == ["train", "train", "train", "val", "test", "test", "train"]
    with open(raw / rows[4]["zip"], "rb") as fh:  # second zip, local id 0
        b = cs.read_member(fh, rows[4]["s2"]["offset"], rows[4]["s2"]["size"])
    assert b == made[4]["s2"]
    assert cs.georef(b) == (made[4]["x0"], Y0, 10.0, 10.0)


def test_utm_inverse_matches_reference_point():
    # A real sample: EPSG:32719, 512 px from (440030, 4573910); its TACO centroid.
    lat, lon = cs.utm_to_latlon(32719, 440030 + 2560, 4573910 - 2560)
    assert lat == pytest.approx(-49.008072, abs=1e-6) and lon == pytest.approx(-69.785036, abs=1e-6)
    lat, lon = cs.utm_to_latlon(32650, 500000, 0)  # equator on the central meridian (117 E)
    assert lat == pytest.approx(0, abs=1e-9) and lon == pytest.approx(117, abs=1e-9)
    with pytest.raises(ValueError):
        cs.utm_to_latlon(3857, 0, 0)


def test_names_and_tiling():
    assert cs.tile_offsets(64, 64, 32) == [(0, 0), (0, 32), (32, 0), (32, 32)]
    with pytest.raises(ValueError):
        cs.tile_offsets(48, 64, 32)
    n = cs.tile_name("ROI_00001__2019_x_T19FDF", 2, 3)
    assert cs.parent_of(n) == "ROI_00001__2019_x_T19FDF" and cs.roi_of(n) == "ROI_00001"
    assert cs.parent_of("ROI_00001__2019_x_T19FDF") == "ROI_00001__2019_x_T19FDF"


# --- integrity ---------------------------------------------------------------------------------

def test_checksum_gate(tmp_path):
    raw, _ = make_raw(tmp_path, log="downloaded only")
    with pytest.raises(RuntimeError, match="SHA256_OK"):
        _prepare(raw, tmp_path)


def test_split_counts_and_roi_crossing():
    s = [{"split": "train", "roi": "A"}, {"split": "test", "roi": "A"}]
    with pytest.raises(RuntimeError, match="cross"):
        cs.check_splits(s, None)
    with pytest.raises(RuntimeError, match="split sizes"):
        cs.check_splits(s[:1], {"train": 2})


# --- preparation ---------------------------------------------------------------------------------

def test_prepare_tiles_masks_padding_and_aux(raw, tmp_path):
    raw, made = raw
    summary = _prepare(raw, tmp_path)
    cache = tmp_path / "cache" / cs.CACHE_NAME
    assert {k: v["tiles"] for k, v in summary.items()} == {"train": 19, "val": 1, "test": 17}
    assert summary["train"]["samples_2048"] == 1 and summary["test"]["aux_status"] == {"ok": 2}
    assert summary["train"]["aux_status"] == {"ok": 3, "missing": 1}
    tr = load_split(cache, "train")
    assert set(tr.images) == {"s2", "s1gbm", "dem"}
    assert tr.images["s2"].shape == (19, T, T, 13) and tr.images["s2"].dtype == np.float16
    assert tr.names[:2] == [made[0]["id"], made[1]["id"]]
    assert tr.names[2:18] == [cs.tile_name(made[2]["id"], r, c) for r in range(4) for c in range(4)]
    # reflectance = DN / 1e4; 65535 -> NaN; padding -> NaN and ignored (released as "clear")
    s2, m = np.asarray(tr.images["s2"][0], np.float32), np.asarray(tr.masks[0])
    assert s2[0, 0, 0] == pytest.approx(0.1, abs=1e-4) and np.isnan(s2[0, 0, 1])
    assert np.isnan(s2[-1]).all() and np.isnan(s2[:, -1]).all()
    assert (m[-1] == 255).all() and (m[:, -1] == 255).all() and m[0, 0] == 0 and m[0, 2] == 3
    # big sample tiles: only the bottom/right tiles carry padding
    tiles = np.asarray(tr.masks[2:18]).reshape(4, 4, T, T)
    assert (tiles[:3, :3] != 255).all() and (tiles[3, :, -1] == 255).all()
    np.testing.assert_array_equal(np.asarray(tr.masks[1])[:-1, :-1] % 4 >= 0, True)
    # aux: S1GBM dB = value / 10 (0 -> NaN); DEM relative to the tile median
    aux = np.asarray(tr.images["s1gbm"][0], np.float32)
    assert aux[0, 0, 0] == pytest.approx(-14.0) and aux[0, 0, 1] == pytest.approx(-22.0)
    assert np.isnan(aux[1, 1, 0])
    dem = np.asarray(tr.images["dem"][0], np.float32)
    assert dem[0, 0, 0] == 100 and dem[5, 5, 0] == 0
    assert np.isnan(np.asarray(tr.images["s1gbm"][-1], np.float32)).all()  # blob missing
    meta = json.loads((cache / "meta.json").read_text())
    assert set(meta["mean"]) == {"s2", "s1gbm", "dem"} and len(meta["mean"]["s2"]) == 13
    assert all(np.isfinite(meta["mean"]["s2"])) and meta["unverified_zips"] == []
    names = json.loads((cache / "train_names.json").read_text())
    rois = {s: {cs.roi_of(n) for n in json.loads((cache / f"{s}_names.json").read_text())}
            for s in ("train", "val", "test")}
    assert not (rois["train"] & rois["val"] or rois["train"] & rois["test"]
                or rois["val"] & rois["test"])
    assert len(names) == len(set(names))


def test_stats_from_train_only(raw, tmp_path):
    raw, made = raw
    _prepare(raw, tmp_path)
    meta = json.loads((tmp_path / "cache" / cs.CACHE_NAME / "meta.json").read_text())
    tr = load_split(tmp_path / "cache" / cs.CACHE_NAME, "train")
    b0 = np.asarray(tr.images["s2"][..., 0], np.float64)
    assert meta["mean"]["s2"][0] == pytest.approx(np.nanmean(b0), rel=1e-4)


def test_no_cloudmask_layers_reach_the_cache(raw, tmp_path):
    raw, _ = raw
    _prepare(raw, tmp_path)
    cache = tmp_path / "cache" / cs.CACHE_NAME
    files = sorted(p.name for p in cache.glob("*_images.*.npy"))
    assert {f.split(".")[1] for f in files} == {"s2", "s1gbm", "dem"}
    for f in files:
        a = np.load(cache / f, mmap_mode="r").astype(np.float32)
        assert not np.isclose(a, CLOUDMASK).any(), f
        assert not np.isclose(a, CLOUDMASK / 10).any(), f
    assert "cloudmask" not in json.dumps(cs.ALL_INPUTS) and \
        not any(n.startswith(cs.FORBIDDEN_PREFIX) for n in cs.AUX_LAYERS)


def test_s2_only_cache(raw, tmp_path):
    raw, _ = raw
    summary = _prepare(raw, tmp_path, aux=False)
    assert summary["train"]["aux_status"] == {"skipped": 4}
    assert set(load_split(tmp_path / "cache" / cs.CACHE_NAME, "val").images) == {"s2"}


# --- metadata ----------------------------------------------------------------------------------

def test_load_vectors(raw, monkeypatch):
    raw, made = raw
    monkeypatch.setattr(cs, "TILE", T)
    names = [made[0]["id"], cs.tile_name(made[2]["id"], 0, 0), cs.tile_name(made[2]["id"], 3, 1)]
    v = cs.load_vectors(raw, names, {"meta": {"fields": cs.META_FIELDS}})["meta"]
    assert v.shape == (3, 5) and v.dtype == np.float32
    exp0 = cs.utm_to_latlon(EPSG, made[0]["x0"] + T / 2 * 10, Y0 - T / 2 * 10)
    np.testing.assert_allclose(v[0, :2], exp0, atol=1e-5)
    exp2 = cs.utm_to_latlon(EPSG, made[2]["x0"] + 1.5 * T * 10, Y0 - 3.5 * T * 10)
    np.testing.assert_allclose(v[2, :2], exp2, atol=1e-5)
    assert v[0, 2] == 31 + 28 + 1 and v[1, 2] == 31 + 28 + 3  # 2019-03-01, 2019-03-03
    assert v[0, 3] == 30 and v[0, 4] == 100 and v[1, 3] == 32
    assert (raw / "s2_georef.json").exists()
    with pytest.raises(ValueError, match="label-derived"):
        cs.load_vectors(raw, names, {"meta": {"fields": {"ml:cloud_frac": {}}}})
    with pytest.raises(ValueError, match="label-derived"):
        cs.load_vectors(raw, names, {"meta": {"fields": {"thick_percentage": {}}}})


def test_variants_share_cache_and_declare_no_leaky_vectors():
    names = ("cloudsen12", "cloudsen12_s2_aux", "cloudsen12_meta")
    dss = [get_dataset(n) for n in names]
    assert {getattr(d, "CACHE_NAME", None) for d in dss} == {cs.CACHE_NAME}
    assert dss[0].INFO.inputs == {"s2": 13} and dss[1].INFO.inputs == cs.ALL_INPUTS
    assert dss[2].INFO.inputs == {"s2": 13}
    assert set(dss[2].INFO.vectors["meta"]["fields"]) == {"latlon", "doy", "sza", "saa"}
    for d in dss:
        assert d.INFO.num_classes == 4 and d.INFO.ignore_index == 255 and d.TASK == "semantic"
        assert d.LOADER_KWARGS["device_resident"] is False and d.LOADER_KWARGS["vflip"]
        assert "CC0-1.0" in d.INFO.license
        assert "Contains modified Copernicus Sentinel data" in d.INFO.license
    assert "S1GBM" in dss[1].INFO.license and "DEM" in dss[1].INFO.license


# --- end to end ----------------------------------------------------------------------------------

@pytest.mark.parametrize("dataset,mode", [("cloudsen12", "stack"),
                                          ("cloudsen12_s2_aux", "dict"),
                                          ("cloudsen12_meta", "dict")])
def test_cli_trains_end_to_end(tmp_path, capsys, monkeypatch, dataset, mode):
    from manyseg_bench.cli import main

    monkeypatch.setattr(cs, "TILE", T)
    raw, _ = make_raw(tmp_path)
    _prepare(raw, tmp_path)
    rc = main(["--data", str(tmp_path), "train", "--provider", "manyseg", "--encoder",
               "resnet18", "--weights", "none", "--dataset", dataset, "--recipe", "smoke",
               "--input-mode", mode, "--set", '{"precision": "fp32", "batch_size": 2}',
               "--device", "cpu", "--out", str(tmp_path / "runs"), "--evaluate-test"])
    assert rc == 0
    assert "final val mIoU" in capsys.readouterr().out
    rec = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert rec["dataset"]["name"] == dataset and rec["split_sizes"]["train"] == 19
    assert set(rec["norm_stats"]["mean"]) == set(get_dataset(dataset).INFO.inputs)
    assert "Copernicus Sentinel" in rec["dataset"]["license"]  # provenance in every record
    assert rec["vectors_dropped"] is False and \
        bool(rec["spec"]["vectors"]) == (dataset == "cloudsen12_meta")
