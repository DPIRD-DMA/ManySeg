"""PASTIS single-date crop types: date subset, fold splits, void handling and the metadata
controls, on a tiny synthetic copy of the release layout (no download)."""

import datetime
import json
from pathlib import Path

import numpy as np
import pytest
from manyseg_bench.datasets import get_dataset
from manyseg_bench.datasets import pastis as P
from manyseg_bench.datasets.base import load_split

S = 16  # patch size of the synthetic data


def _dates(start: str, every: int, n: int) -> list[int]:
    d0 = datetime.date.fromisoformat(start)
    return [int(f"{d0 + datetime.timedelta(days=every * i):%Y%m%d}") for i in range(n)]


# Two tiles with their own date lists (as in the release), Sep 2018 .. Oct 2019.
TILES = {"t30uxv": _dates("2018-09-17", 15, 28), "t31tfj": _dates("2018-09-20", 10, 40)}
# (patch id, tile, fold): train = folds 1-3, val = 4, test = 5
PATCHES = [(10000, "t30uxv", 1), (10001, "t30uxv", 2), (10002, "t30uxv", 4),
           (10003, "t30uxv", 5), (20000, "t31tfj", 3), (20001, "t31tfj", 1),
           (20002, "t31tfj", 4), (20003, "t31tfj", 5)]
EXPECTED = {"train": 4, "val": 2, "test": 2}


def make_raw(root, patches=PATCHES, log="MD5_OK"):
    raw = root / "raw" / P.CACHE_NAME
    base = raw / "PASTIS"
    (base / "DATA_S2").mkdir(parents=True)
    (base / "ANNOTATIONS").mkdir()
    feats = []
    for pid, tile, fold in patches:
        dates = TILES[tile]
        # band b of date t = 100 * t + b + pid % 7 (DN); cloud-free values
        s2 = (np.arange(len(dates))[:, None, None, None] * 100
              + np.arange(10)[None, :, None, None] + pid % 7
              + np.zeros((1, 1, S, S))).astype(np.int16)
        np.save(base / "DATA_S2" / f"S2_{pid}.npy", s2)
        sem = np.full((S, S), pid % 4, np.uint8)  # background (0) occurs for pid % 4 == 0
        sem[:, :3] = P.VOID
        sem[0, 5] = 18
        target = np.stack([sem, np.zeros_like(sem), np.ones_like(sem)])
        np.save(base / "ANNOTATIONS" / f"TARGET_{pid}.npy", target)
        feats.append({"type": "Feature", "geometry": None, "properties": {
            "Fold": fold, "ID_PATCH": pid, "N_Parcel": 7, "Parcel_Cover": 0.5,
            "TILE": tile, "dates-S2": {str(i): d for i, d in enumerate(dates)}}})
    # the release has S2 series without metadata or annotation: must be skipped
    np.save(base / "DATA_S2" / "S2_99999.npy", np.zeros((3, 10, S, S), np.int16))
    (base / "metadata.geojson").write_text(json.dumps({"type": "FeatureCollection",
                                                       "features": feats}))
    (raw / "download.log").write_text(f"PASTIS.zip: {P.MD5}\n{log}\n")
    return raw


def _prepare(raw, tmp_path, **kw):
    return P.prepare(raw, tmp_path / "cache" / P.CACHE_NAME, size=S, workers=1,
                     expected=EXPECTED, log=lambda m: None, **kw)


# --- date subset ------------------------------------------------------------------------------

def test_select_dates_spreads_over_the_season_window():
    dates = TILES["t31tfj"]
    idx = P.select_dates(dates, 8)
    chosen = [P._date(dates[i]) for i in idx]
    assert len(idx) == len(set(idx)) == 8 and chosen == sorted(chosen)
    lo, hi = (datetime.date.fromisoformat(s) for s in P.SEASON)
    assert all(lo <= d <= hi for d in chosen)  # September 2018 is never taken
    # one acquisition per eighth of the window (dense list: every 10 days)
    parts = [(d - lo).days * 8 // ((hi - lo).days + 1) for d in chosen]
    assert parts == list(range(8))
    assert P.select_dates(dates, 8) == idx  # deterministic


def test_select_dates_sparse_and_ties():
    assert P.select_dates([20190101, 20190601], 8) == [0, 1]
    assert P.select_dates([20180915, 20191115], 8) == []  # outside the window
    # centre of the single part is 2019-05-02 (182.5 days after 2018-11-01): equidistant
    # neighbours -> the earlier one
    assert P.select_dates([20190430, 20190505], 1) == [0]
    assert len(P.select_dates(TILES["t30uxv"], 8)) == 8


def test_names_round_trip():
    n = P.sample_name(10000, "t30uxv", 20190117)
    assert n == "10000_t30uxv_20190117"
    assert P.parse_name(n) == ("10000", "t30uxv", datetime.date(2019, 1, 17))
    assert P.doy_of(n) == 17.0


# --- splits -----------------------------------------------------------------------------------

def test_fold_splits_and_crossing(tmp_path):
    raw = make_raw(tmp_path)
    by = P.check_splits(P.patch_table(raw / "PASTIS"), EXPECTED)
    assert {s: sorted(p["fold"] for p in v) for s, v in by.items()} == {
        "train": [1, 1, 2, 3], "val": [4, 4], "test": [5, 5]}
    ps = P.patch_table(raw / "PASTIS")
    with pytest.raises(RuntimeError, match="crosses splits"):
        P.check_splits(ps + [{**ps[0], "fold": 5}], None)
    with pytest.raises(RuntimeError, match="occurs twice"):
        P.check_splits(ps + [dict(ps[0])], None)
    with pytest.raises(RuntimeError, match="split sizes"):
        P.check_splits(ps, {"train": 5, "val": 2, "test": 1})
    with pytest.raises(RuntimeError, match="fold"):
        P.check_splits([{**ps[0], "fold": 6}], None)


def test_patch_table_reads_no_label_derived_properties(tmp_path):
    raw = make_raw(tmp_path)
    for p in P.patch_table(raw / "PASTIS"):
        assert set(p) == {"id", "fold", "tile", "dates"}


def test_checksum_gate(tmp_path):
    raw = make_raw(tmp_path, log="not verified")
    with pytest.raises(RuntimeError, match="MD5_OK"):
        _prepare(raw, tmp_path)


# --- preparation ------------------------------------------------------------------------------

def test_prepare_samples_masks_and_no_leakage(tmp_path):
    raw = make_raw(tmp_path)
    summary = _prepare(raw, tmp_path)
    cache = tmp_path / "cache" / P.CACHE_NAME
    patches_of = {}
    for split in ("train", "val", "test"):
        sp = load_split(cache, split)
        n = EXPECTED[split] * 8
        assert summary[split]["samples"] == len(sp.names) == n
        assert sp.images["s2"].shape == (n, S, S, 10) and sp.images["s2"].dtype == np.float16
        assert sp.masks.shape == (n, S, S)
        patches_of[split] = {P.parse_name(x)[0] for x in sp.names}
        for i, name in enumerate(sp.names):
            pid, tile, date = P.parse_name(name)
            t = [P._date(d) for d in TILES[tile]].index(date)
            want = (100 * t + np.arange(10) + int(pid) % 7) / 1e4
            np.testing.assert_allclose(sp.images["s2"][i, 3, 3].astype(np.float32), want,
                                       rtol=1e-3)
            m = np.asarray(sp.masks[i])
            assert (m[:, :3] == P.IGNORE).all()          # void -> ignore
            assert m[0, 5] == 18 and (m[1:, 3:] == int(pid) % 4).all()  # 0 stays background
        # all dates of a patch: same mask, 8 distinct dates
        assert all(sum(P.parse_name(x)[0] == p for x in sp.names) == 8
                   for p in patches_of[split])
    assert not (patches_of["train"] & patches_of["val"]) and \
        not (patches_of["train"] & patches_of["test"]) and \
        not (patches_of["val"] & patches_of["test"])
    assert summary["train"]["class_frac"]["background"] > 0
    assert 0 < summary["train"]["void_frac"] < 1
    assert summary["train"]["doy_min_max"][0] >= 1
    assert summary["unlisted_s2_series"] == 1
    assert json.loads((cache / "meta.json").read_text())["unlisted_s2_ids"] == [99999]


def test_stats_from_train_only(tmp_path):
    raw = make_raw(tmp_path)
    _prepare(raw, tmp_path)
    cache = tmp_path / "cache" / P.CACHE_NAME
    meta = json.loads((cache / "meta.json").read_text())
    tr = load_split(cache, "train").images["s2"].astype(np.float64)
    np.testing.assert_allclose(meta["mean"]["s2"], tr.reshape(-1, 10).mean(0), rtol=1e-6)
    np.testing.assert_allclose(meta["std"]["s2"], tr.reshape(-1, 10).std(0) + 1e-6, rtol=1e-5)


# --- metadata vectors -------------------------------------------------------------------------

def _names(n_per_tile=400):
    rng = np.random.default_rng(1)
    out = []
    for tile, dates in TILES.items():
        for i in range(n_per_tile):
            out.append(P.sample_name(30000 + i, tile, dates[rng.integers(len(dates))]))
    return out


def test_load_vectors_doy_and_controls():
    names = _names()
    fields = {"doy": {}, "doy_shuffled": {}, "doy_tileshuffled": {}}
    v = P.load_vectors(None, names, {"meta": {"fields": fields}})["meta"]
    doy, sh, tsh = v.T
    assert v.dtype == np.float32 and v.shape == (len(names), 3)
    np.testing.assert_array_equal(doy, [P.doy_of(n) for n in names])
    # both controls are permutations of the true column (same marginal distribution) ...
    np.testing.assert_array_equal(np.sort(sh), np.sort(doy))
    tiles = np.asarray([P.parse_name(n)[1] for n in names])
    for t in TILES:
        np.testing.assert_array_equal(np.sort(tsh[tiles == t]), np.sort(doy[tiles == t]))
    # ... and decorrelated from the image's real date (cyclic, so compare sin/cos)
    def enc(x):
        a = 2 * np.pi * x / 365.25
        return np.stack([np.sin(a), np.cos(a)], 1)
    for ctrl in (sh, tsh):
        assert (ctrl != doy).mean() > 0.8
        for k in range(2):
            assert abs(np.corrcoef(enc(doy)[:, k], enc(ctrl)[:, k])[0, 1]) < 0.15
    # deterministic (fixed seed)
    again = P.load_vectors(None, names, {"meta": {"fields": fields}})["meta"]
    np.testing.assert_array_equal(v, again)


def test_shuffled_index_stays_within_groups():
    groups = ["a", "b", "a", "c", "b", "a"]
    perm = P.shuffled_index(groups)
    assert sorted(perm) == list(range(6))
    assert all(groups[perm[i]] == groups[i] for i in range(6))


def test_label_derived_and_unknown_fields_refused():
    names = _names(2)
    for f in ("N_Parcel", "Parcel_Cover", "parcel_count"):
        with pytest.raises(ValueError, match="label-derived"):
            P.load_vectors(None, names, {"meta": {"fields": {f: {}}}})
    with pytest.raises(ValueError, match="no extractor"):
        P.load_vectors(None, names, {"meta": {"fields": {"TILE": {}}}})


def test_registry_variants():
    for name, field in (("pastis", None), ("pastis_meta", "doy"),
                        ("pastis_meta_shuffled", "doy_shuffled"),
                        ("pastis_meta_tileshuffled", "doy_tileshuffled")):
        d = get_dataset(name)
        assert d.CACHE_NAME == "pastis" and d.TASK == "semantic"
        assert d.INFO.num_classes == 19 and d.INFO.ignore_index == 255
        assert d.INFO.inputs == {"s2": 10} and d.LOADER_KWARGS["device_resident"] is False
        assert "CC BY 4.0" in d.INFO.license and "10.5281/zenodo.5012942" in d.INFO.license
        if field is None:
            assert d.INFO.vectors is None
        else:
            assert d.INFO.vectors == {"meta": {"fields": {field: {"type": "cyclic",
                                                                  "period": 365.25}}}}


# --- end to end -------------------------------------------------------------------------------

CHANNELS = '{"input_options": {"meta": {"conditioning": "channels"}}}'


@pytest.mark.parametrize("dataset,mode,options", [
    ("pastis", "stack", "{}"), ("pastis_meta", "dict", "{}"), ("pastis_meta", "dict", CHANNELS),
    ("pastis_meta_shuffled", "dict", "{}"), ("pastis_meta_tileshuffled", "dict", "{}")])
def test_cli_trains_end_to_end(tmp_path, capsys, dataset, mode, options):
    """Every E016 arm (run.sh) trains and evaluates through the CLI."""
    from manyseg_bench.cli import main

    raw = make_raw(tmp_path)
    _prepare(raw, tmp_path)
    rc = main(["--data", str(tmp_path), "train", "--provider", "manyseg", "--encoder",
               "resnet18", "--weights", "none", "--dataset", dataset, "--recipe", "smoke",
               "--input-mode", mode, "--options", options,
               "--set", '{"precision": "fp32", "batch_size": 4}',
               "--device", "cpu", "--out", str(tmp_path / "runs"), "--evaluate-test"])
    assert rc == 0
    assert "final val mIoU" in capsys.readouterr().out
    rec = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert rec["dataset"]["name"] == dataset and rec["split_sizes"]["train"] == 32
    assert rec["vectors_dropped"] is False
    assert bool(rec["spec"]["vectors"]) == dataset.startswith("pastis_meta")
    assert "test" in rec


GROUPS = ('{"band_groups": {"s2_10m": {"bands": [0, 1, 2, 6], "factor": 1}, "s2_20m": '
          '{"bands": [3, 4, 5, 7, 8, 9], "factor": 2}}, "fusion": {"type": "intermediate", '
          '"op": "concat"}, "band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}}}')


def test_band_group_indices_match_the_band_order():
    """E016 Q2: 10 m = B2, B3, B4, B8; 20 m = B5, B6, B7, B8A, B11, B12 (PASTIS order)."""
    g = json.loads(GROUPS)["band_groups"]
    assert [P.BANDS[i] for i in g["s2_10m"]["bands"]] == ["B2", "B3", "B4", "B8"]
    assert [P.BANDS[i] for i in g["s2_20m"]["bands"]] == ["B5", "B6", "B7", "B8A", "B11", "B12"]
    run_sh = Path(__file__).parents[3] / "experiments" / "E016_pastis_date_metadata" / "run.sh"
    if run_sh.exists():
        text = run_sh.read_text()
        assert '"s2_10m": {"bands": [0, 1, 2, 6], "factor": 1}' in text
        assert '"s2_20m": {"bands": [3, 4, 5, 7, 8, 9], "factor": 2}' in text


def test_cli_trains_band_groups(tmp_path, capsys):
    """E016 Q2 arm: 10 m / 20 m band groups through one shared encoder."""
    from manyseg_bench.cli import main

    raw = make_raw(tmp_path)
    _prepare(raw, tmp_path)
    rc = main(["--data", str(tmp_path), "train", "--provider", "manyseg", "--encoder",
               "resnet18", "--weights", "none", "--dataset", "pastis", "--recipe", "smoke",
               "--options", GROUPS, "--set", '{"precision": "fp32", "batch_size": 4}',
               "--device", "cpu", "--out", str(tmp_path / "runs")])
    assert rc == 0
    assert "final val mIoU" in capsys.readouterr().out
