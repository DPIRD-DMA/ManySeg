"""Metadata (vector) inputs in the bench: loader, provider, trainer, Sen1Floods11 metadata and
leave-events-out splits."""

import json

import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader, get_dataset, sen1floods11
from manyseg_bench.datasets.base import save_split
from manyseg_bench.providers import ModelSpec, get_provider

MEAN = {"a": [0, 0], "b": [0, 0, 0]}
STD = {"a": [1, 1], "b": [1, 1, 1]}
FIELDS = {"doy": {"type": "cyclic", "period": 365.25}, "orbit": {"type": "categorical", "num": 2}}


def data(n=6, h=32, w=32):
    """Sample i has image channel a[0] == i everywhere and vector row (i, i % 2)."""
    a = np.zeros((n, h, w, 2), np.float16)
    a[..., 0] = np.arange(n)[:, None, None]
    b = np.zeros((n, h, w, 3), np.float16)
    masks = np.zeros((n, h, w), np.uint8)
    vec = np.stack([np.arange(n), np.arange(n) % 2], axis=1).astype(np.float32)
    return {"a": a, "b": b}, masks, {"meta": vec}


def loader(stack=False, train=False, n=6, **kw):
    imgs, masks, vecs = data(n)
    return GPULoader(imgs, masks, 3, MEAN, STD, torch.device("cpu"), train, seed=1,
                     stack=stack, vectors=vecs, **kw)


def test_dict_mode_yields_vectors_in_order():
    rows = [x["meta"] for x, _ in loader()]
    torch.testing.assert_close(torch.cat(rows)[:, 0], torch.arange(6.0))


def test_vectors_follow_their_samples_when_shuffled_and_augmented():
    for x, _ in loader(train=True):
        ids = x["a"][:, 0].mean(dim=(-2, -1))  # constant per sample: survives the crop
        torch.testing.assert_close(ids, x["meta"][:, 0], atol=1e-3, rtol=0)


def test_stack_mode_drops_vectors():
    x, _ = next(iter(loader(stack=True)))
    assert isinstance(x, torch.Tensor) and x.shape[1] == 5


def test_vector_rows_must_match_samples():
    imgs, masks, _ = data()
    with pytest.raises(ValueError, match="rows"):
        GPULoader(imgs, masks, 3, MEAN, STD, torch.device("cpu"), False, stack=False,
                  vectors={"meta": np.zeros((5, 2), np.float32)})


def _spec(**opts):
    return ModelSpec("unet", "resnet18", weights=None, num_classes=2, in_channels=5,
                     inputs={"a": 2, "b": 3}, vectors={"meta": {"fields": FIELDS}},
                     input_mode="dict", options=opts)


@pytest.mark.parametrize("mode", ["film", "channels"])
def test_manyseg_provider_builds_conditioned_model(mode):
    built = get_provider("manyseg").build(
        _spec(input_options={"meta": {"conditioning": mode}}))
    model = built.model.eval()
    assert model.conditioner.modes == {"meta": mode}
    x, _ = next(iter(loader()))
    assert model(x).shape == (3, 2, 32, 32)
    conditioner = {id(p) for p in model.conditioner.parameters()}
    assert conditioner <= {id(p) for p in built.decoder_params}  # trained at the decoder LR


@pytest.mark.parametrize("channels_last", [False, True])
def test_trainer_runs_with_vector_inputs(channels_last):
    from manyseg_bench.training import get_recipe, train

    b = get_provider("manyseg").build(_spec())
    rec = get_recipe("standard", epochs=1, batch_size=3, precision="fp32", warmup_epochs=0,
                     channels_last=channels_last)
    res = train(b.model, b.encoder_params, b.decoder_params, loader(train=True), loader(),
                num_classes=2, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
    assert 0 <= res["final_val"]["miou"] <= 1


# --- Sen1Floods11 metadata ------------------------------------------------------------------

EVENTS = {"Ghana": ("2018/09/19", "ASCENDING", (9.0, -1.0)),
          "India": ("2016/08/12", "DESCENDING", (26.0, 93.0)),
          "Cambodia": ("2018/08/04", "ASCENDING", (12.5, 105.8)),
          "Spain": ("2019/09/18", "DESCENDING", (38.1, -0.8)),
          "USA": ("2019/05/22", "ASCENDING", (39.0, -95.0))}
CHIPS = ["Ghana_1", "Ghana_2", "India_1", "Mekong_1", "Mekong_2", "Spain_1", "USA_1", "USA_2",
         "Peru_1"]  # Mekong is the catalogue's Cambodia; Peru is not in the catalogue


def fake_raw(raw, size=8, scale=0.001):
    import tifffile

    root = raw / "sen1floods11_v1.1"
    (root / "data" / "S2L1CHand").mkdir(parents=True)
    feats = [{"type": "Feature", "geometry": None, "properties": {
        "location": e, "s2_date": d, "s1_date": d, "orbit": o}} for e, (d, o, _) in EVENTS.items()]
    (root / "Sen1Floods11_Metadata.geojson").write_text(json.dumps({"features": feats}))
    centres = {}
    for i, n in enumerate(CHIPS):
        ev = sen1floods11.EVENT_ALIASES.get(n.split("_")[0], n.split("_")[0])
        lat, lon = EVENTS.get(ev, ("", "", (0.0, 0.0)))[2]
        lat, lon = lat + 0.01 * i, lon - 0.01 * i
        tie = (0.0, 0.0, 0.0, lon - size / 2 * scale, lat + size / 2 * scale, 0.0)
        tifffile.imwrite(root / "data" / "S2L1CHand" / f"{n}_S2Hand.tif",
                         np.zeros((13, size, size), np.int16), extratags=[
                             (33550, "d", 3, (scale, scale, 0.0)), (33922, "d", 6, tie)])
        centres[n] = (lat, lon)
    return raw, centres


def test_event_of_and_aliases():
    assert sen1floods11.event_of("Sri-Lanka_120804") == "Sri-Lanka"
    assert sen1floods11.EVENT_ALIASES["Mekong"] == "Cambodia"


def test_load_vectors_from_georeferencing_and_catalogue(tmp_path):
    raw, centres = fake_raw(tmp_path / "raw")
    fields = {"latlon": {"type": "coords"}, **FIELDS}
    v = sen1floods11.load_vectors(raw, CHIPS, {"meta": {"fields": fields}})["meta"]
    assert v.shape == (len(CHIPS), 4) and v.dtype == np.float32
    for i, n in enumerate(CHIPS):
        np.testing.assert_allclose(v[i, :2], centres[n], atol=1e-4)
    assert v[0, 2] == 262 and v[0, 3] == 0      # Ghana: 2018-09-19, ascending
    assert v[2, 2] == 225 and v[2, 3] == 1      # India: 2016-08-12, descending
    assert v[3, 2] == 216                        # Mekong -> Cambodia, 2018-08-04
    assert np.isnan(v[-1, 2:]).all()             # unknown event -> missing


def test_event_folds_never_share_events():
    names = [f"{e}_{i}" for e in ("A", "B", "C", "D", "E", "F") for i in range(3)]
    tested = []
    for fold in range(3):
        part = sen1floods11.event_folds(names, k=3, fold=fold)
        assert sorted(sum(part.values(), [])) == list(range(len(names)))
        ev = {k: {sen1floods11.event_of(names[i]) for i in v} for k, v in part.items()}
        assert not (ev["train"] & ev["val"] or ev["train"] & ev["test"] or ev["val"] & ev["test"])
        assert ev["train"] and ev["val"] and ev["test"]
        tested += sorted(ev["test"])
    assert sorted(tested) == ["A", "B", "C", "D", "E", "F"]  # every event tested exactly once
    with pytest.raises(ValueError, match="fold"):
        sen1floods11.event_folds(names, k=3, fold=3)


def test_split_getter_events_scheme_with_vectors(tmp_path):
    from manyseg_bench.experiments.runner import split_getter

    raw, _ = fake_raw(tmp_path / "raw")
    cache = tmp_path / "cache"
    groups = {"train": CHIPS[:4], "val": CHIPS[4:6], "test": CHIPS[6:], "bolivia": CHIPS[:1]}
    for split, names in groups.items():
        n = len(names)
        save_split(cache, split, {"s1": np.zeros((n, 4, 4, 2), np.float16),
                                  "s2": np.full((n, 4, 4, 13), CHIPS.index(names[0]), np.float16),
                                  "dem": np.zeros((n, 4, 4, 1), np.float16)},
                   np.zeros((n, 4, 4), np.uint8), names)
    ds = get_dataset("sen1floods11_meta")
    get = split_getter(ds, cache, raw, "events:3:0")
    parts = {s: get(s) for s in ("train", "val", "test")}
    events = {s: {sen1floods11.event_of(n) for n in p.names} for s, p in parts.items()}
    assert sum(len(p.names) for p in parts.values()) == len(CHIPS)
    assert not (events["train"] & events["test"]) and not (events["train"] & events["val"])
    for p in parts.values():
        assert set(p.images) == {"s1", "s2"}  # dataset variant's inputs only
        assert p.vectors["meta"].shape == (len(p.names), 4)  # latlon (2) + doy + orbit
    assert get("bolivia").names == CHIPS[:1]
    assert get_dataset("sen1floods11_meta_nogeo").INFO.vectors["meta"]["fields"].keys() == \
        {"doy", "orbit"}
    with pytest.raises(ValueError, match="scheme"):
        split_getter(ds, cache, raw, "random")


def fake_dataset(tmp_path):
    """Synthetic Sen1Floods11 cache + raw catalogue. Chip i's S2 values are ~ 10 * i, so
    normalisation statistics reveal which chips they were computed from."""
    fake_raw(tmp_path / "data" / "raw" / "sen1floods11")
    cache = tmp_path / "data" / "cache" / "sen1floods11"
    groups = {"train": CHIPS[:4], "val": CHIPS[4:6], "test": CHIPS[6:], "bolivia": CHIPS[:2]}
    rng = np.random.default_rng(0)
    for split, names in groups.items():
        n = len(names)
        s2 = rng.normal(size=(n, 32, 32, 13)) + 10 * np.array(
            [CHIPS.index(c) for c in names])[:, None, None, None]
        save_split(cache, split, {"s1": rng.normal(size=(n, 32, 32, 2)).astype(np.float16),
                                  "s2": s2.astype(np.float16),
                                  "dem": np.zeros((n, 32, 32, 1), np.float16)},
                   rng.integers(0, 2, (n, 32, 32)).astype(np.uint8), names)
    (cache / "meta.json").write_text(json.dumps({
        "mean": {"s1": [0] * 2, "s2": [0] * 13, "dem": [0]},
        "std": {"s1": [1] * 2, "s2": [1] * 13, "dem": [1]}}))
    return tmp_path / "data", cache


def _run(tmp_path, data, mode="dict", split="events:3:0"):
    from manyseg_bench.experiments import run_training

    spec = ModelSpec("unet", "resnet18", weights=None, input_mode=mode)
    return run_training("manyseg", spec, "sen1floods11_meta", "smoke", device="cpu",
                        out_dir=tmp_path / "runs", data=data, evaluate_test=True,
                        recipe_overrides={"batch_size": 2, "precision": "fp32"}, split=split)


@pytest.mark.parametrize("mode", ["dict", "stack"])
def test_run_training_end_to_end_with_metadata(tmp_path, mode):
    """CLI-equivalent run on a synthetic Sen1Floods11 cache + raw catalogue (CPU)."""
    data, _ = fake_dataset(tmp_path)
    rec = _run(tmp_path, data, mode)
    assert rec["split_scheme"] == "events:3:0" and rec["vectors_dropped"] == (mode == "stack")
    assert {"test", "bolivia"} <= set(rec)
    assert rec["spec"]["vectors"] == {"meta": {"fields": sen1floods11.META_FIELDS}}


def test_channel_stats_match_numpy_with_nans_and_chunks():
    from manyseg_bench.datasets.base import channel_stats

    rng = np.random.default_rng(0)
    a = (rng.normal(size=(7, 5, 6, 3)) * [1, 2, 3] + [0, 5, -2]).astype(np.float16)
    a[0, 0, 0, 1] = np.nan
    mean, std = channel_stats(a, chunk=2)
    flat = a.reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(mean, np.nanmean(flat, 0), rtol=1e-6)
    np.testing.assert_allclose(std, np.nanstd(flat, 0) + 1e-6, rtol=1e-5)
    u8 = rng.integers(0, 256, (4, 3, 3, 2)).astype(np.uint8)
    m8, _ = channel_stats(u8)
    np.testing.assert_allclose(m8, u8.reshape(-1, 2).mean(0) / 255, rtol=1e-6)  # loader units


def test_event_split_normalises_with_training_events_only(tmp_path):
    """Regression: event folds pool the official splits, so meta.json statistics (official
    train) would leak held-out events; stats must come from the fold's training chips."""
    from manyseg_bench.experiments.runner import split_getter

    data, cache = fake_dataset(tmp_path)
    rec = _run(tmp_path, data)
    train = split_getter(get_dataset("sen1floods11_meta"), cache, data / "raw" / "sen1floods11",
                         "events:3:0")("train")
    expected = train.images["s2"].astype(np.float64).reshape(-1, 13).mean(0)
    norm = rec["norm_stats"]
    assert norm["source"] == "train split of events:3:0"
    np.testing.assert_allclose(norm["mean"]["s2"], expected, rtol=1e-4)
    assert set(norm["mean"]) == {"s1", "s2"}


def test_official_split_keeps_meta_json_statistics(tmp_path):
    data, _ = fake_dataset(tmp_path)
    norm = _run(tmp_path, data, split="official")["norm_stats"]
    assert norm["source"] == "meta.json (official train split)"
    assert norm["mean"]["s2"] == [0] * 13 and norm["std"]["s2"] == [1] * 13


@pytest.mark.parametrize("shuffle", [False, True])
def test_oracle_metadata_learned_through_the_bench_training_path(shuffle):
    """Decisive conditioning check (Codex diagnosis of E009/E013/E016): noise images whose
    labels are decided only by a metadata bit, trained with the real `fastai_like` recipe
    (frozen first epoch, 0.1x encoder LR, weight decay) via GPULoader + train(). The real bit
    must be learned; a shuffled bit (the controls' path) must stay at chance."""
    from manyseg_bench.training import get_recipe, train

    rng = np.random.default_rng(0)
    n, h, w = 48, 32, 32
    bit = rng.integers(0, 2, n)
    imgs = {"a": rng.normal(size=(n, h, w, 2)).astype(np.float16),
            "b": rng.normal(size=(n, h, w, 3)).astype(np.float16)}
    masks = np.broadcast_to(bit[:, None, None], (n, h, w)).astype(np.uint8).copy()
    shown = rng.permutation(bit) if shuffle else bit
    vec = np.stack([np.zeros(n), shown], axis=1).astype(np.float32)  # (doy unused, orbit=bit)

    def ld(train_):
        return GPULoader(imgs, masks, 8, MEAN, STD, torch.device("cpu"), train_, seed=0,
                         stack=False, vectors={"meta": vec}, scale=(1.0, 1.0))

    torch.manual_seed(0)
    b = get_provider("manyseg").build(_spec(), img_size=(h, w))
    rec = get_recipe("fastai_like", epochs=8, batch_size=8, precision="fp32")
    out = train(b.model, b.encoder_params, b.decoder_params, ld(True), ld(False),
                num_classes=2, recipe=rec, device=torch.device("cpu"), log=lambda *_: None)
    miou = out["final_val"]["miou"]
    if shuffle:
        assert miou < 0.7, miou  # labels unpredictable from a shuffled bit (chance ~0.33-0.5)
    else:
        assert miou > 0.95, miou
