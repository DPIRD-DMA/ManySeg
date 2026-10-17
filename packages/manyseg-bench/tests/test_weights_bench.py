"""Remote-sensing pretrained weights in the bench: the input is normalised the way the weights
were pretrained (not with dataset statistics), dataset bands are mapped onto the weights' band
order (missing bands filled with the pretraining mean), and clipped where pretraining clipped."""

import numpy as np
import pytest
import torch
from manyseg_bench.providers import ModelSpec, get_provider

DS_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]


def fake_hub(monkeypatch, model="resnet18", chans=13):
    import manyseg.weights as mw
    import timm

    sd = timm.create_model(model, in_chans=chans).state_dict()
    monkeypatch.setattr(mw, "_download", lambda url: {k: v.clone() for k, v in sd.items()})


def test_weights_input_adapter_maps_bands_fills_and_clips(monkeypatch):
    from manyseg.weights import get_weights

    fake_hub(monkeypatch)
    spec = ModelSpec("unet", "resnet18", weights="ssl4eo_s2_moco", in_channels=10, num_classes=2,
                     band_names=DS_BANDS)
    built = get_provider("manyseg").build(spec)
    adapter = built.model.weights_input
    w = get_weights("resnet18", "ssl4eo_s2_moco")
    x = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1).expand(1, 10, 4, 4) / 10 - 0.2
    y = adapter(x)
    assert y.shape == (1, 13, 4, 4)
    for j, b in enumerate(w.bands):
        expect = (x[0, DS_BANDS.index(b), 0, 0].clamp(0, 1) if b in DS_BANDS
                  else torch.tensor(0.0))
        assert float(y[0, j, 0, 0]) == pytest.approx(float(expect)), b
    assert set(built.notes[0].split(": ")[1].split(", ")) == {"B01", "B09", "B10"}
    assert built.model(torch.randn(1, 10, 64, 64)).shape == (1, 2, 64, 64)


def test_weights_normalisation_replaces_dataset_statistics():
    from manyseg.weights import get_weights
    from manyseg_bench.experiments.runner import weights_normalisation

    w = get_weights("resnet50", "satlas_s2_si_ms")
    mean, std = weights_normalisation(w, DS_BANDS, dn_per_unit=1e4)
    # B02 (TCI blue): DN / 2000 -> in cache units (reflectance) std 0.2; B05: 8160 -> 0.816
    assert mean[0] == 0.0 and std[0] == pytest.approx(0.2)
    assert std[DS_BANDS.index("B05")] == pytest.approx(0.816)
    assert std[DS_BANDS.index("B8A")] == 1.0  # not used by these weights: left as is


def test_registry_weights_need_band_names(monkeypatch):
    fake_hub(monkeypatch)
    with pytest.raises(ValueError, match="band"):
        get_provider("manyseg").build(ModelSpec("unet", "resnet18", weights="ssl4eo_s2_moco",
                                                in_channels=13, num_classes=2))


def test_run_training_with_registry_weights(tmp_path, monkeypatch):
    """End to end on a fake single-input S2 dataset: the run records the weights'
    normalisation as the source of its statistics."""
    from manyseg_bench import datasets
    from manyseg_bench.datasets.base import DatasetInfo, save_split
    from manyseg_bench.experiments import run_training

    fake_hub(monkeypatch)

    class Fake:
        INFO = DatasetInfo(name="fake_s2", num_classes=2, in_channels=10, class_names=("a", "b"),
                           mean={"s2": [0.1] * 10}, std={"s2": [0.05] * 10},
                           inputs={"s2": 10})
        BAND_NAMES = {"s2": tuple(DS_BANDS)}
        DN_PER_UNIT = {"s2": 1e4}

    monkeypatch.setitem(datasets.DATASETS, "fake_s2", Fake)
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        save_split(tmp_path / "cache" / "fake_s2", split,
                   {"s2": rng.random((n, 32, 32, 10)).astype(np.float16) * 0.3},
                   rng.integers(0, 2, (n, 32, 32)).astype(np.uint8), [f"{split}{i}" for i in
                                                                       range(n)])
    spec = ModelSpec("unet", "resnet18", weights="ssl4eo_s2_moco")
    rec = run_training("manyseg", spec, "fake_s2", "smoke", device="cpu", data=tmp_path,
                       out_dir=tmp_path / "runs", recipe_overrides={"batch_size": 2,
                                                                    "precision": "fp32"})
    assert rec["norm_stats"]["source"].startswith("weights resnet18.ssl4eo_s2_moco")
    assert rec["norm_stats"]["mean"]["s2"] == [0.0] * 10
    assert rec["spec"]["band_names"] == DS_BANDS
    assert rec["model_notes"] == ["bands filled with the pretraining mean: B01, B09, B10"]
