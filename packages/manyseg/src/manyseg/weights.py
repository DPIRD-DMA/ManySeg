"""Remote-sensing pretrained weights for standard timm architectures.

``create_model(backbone="resnet50", pretrained="ssl4eo_s2_moco", in_channels=13)`` builds the
timm architecture with the weights' band count and loads the published state dict (downloaded
and cached by ``torch.hub``, hash-checked). Every model parameter must be covered by the
checkpoint; only the classifier may be left out.

Inputs must match the weights: the same bands in the same order (``Weights.bands``), scaled as
the weights expect: ``(x - mean) / std`` with ``Weights.mean`` / ``Weights.std`` in the units
noted in ``Weights.units``. Normalising with your own dataset statistics instead usually works
but discards part of what the encoder learned about absolute reflectance.

Sources (URLs and preprocessing as published by TorchGeo; licences from the authors):

* **SSL4EO-S12** (Wang et al. 2023, https://github.com/zhu-xlab/SSL4EO-S12): self-supervised
  (MoCo / DINO) on Sentinel-2 L1C, all 13 bands, or Sentinel-1 GRD. Weights CC BY 4.0.
* **SatlasPretrain** (Bastani et al. 2023, https://github.com/allenai/satlaspretrain_models):
  supervised multi-task on Sentinel-2, 9 bands; ``si`` single image, ``mi`` multi-image.
  Checkpoints ODC-BY per the authors (TorchGeo's mirror is labelled Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_S2_L1C = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11",
           "B12")
_SATLAS_S2 = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")
_HF = "https://hf.co/torchgeo"
_SSL4EO = "https://github.com/zhu-xlab/SSL4EO-S12"
_SATLAS = "https://github.com/allenai/satlaspretrain_models"


@dataclass(frozen=True)
class Weights:
    model: str          # timm architecture
    tag: str
    url: str
    bands: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    units: str
    license: str
    source: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def in_chans(self) -> int:
        return len(self.bands)

    @property
    def key(self) -> str:
        return f"{self.model}.{self.tag}"


def _s2_l1c(model, tag, url, notes=()):
    return Weights(model, tag, url, _S2_L1C, (0.0,) * 13, (10000.0,) * 13,
                   "Sentinel-2 L1C digital numbers (reflectance x 10000)", "CC BY 4.0",
                   _SSL4EO, notes)


def _satlas_s2(model, tag, url):
    return Weights(model, tag, url, _SATLAS_S2, (0.0,) * 9, (255.0,) * 3 + (8160.0,) * 6,
                   "B04/B03/B02 as the 8-bit true-colour product (TCI, 0-255); B05-B12 as "
                   "Sentinel-2 digital numbers", "ODC-BY", _SATLAS,
                   ("Satlas used TCI for RGB; with L2A reflectance x 10000 for all bands, "
                    "use std 3000 for B04/B03/B02 as an approximation",))


_ENTRIES = (
    _s2_l1c("resnet18", "ssl4eo_s2_moco", f"{_HF}/resnet18_sentinel2_all_moco/resolve/"
            "5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth"),
    _s2_l1c("resnet50", "ssl4eo_s2_moco", f"{_HF}/resnet50_sentinel2_all_moco/resolve/"
            "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth"),
    _s2_l1c("resnet50", "ssl4eo_s2_dino", f"{_HF}/resnet50_sentinel2_all_dino/resolve/"
            "d7f14bf5530d70ac69d763e58e77e44dbecfec7c/resnet50_sentinel2_all_dino-d6c330e9.pth"),
    _s2_l1c("vit_small_patch16_224", "ssl4eo_s2_moco",
            f"{_HF}/vit_small_patch16_224_sentinel2_all_moco/resolve/"
            "1cb683f6c14739634cdfaaceb076529adf898c74/"
            "vit_small_patch16_224_sentinel2_all_moco-67c9032d.pth"),
    _s2_l1c("vit_small_patch16_224", "ssl4eo_s2_dino",
            f"{_HF}/vit_small_patch16_224_sentinel2_all_dino/resolve/"
            "5b41dd418a79de47ac9f5be3e035405a83818a62/"
            "vit_small_patch16_224_sentinel2_all_dino-36bcc127.pth"),
    Weights("resnet50", "ssl4eo_s1_moco", f"{_HF}/resnet50_sentinel1_all_moco/resolve/"
            "e79862c667853c10a709bdd77ea8ffbad0e0f1cf/resnet50_sentinel1_all_moco-906e4356.pth",
            ("VV", "VH"), (-12.59, -20.26), (5.26, 5.91), "Sentinel-1 GRD backscatter in dB",
            "CC BY 4.0", _SSL4EO),
    _satlas_s2("resnet50", "satlas_s2_si_ms", f"{_HF}/satlas/resolve/"
               "081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet50_si_ms-1f454cc6.pth"),
    _satlas_s2("resnet50", "satlas_s2_mi_ms", f"{_HF}/satlas/resolve/"
               "081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet50_mi_ms-da5413d2.pth"),
    _satlas_s2("resnet152", "satlas_s2_si_ms", f"{_HF}/satlas/resolve/"
               "081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet152_si_ms-4500c6cb.pth"),
    _satlas_s2("resnet152", "satlas_s2_mi_ms", f"{_HF}/satlas/resolve/"
               "081d6607431bf36bdb59c223777cbb267131b8f2/sentinel2_resnet152_mi_ms-fd35b4bb.pth"),
)
WEIGHTS: dict[str, Weights] = {w.key: w for w in _ENTRIES}

# Checkpoint keys outside the backbone (classifier / pre-logits norm) that may be unused.
_HEAD_PREFIXES = ("fc.", "head.", "classifier.", "norm.", "fc_norm.")


def get_weights(model: str, tag: str) -> Weights:
    arch = model.split(".", 1)[0]
    w = WEIGHTS.get(f"{arch}.{tag}")
    if w is None:
        tags = sorted(v.tag for v in WEIGHTS.values() if v.model == arch)
        archs = sorted({v.model for v in WEIGHTS.values() if v.tag == tag})
        hint = (f"available for {arch!r}: {tags}" if tags else
                f"{tag!r} exists for {archs}" if archs else f"known: {sorted(WEIGHTS)}")
        raise ValueError(f"no pretrained weights {tag!r} for {arch!r}; {hint}")
    return w


def _download(url: str) -> dict:
    from torch.hub import load_state_dict_from_url

    return load_state_dict_from_url(url, map_location="cpu", progress=False, check_hash=True)


def load_into(model, weights: Weights) -> None:
    """Load ``weights`` into a timm model (``features_only`` or full). Every parameter and
    buffer of ``model`` must come from the checkpoint; unused checkpoint keys may only be the
    classifier / head."""
    raw = _download(weights.url)
    own = model.state_dict()
    # timm's feature wrappers for some families (e.g. ViT) prefix the wrapped model's keys
    sd = {k: raw[k] if k in raw else raw.get(k.removeprefix("model.")) for k in own}
    missing = sorted(k for k, v in sd.items() if v is None)
    if missing:
        raise RuntimeError(f"{weights.key}: checkpoint does not cover {len(missing)} model "
                           f"entries, e.g. {missing[:5]}")
    used = {k if k in raw else k.removeprefix("model.") for k in own}
    extra = sorted(k for k in raw if k not in used and not k.startswith(_HEAD_PREFIXES))
    if extra:
        raise RuntimeError(f"{weights.key}: unexpected checkpoint entries {extra[:5]}")
    bad = sorted(k for k in own if own[k].shape != sd[k].shape)
    if bad:
        raise RuntimeError(f"{weights.key}: shape mismatch for {bad[:5]}")
    model.load_state_dict(sd, strict=True)


__all__ = ["WEIGHTS", "Weights", "get_weights", "load_into"]
