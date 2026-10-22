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
_SSL4EO_CITE = ("Wang, Y., Braham, N. A. A., Xiong, Z., Liu, C., Albrecht, C. M., Zhu, X. X. "
                "(2023). SSL4EO-S12: A large-scale multimodal, multitemporal dataset for "
                "self-supervised learning in Earth observation. IEEE Geoscience and Remote "
                "Sensing Magazine, 11(3), 98-106.")
_SATLAS_CITE = ("Bastani, F., Wolters, P., Gupta, R., Ferdinando, J., Kembhavi, A. (2023). "
                "SatlasPretrain: A large-scale dataset for remote sensing image understanding. "
                "ICCV 2023.")


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
    clip: bool = False  # clip (x - mean) / std to [0, 1] after normalising, as in pretraining
    citation: str = ""  # cite this when you publish work using the weights (licence: attribution)

    @property
    def in_channels(self) -> int:
        return len(self.bands)

    @property
    def key(self) -> str:
        return f"{self.model}.{self.tag}"


def _s2_l1c(model, tag, url, notes=()):
    # Pretraining (SSL4EO-S12 pretrain_*_s2c.py with --dtype uint8, no --normalize): DN / 10000
    # -> 8 bit -> ToTensor, i.e. reflectance clipped to [0, 1]; no mean/std step.
    return Weights(model, tag, url, _S2_L1C, (0.0,) * 13, (10000.0,) * 13,
                   "Sentinel-2 L1C digital numbers (reflectance x 10000)", "CC BY 4.0",
                   _SSL4EO, notes, clip=True, citation=_SSL4EO_CITE)


def _satlas_s2(model, tag, url):
    # Satlas Normalization.md: TCI (RGB, 0-255) / 255 and B05..B12 / 8160, clipped to [0, 1].
    # ESA's L1C TCI saturates at reflectance 0.2 (DN 2000), so TCI / 255 ~= DN / 2000.
    return Weights(model, tag, url, _SATLAS_S2, (0.0,) * 9, (2000.0,) * 3 + (8160.0,) * 6,
                   "Sentinel-2 digital numbers (reflectance x 10000); RGB stands in for the "
                   "8-bit TCI product (saturating at DN 2000)", "ODC-BY", _SATLAS,
                   ("Satlas inputs were L1C TCI + bands; DN / 2000 approximates TCI / 255",),
                   clip=True, citation=_SATLAS_CITE)


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
            "CC BY 4.0", _SSL4EO, citation=_SSL4EO_CITE),
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


def _resample_pos_embed(model, key: str, pos, shape):
    """Resample a ViT's absolute position embedding to the model's patch grid (timm's
    bicubic resampling, prefix tokens kept)."""
    from timm.layers import resample_abs_pos_embed

    owner = model.get_submodule(key.rsplit(".", 1)[0]) if "." in key else model
    grid = owner.patch_embed.grid_size
    prefix = shape[1] - grid[0] * grid[1]
    return resample_abs_pos_embed(pos, new_size=list(grid), num_prefix_tokens=prefix)


def load_into(model, weights: Weights) -> None:
    """Load ``weights`` into a timm model (``features_only`` or full). Every parameter and
    buffer of ``model`` must come from the checkpoint; unused checkpoint keys may only be the
    classifier / head."""
    raw = _download(weights.url)
    own = model.state_dict()
    # timm's feature wrappers for some families (e.g. ViT) prefix the wrapped model's keys
    sd = {k: raw[k] if k in raw else raw.get(k.removeprefix("model.")) for k in own}
    missing = sorted(k for k, v in sd.items() if v is None)
    for k, v in sd.items():  # a ViT built for another img_size: resample the position table
        if v is not None and k.endswith("pos_embed") and v.shape != own[k].shape:
            sd[k] = _resample_pos_embed(model, k, v, own[k].shape)
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


def attribution(model) -> list[dict]:
    """The pretrained weights each backbone of ``model`` started from: ``[{"module", "weights",
    "provider", "license", "citation", "source"}]``, one entry per pretrained backbone module
    (an encoder shared as one module appears once; ``shared:<input>`` copies, which share
    weights but are separate modules, appear once each).

    Fine-tuned weights stay subject to these licences, most of which require attribution
    (SSL4EO-S12 CC BY 4.0, SatlasPretrain ODC-BY): cite ``citation`` when you publish or share
    work built on them. ``provider`` is ``"manyseg"`` for the remote-sensing weights here and
    ``"timm"`` for timm's ImageNet checkpoints, with the licence timm lists ("unknown" when it
    lists none: check the model card at ``source``). :func:`manyseg.save` stores this list and
    :func:`manyseg.load` restores each entry on its backbone, which was rebuilt without
    downloading anything, so a reloaded and re-saved model keeps it; a backbone replaced after
    loading reports its own weights."""
    from .backbones import TimmBackbone

    out = []
    for path, m in model.named_modules():
        if isinstance(m, TimmBackbone):
            rec = _backbone_attribution(m)
            if rec is not None:
                out.append({"module": path, **rec})
    return out


def _backbone_attribution(m) -> dict | None:
    if m.restored_attribution is not None:
        return dict(m.restored_attribution)
    if not m.pretrained:
        return None
    if m.weights is not None:
        w = WEIGHTS[m.weights]
        return {"weights": w.key, "provider": "manyseg", "license": w.license,
                "citation": w.citation, "source": w.source}
    cfg = getattr(m.model, "pretrained_cfg", {}) or {}
    tag = cfg.get("tag")
    hub = cfg.get("hf_hub_id")
    papers = cfg.get("paper_ids") or ""
    return {"weights": f"{cfg.get('architecture', m.name)}.{tag}" if tag else m.name,
            "provider": "timm", "license": cfg.get("license") or "unknown",
            "citation": papers if isinstance(papers, str) else ", ".join(papers),
            "source": f"https://huggingface.co/{hub}" if hub else cfg.get("url", "")}


__all__ = ["WEIGHTS", "Weights", "attribution", "get_weights", "load_into"]
