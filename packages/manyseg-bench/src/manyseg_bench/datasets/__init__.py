from . import (
    ai4boundaries,
    bigearthnet,
    cloudsen12,
    eurosat,
    oam_tcd,
    oxford_pet,
    oxford_pet_breeds,
    pastis,
    pastis_ts,
    sen1floods11,
)
from .base import DatasetInfo, Split, data_root, load_split
from .gpu_loader import GPULoader

DATASETS = {"oxford_pet": oxford_pet, "sen1floods11": sen1floods11, "oam_tcd": oam_tcd,
            "ai4boundaries": ai4boundaries, "cloudsen12": cloudsen12, "pastis": pastis,
            "eurosat": eurosat, "oxford_pet_breeds": oxford_pet_breeds,
            "bigearthnet": bigearthnet, "pastis_ts": pastis_ts}


class _Variant:
    """A dataset module with a different INFO (e.g. extra inputs) sharing preparation; every
    other attribute (constants, ``prepare``, ``load_vectors``, ...) comes from the base."""

    def __init__(self, base, info, cache_name, **overrides):
        self._base, self.INFO, self.CACHE_NAME = base, info, cache_name
        self._overrides = overrides  # e.g. TASK, split_labels (None: the variant has none)

    def __getattr__(self, k):
        if k in self._overrides:
            return self._overrides[k]
        return getattr(self._base, k)


for _info in (sen1floods11.INFO_DEM, sen1floods11.INFO_S1, sen1floods11.INFO_S2,
              sen1floods11.INFO_META, sen1floods11.INFO_META_NOGEO):
    DATASETS[_info.name] = _Variant(sen1floods11, _info, "sen1floods11")
for _info in cloudsen12.VARIANTS:
    DATASETS[_info.name] = _Variant(cloudsen12, _info, cloudsen12.CACHE_NAME)
for _info in pastis.VARIANTS:
    DATASETS[_info.name] = _Variant(pastis, _info, pastis.CACHE_NAME)
for _info in pastis_ts.VARIANTS:
    DATASETS[_info.name] = _Variant(pastis_ts, _info, pastis_ts.CACHE_NAME,
                                    **pastis_ts.VARIANT_OVERRIDES.get(_info.name, {}))
for _info in bigearthnet.VARIANTS:
    DATASETS[_info.name] = _Variant(bigearthnet, _info, bigearthnet.CACHE_NAME,
                                    **bigearthnet.VARIANT_OVERRIDES.get(_info.name, {}))


def get_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name]


__all__ = ["DATASETS", "DatasetInfo", "GPULoader", "Split", "data_root", "get_dataset",
           "load_split"]
