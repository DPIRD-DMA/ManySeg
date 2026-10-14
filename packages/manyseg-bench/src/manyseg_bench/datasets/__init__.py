from . import ai4boundaries, oam_tcd, oxford_pet, sen1floods11
from .base import DatasetInfo, Split, data_root, load_split
from .gpu_loader import GPULoader

DATASETS = {"oxford_pet": oxford_pet, "sen1floods11": sen1floods11, "oam_tcd": oam_tcd,
            "ai4boundaries": ai4boundaries}


class _Variant:
    """A dataset module with a different INFO (e.g. extra inputs) sharing preparation."""

    def __init__(self, base, info, cache_name):
        self.__dict__.update({k: getattr(base, k) for k in dir(base) if k.isupper()})
        self.INFO, self.prepare, self.CACHE_NAME = info, base.prepare, cache_name


for _info in (sen1floods11.INFO_DEM, sen1floods11.INFO_S1, sen1floods11.INFO_S2):
    DATASETS[_info.name] = _Variant(sen1floods11, _info, "sen1floods11")


def get_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name]


__all__ = ["DATASETS", "DatasetInfo", "GPULoader", "Split", "data_root", "get_dataset",
           "load_split"]
