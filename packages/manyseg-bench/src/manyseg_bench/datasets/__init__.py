from . import oxford_pet
from .base import DatasetInfo, Split, data_root, load_split
from .gpu_loader import GPULoader

DATASETS = {"oxford_pet": oxford_pet}


def get_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name]


__all__ = ["DATASETS", "DatasetInfo", "GPULoader", "Split", "data_root", "get_dataset",
           "load_split"]
