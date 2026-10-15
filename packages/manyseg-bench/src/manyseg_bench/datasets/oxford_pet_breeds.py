"""Oxford-IIIT Pet breed classification (37 classes; Parkhi et al. 2012). License: CC BY-SA 4.0.

Same images, resize and splits as ``oxford_pet`` (shares its cache; validation = the sha1 10%
of ``trainval``), with the official class ids from ``annotations/list.txt`` (1-37 -> 0-36)
as image-level labels instead of the trimaps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import oxford_pet
from .base import DatasetInfo

CACHE_NAME = "oxford_pet"
TASK = "classification"
EVAL_BATCH_SIZE = 64


def _class_ids(raw: Path) -> dict[str, int]:
    ids = {}
    for ln in (Path(raw) / "annotations" / "list.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            name, cid, *_ = ln.split()
            ids[name] = int(cid) - 1
    return ids


def _breeds() -> tuple[str, ...]:
    # Official id order (alphabetical, cats and dogs interleaved as in list.txt).
    return ("Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
            "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair", "chihuahua",
            "Egyptian_Mau", "english_cocker_spaniel", "english_setter", "german_shorthaired",
            "great_pyrenees", "havanese", "japanese_chin", "keeshond", "leonberger",
            "Maine_Coon", "miniature_pinscher", "newfoundland", "Persian", "pomeranian", "pug",
            "Ragdoll", "Russian_Blue", "saint_bernard", "samoyed", "scottish_terrier",
            "shiba_inu", "Siamese", "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier",
            "yorkshire_terrier")


INFO = DatasetInfo(**{**oxford_pet.INFO.to_dict(), "name": "oxford_pet_breeds",
                      "num_classes": 37, "class_names": _breeds(),
                      "notes": ("labels: official class ids of annotations/list.txt",
                                *oxford_pet.INFO.notes)})


def split_labels(raw: Path, names: list[str]) -> np.ndarray:
    ids = _class_ids(raw)
    missing = [n for n in names if n not in ids]
    if missing:
        raise RuntimeError(f"no class id for {len(missing)} images, e.g. {missing[:3]}")
    labels = np.array([ids[n] for n in names], np.int64)
    for n, c in zip(names, labels):  # the id must name the file's breed
        if n.rsplit("_", 1)[0] != INFO.class_names[c]:
            raise RuntimeError(f"class id {c} of {n} is {INFO.class_names[c]}")
    return labels


def prepare(raw: Path, cache: Path, **kw) -> dict:
    return oxford_pet.prepare(raw, cache, **kw)
