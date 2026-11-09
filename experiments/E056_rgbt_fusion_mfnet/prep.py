"""Pack MFNet (chenzeyang1/RGB-T_datasets copy) into train/val/test .npz files.

test = test_day.txt + test_night.txt (the official 393-image test split, 205 day + 188 night).
train_noflip.txt holds the official train + val (1,175); every third image goes to val (391),
the rest to train (784). Val is only for monitoring: runs use a fixed epoch count, no selection.
Night images have ids ending in N.
Usage: python prep.py <MFNet dir> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def ids(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def pack(root: Path, names: list[str], out: Path) -> None:
    rgb = np.stack([np.array(Image.open(root / "RGB" / f"{n}.png").convert("RGB")) for n in names])
    th = np.stack([np.array(Image.open(root / "Modal" / f"{n}.png").convert("L")) for n in names])
    lab = np.stack([np.array(Image.open(root / "Label" / f"{n}.png")) for n in names])
    night = np.array([n.endswith("N") for n in names])
    assert lab.max() < 9, lab.max()
    np.savez(out, rgb=rgb, thermal=th, label=lab, night=night, ids=np.array(names))
    print(out.name, rgb.shape, th.shape, lab.shape, f"night {night.sum()}/{len(names)}")


def main(src: str, dst: str) -> None:
    root, out = Path(src), Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    trval = ids(root / "train_noflip.txt")
    # test.txt has 394 ids; 01203N is in neither day nor night list. Use day + night (393).
    test = ids(root / "test_day.txt") + ids(root / "test_night.txt")
    assert not set(trval) & set(test)
    pack(root, [n for i, n in enumerate(trval) if i % 3 != 2], out / "train.npz")
    pack(root, [n for i, n in enumerate(trval) if i % 3 == 2], out / "val.npz")
    pack(root, test, out / "test.npz")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
