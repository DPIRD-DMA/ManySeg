"""Train-only subset: reproducible by seed, nested across sizes, aligned, RNG untouched."""

import numpy as np
import pytest
from manyseg_bench.datasets.base import Split
from manyseg_bench.experiments.runner import select_train_subset


def _split(n=20, inputs=False):
    images = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1)
    return Split({"s2": images} if inputs else images, np.arange(n, dtype=np.uint8),
                 [str(i) for i in range(n)], {"v": np.arange(n)})


def test_same_seed_same_samples_and_other_seed_differs():
    a = select_train_subset(_split(), 5, seed=0).names
    assert a == select_train_subset(_split(), 5, seed=0).names
    assert a != select_train_subset(_split(), 5, seed=1).names


def test_sizes_nest_for_one_seed():
    small = select_train_subset(_split(), 5, seed=3).names
    assert set(small) < set(select_train_subset(_split(), 10, seed=3).names)


@pytest.mark.parametrize("inputs", [False, True])
def test_images_masks_names_and_vectors_stay_aligned(inputs):
    s = select_train_subset(_split(inputs=inputs), 6, seed=2)
    images = s.images["s2"] if inputs else s.images
    expected = [int(n) for n in s.names]
    assert images[:, 0, 0, 0].tolist() == expected
    assert s.masks.tolist() == expected
    assert s.vectors["v"].tolist() == expected


def test_global_rng_is_untouched():
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    select_train_subset(_split(), 5, seed=0)
    assert np.random.random() == expected


@pytest.mark.parametrize("n", [0, 21])
def test_out_of_range_size_is_an_error(n):
    with pytest.raises(ValueError, match="train_subset"):
        select_train_subset(_split(), n, seed=0)
