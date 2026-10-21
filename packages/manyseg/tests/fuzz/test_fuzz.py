"""Seeded random configurations (see fuzz_configs.py): every one builds and passes the model
invariants, or is refused with a clear ValueError / NotImplementedError. A rejection counts as
a pass; anything else is a bug.

For a large one-off run: ``uv run python tests/fuzz/fuzz_configs.py -n 2000 --export-every 5``.
"""

import random

import pytest
from fuzz_configs import Rejected, check, sample_config

# Seeds 0-19 that are invalid configurations (each reason checked by hand): pinned so that a
# change refusing valid configurations (or accepting invalid ones) fails here instead of
# quietly counting as a pass. Changing the generator reshuffles the seeds: re-pin, re-check.
REJECTED = {0: "out_indices", 2: "input image", 3: "out_stride=2", 5: "time series",
            6: "out_stride=2", 8: "time series", 10: "time series", 16: "early fusion",
            17: "cannot also have its own decoder", 18: "no image-scope heads"}


def _run(seed: int, export: bool = False) -> str:
    cfg = sample_config(random.Random(seed))
    try:
        return check(cfg, seed, export=export)
    except Rejected as e:
        assert str(e).strip(), f"empty rejection message for {cfg}"
        return str(e)


@pytest.mark.parametrize("seed", range(20))
def test_random_config(seed):
    out = _run(seed)
    if seed in REJECTED:
        assert REJECTED[seed] in out, out
    else:
        assert out == "ok", out


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(20, 220))
def test_random_config_with_export(seed):
    _run(seed, export=seed % 4 == 0)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize("seed", range(220, 232))
def test_random_config_on_cuda(seed):
    """CUDA vs CPU, a bf16 training step, bf16 / channels_last vs fp32; compile on some
    (see fuzz_gpu.py)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from fuzz_gpu import check_gpu

    try:
        check_gpu(sample_config(random.Random(seed)), seed, compile_=seed % 4 == 0)
    except Rejected as e:
        assert str(e).strip()
