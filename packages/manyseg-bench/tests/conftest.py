import os

import pytest  # noqa: F401
import torch

# Cap intra-op threads per test process: pytest-xdist workers each default to one thread per
# core, which oversubscribes the (shared) machine badly.
torch.set_num_threads(int(os.environ.get("MANYSEG_TEST_THREADS", "4")))
