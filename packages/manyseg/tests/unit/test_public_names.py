"""What users read and call doesn't carry names from inside the experiments."""

import importlib
import re
from pathlib import Path

import manyseg
import pytest

SOURCE = Path(manyseg.__file__).parent
# experiment arms; "PVT-v2-B0" names an architecture, and "B0" after it isn't an arm
ARMS = re.compile(r"(?<![-\w])(B[023]|R[024]|F[03]|J[12])\b")
BRANCHES = re.compile(r"\b(vec|feature|fix|exp|chore)/[a-z][\w-]+")
# experiment IDs (E026, E044b) and repository files that don't ship in the wheel
INTERNAL = re.compile(r"\bE0\d{2}[a-z]?\b|experiments/|architecture_review")


def test_the_library_source_names_no_experiment_arm_or_branch():
    found = [f"{path.relative_to(SOURCE)}:{n}: {line.strip()}"
             for path in sorted(SOURCE.rglob("*.py"))
             for n, line in enumerate(path.read_text().splitlines(), 1)
             if ARMS.search(line) or BRANCHES.search(line)]
    assert not found, "\n".join(found)


def test_the_library_source_cites_no_experiment_id_or_unshipped_file():
    found = [f"{path.relative_to(SOURCE)}:{n}: {line.strip()}"
             for path in sorted(SOURCE.rglob("*.py"))
             for n, line in enumerate(path.read_text().splitlines(), 1)
             if INTERNAL.search(line)]
    assert not found, "\n".join(found)


def test_vector_public_names_are_all_importable_and_unique():
    vectors = __import__("manyseg.vectors", fromlist=["__all__"])
    assert len(vectors.__all__) == len(set(vectors.__all__))
    assert all(hasattr(vectors, name) for name in vectors.__all__)


def test_model_presets_are_gone():
    """Removed before 0.1.0; docs/roadmap.md has the plan to rebuild them around goals."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("manyseg.presets")
