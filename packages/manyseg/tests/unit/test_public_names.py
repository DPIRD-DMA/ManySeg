"""What users read and call doesn't carry names from inside the experiments."""

import re
from pathlib import Path

import manyseg

SOURCE = Path(manyseg.__file__).parent
# experiment arms; "PVT-v2-B0" names an architecture, and "B0" after it isn't an arm
ARMS = re.compile(r"(?<![-\w])(B[023]|R[024]|F[03]|J[12])\b")
BRANCHES = re.compile(r"\b(vec|feature|fix|exp|chore)/[a-z][\w-]+")


def test_the_library_source_names_no_experiment_arm_or_branch():
    found = [f"{path.relative_to(SOURCE)}:{n}: {line.strip()}"
             for path in sorted(SOURCE.rglob("*.py"))
             for n, line in enumerate(path.read_text().splitlines(), 1)
             if ARMS.search(line) or BRANCHES.search(line)]
    assert not found, "\n".join(found)


def test_vector_public_names_are_all_importable_and_unique():
    vectors = __import__("manyseg.vectors", fromlist=["__all__"])
    assert len(vectors.__all__) == len(set(vectors.__all__))
    assert all(hasattr(vectors, name) for name in vectors.__all__)
