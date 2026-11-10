"""Architectural rule: manyseg (the library) must never import manyseg_bench."""

import pathlib
import subprocess
import sys


def test_library_does_not_import_bench():
    import manyseg

    src = pathlib.Path(manyseg.__file__).parent
    offenders = [str(p) for p in src.rglob("*.py") if "manyseg_bench" in p.read_text()]
    assert not offenders


BLOCKED = ("segmentation_models_pytorch", "fastai", "timm", "manyseg", "manyseg.losses",
           "manyseg.utils", "manyseg.instances", "manyseg.points", "manyseg.vectors",
           "manyseg.weights", "manyseg.quantize")


def test_bench_core_imports_without_model_libraries():
    """Provider modules and the ManySeg re-exports (losses) import their libraries lazily, so
    ``manyseg-bench[smp]`` (no ManySeg) imports everything a run needs."""
    code = ("import sys\n"
            f"for m in {BLOCKED!r}: sys.modules[m] = None\n"
            "import manyseg_bench.providers, manyseg_bench.training, manyseg_bench.cli\n"
            "import manyseg_bench.tasks, manyseg_bench.losses, manyseg_bench.experiments\n"
            "import manyseg_bench.experiments.runner, manyseg_bench.experiments.evaluate\n"
            "import manyseg_bench.evaluation, manyseg_bench.reporting, manyseg_bench.datasets\n")
    subprocess.run([sys.executable, "-c", code], check=True)
