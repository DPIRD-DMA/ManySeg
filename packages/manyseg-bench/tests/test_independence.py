"""Architectural rule: manyseg (the library) must never import manyseg_bench."""

import pathlib
import subprocess
import sys


def test_library_does_not_import_bench():
    import manyseg

    src = pathlib.Path(manyseg.__file__).parent
    offenders = [str(p) for p in src.rglob("*.py") if "manyseg_bench" in p.read_text()]
    assert not offenders


def test_bench_core_imports_without_model_libraries():
    """Provider modules import their libraries lazily."""
    code = ("import sys; sys.modules['segmentation_models_pytorch'] = None; "
            "sys.modules['fastai'] = None; sys.modules['manyseg'] = None; "
            "import manyseg_bench.providers, manyseg_bench.training, manyseg_bench.cli")
    subprocess.run([sys.executable, "-c", code], check=True)
