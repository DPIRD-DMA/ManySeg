"""The docs site's API reference (scripts/build_docs.py, scripts/griffe_rst.py): every public
subpackage has a page, the reST-to-Markdown docstring conversion, and (when an interpreter with
the docs tools is available) a full offline build with no reference warnings. Needs a repo
checkout; skipped otherwise. The build test uses ``MANYSEG_DOCS_PYTHON`` (an interpreter with
docs/requirements.txt installed) or the current one, and skips when neither has the tools."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts"
PACKAGE = ROOT / "packages" / "manyseg" / "src" / "manyseg"
pytestmark = pytest.mark.skipif(not (SCRIPTS / "build_docs.py").exists(),
                                reason="needs a repo checkout (scripts/)")

# Subpackages with __all__ that are not pages of their own: re-exported at the top level.
NOT_PAGES = {"manyseg.models"}


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_all(path: Path) -> bool:
    return any(isinstance(n, (ast.Assign, ast.AugAssign)) and "__all__" in ast.dump(n)
               for n in ast.parse(path.read_text()).body)


def test_every_public_subpackage_has_a_page():
    pages = set(_module("build_docs").PAGES)
    public = {"manyseg." + ".".join(p.parent.relative_to(PACKAGE).parts)
              for p in PACKAGE.rglob("__init__.py")
              if p.parent != PACKAGE and _has_all(p)
              and not any(s.startswith("_") for s in p.parent.relative_to(PACKAGE).parts)}
    assert "manyseg" in pages
    assert public - NOT_PAGES <= pages, sorted(public - NOT_PAGES - pages)
    for page in pages:  # every page names a module that exists
        rel = Path(*page.split(".")[1:])
        assert (PACKAGE / rel / "__init__.py").exists() or (PACKAGE / f"{rel}.py").exists(), page


def test_convert_roles_blocks_directives():
    convert = _module("griffe_rst").convert
    link = {"decode_scene": "manyseg.vectors.decode_scene"}.get
    text = ("Decode with :func:`decode_scene` or :class:`~manyseg.weights.Weights`,\n"
            "one ``Polygon``\\ s per line::\n\n"
            "    a = 1\n"
            "        b = 2\n\n"
            "Then.\n\n"
            ".. warning:: Noisy on small sets,\n"
            "   calibrate on folds.\n\n"
            "Args:\n"
            "    x: see :mod:`manyseg.points`.")
    out = convert(text, link)
    assert out.startswith("Decode with [`decode_scene`][manyseg.vectors.decode_scene] or "
                          "`Weights`, one ``Polygon``s per line:\n")  # first paragraph joined
    assert "\n```python\na = 1\n    b = 2\n```\n" in out
    assert "!!! warning\n    Noisy on small sets,\n    calibrate on folds.\n" in out
    assert "    x: see `manyseg.points`." in out
    assert not re.search(r":(func|class|mod):`|::\n|\.\. ", out)
    assert convert("Args:\n    x: y") == "Args:\n    x: y"  # sections are not joined


def _docs_python() -> str | None:
    exe = os.environ.get("MANYSEG_DOCS_PYTHON", sys.executable)
    probe = subprocess.run([exe, "-c", "import mkdocs, material, mkdocstrings_handlers.python"],
                           capture_output=True)
    return exe if probe.returncode == 0 else None


@pytest.mark.slow
def test_reference_builds_offline_without_warnings(tmp_path, monkeypatch):
    exe = _docs_python()
    if exe is None:
        pytest.skip("no interpreter with docs/requirements.txt (set MANYSEG_DOCS_PYTHON)")
    build_docs = _module("build_docs")
    monkeypatch.setattr(build_docs, "SRC", tmp_path / "site_src")
    monkeypatch.setattr(build_docs, "OUT", tmp_path / "site")
    monkeypatch.setattr(build_docs.subprocess, "call", lambda cmd: 0)  # assemble only
    assert build_docs.main() == 0
    # No network: no proxy can be reached, and mkdocstrings loads no inventories.
    env = {**os.environ, "NO_MKDOCS_2_WARNING": "1", "HTTP_PROXY": "http://127.0.0.1:9",
           "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""}
    out = subprocess.run([exe, "-m", "mkdocs", "build", "-f", str(tmp_path / "mkdocs.gen.yml")],
                         capture_output=True, text=True, env=env, timeout=600)
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-4000:]
    bad = [line for line in log.splitlines() if "WARNING" in line
           and re.search(r"mkdocstrings|griffe|autorefs|reference/", line)]
    assert not bad, "\n".join(bad)
    for page in build_docs.PAGES:
        html = (tmp_path / "site" / "reference" / page / "index.html").read_text()
        assert f'id="{page}"' in html, page
        assert not re.search(r":(func|class|mod|meth|attr|data):`|\]\[manyseg", html), page
    html = (tmp_path / "site" / "reference" / "manyseg.vectors" / "index.html").read_text()
    assert 'id="manyseg.vectors.road_targets"' in html  # the function, not the submodule
    assert "doc-signature" in html
    heads = (tmp_path / "site" / "reference" / "manyseg.heads" / "index.html").read_text()
    assert 'id="manyseg.heads.ImageHead.fc"' not in heads  # layers set in __init__
    assert 'id="manyseg.heads.PixelHead.forward"' in heads  # documented forward stays
