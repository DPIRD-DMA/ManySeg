"""Build the documentation site (MkDocs + Material) from the repo's existing Markdown.

Assembles README.md (home), CHANGELOG.md, docs/*.md, the model builder page (docs/builder),
examples/README.md, the notebooks (with their saved outputs; nothing is executed), every
experiments/E*/README.md and an API reference generated from the
docstrings (mkdocstrings, one page per public module in REFERENCE) into build/site_src
(gitignored), writes a generated MkDocs config with the nav, and builds static HTML into
build/site. No new runtime dependency: the docs tools are listed in docs/requirements.txt, and
mkdocstrings reads the source statically (griffe), so the build needs neither torch nor the
optional extras, a GPU or the network (once the tools are installed). Run with
    uv run python scripts/build_docs.py    # uvx runs mkdocs with docs/requirements.txt
    python scripts/build_docs.py           # with docs/requirements.txt pip-installed (RTD)
Publishing (GitHub Pages or elsewhere) is a separate, deliberate step.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "site_src"
# Read the Docs sets READTHEDOCS_OUTPUT and installs mkdocs with pip (.readthedocs.yaml).
RTD_OUT = os.environ.get("READTHEDOCS_OUTPUT")
REQUIREMENTS = ROOT / "docs" / "requirements.txt"
OUT = Path(RTD_OUT) / "html" if RTD_OUT else ROOT / "build" / "site"

DOCS = [("Architecture review", "architecture_review.md"), ("API design", "api_design.md"),
        ("Benchmark design", "benchmark_design.md"), ("Testing strategy", "testing_strategy.md"),
        ("Backbone compatibility", "backbone_compatibility.md"), ("Roadmap", "roadmap.md")]

# API reference: (section, modules), one page per module. A page lists the module's __all__, or
# without one its public names defined there (imports left out). Every public subpackage
# (an __init__ with __all__) is listed; tests/unit/test_docs_reference.py checks that.
REFERENCE = [
    ("Models", ["manyseg", "manyseg.presets", "manyseg.weights", "manyseg.io", "manyseg.hub"]),
    ("Outputs", ["manyseg.losses", "manyseg.instances", "manyseg.points", "manyseg.vectors",
                 "manyseg.vectors.roofs", "manyseg.utils"]),
    ("Deployment", ["manyseg.export", "manyseg.quantize", "manyseg.distributed",
                    "manyseg.describe"]),
    ("Inputs", ["manyseg.conditioning", "manyseg.temporal"]),
    ("Building blocks", ["manyseg.backbones", "manyseg.decoders", "manyseg.heads",
                         "manyseg.fusion", "manyseg.adapters", "manyseg.registry"]),
]
PAGES = [m for _, mods in REFERENCE for m in mods]
# mkdocstrings-python options: Google sections, signatures, source order, only what a module
# exports, and nothing inherited from torch.nn.Module (forward, training, dump_patches, ...).
HANDLER = {
    "paths": [str(ROOT / "packages" / "manyseg" / "src")],
    "options": {
        "docstring_style": "google",
        "docstring_options": {"ignore_init_summary": True, "warn_unknown_params": False,
                              "warn_missing_types": False},
        "docstring_section_style": "table",
        "merge_init_into_class": True,
        "members_order": "source",
        "filters": "public",
        "inherited_members": False,
        "show_if_no_docstring": True,
        "show_source": False,
        "show_bases": True,
        "show_root_heading": True,
        "show_root_full_path": True,
        "heading_level": 1,
        "show_symbol_type_heading": True,
        "show_symbol_type_toc": True,
        "separate_signature": True,
        "show_signature_annotations": True,
        "signature_crossrefs": True,
        "line_length": 88,
        "summary": {"attributes": True, "functions": True, "classes": True, "modules": False},
        "extensions": [{str(ROOT / "scripts" / "griffe_rst.py"): {"pages": PAGES}}],
    },
}


def _title(md: Path) -> str:
    for line in md.read_text().splitlines():
        if line.startswith("# "):
            return re.sub(r"\s*\(pre-registered\)\s*$", "", line[2:].strip())
    return md.parent.name


def main() -> int:
    shutil.rmtree(SRC, ignore_errors=True)
    (SRC / "experiments").mkdir(parents=True)
    shutil.copytree(ROOT / "docs" / "assets", SRC / "docs" / "assets")
    # The model builder is a static page (HTML + JS); MkDocs copies it through unchanged.
    shutil.copytree(ROOT / "docs" / "builder", SRC / "docs" / "builder")
    # README links point at docs/ and examples/ relative to the repo root; keep that layout.
    shutil.copy(ROOT / "README.md", SRC / "index.md")
    shutil.copy(ROOT / "CHANGELOG.md", SRC / "CHANGELOG.md")  # linked from README
    for _, name in DOCS:
        shutil.copy(ROOT / "docs" / name, SRC / "docs" / name)
    (SRC / "examples").mkdir()
    shutil.copy(ROOT / "examples" / "README.md", SRC / "examples" / "index.md")
    (SRC / "notebooks").mkdir()
    shutil.copy(ROOT / "notebooks" / "README.md", SRC / "notebooks" / "index.md")
    notebooks = []
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy(nb, SRC / "notebooks" / nb.name)
        title = next(c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "markdown")
        notebooks.append((title["source"][0].lstrip("# ").strip(), f"notebooks/{nb.name}"))
    (SRC / "reference").mkdir()
    for mod in PAGES:
        (SRC / "reference" / f"{mod}.md").write_text(f"::: {mod}\n")
    exps = []
    for d in sorted((ROOT / "experiments").glob("E*/README.md")):
        dst = SRC / "experiments" / f"{d.parent.name}.md"
        shutil.copy(d, dst)
        exps.append((f"{d.parent.name.split('_')[0]}: {_title(d).split(':', 1)[-1].strip()}",
                     f"experiments/{dst.name}"))
    nav = ["  - Home: index.md", "  - Documentation:"]
    nav += [f"      - {t}: docs/{n}" for t, n in DOCS]
    nav += ["      - Changelog: CHANGELOG.md"]
    nav += ["  - Model builder: docs/builder/index.html"]
    nav += ["  - Examples: examples/index.md", "  - Notebooks:", "      - Overview: notebooks/index.md"]
    nav += [f"      - {json.dumps(t)}: {p}" for t, p in notebooks]
    nav += ["  - API reference:"]
    for section, mods in REFERENCE:
        nav += [f"      - {section}:"] + [f"          - {m}: reference/{m}.md" for m in mods]
    nav += ["  - Experiments:"]
    nav += [f"      - {json.dumps(t)}: {p}" for t, p in exps]  # JSON strings are valid YAML
    cfg = SRC.parent / "mkdocs.gen.yml"
    cfg.write_text("\n".join([
        "site_name: ManySeg",
        "repo_url: https://github.com/DPIRD-DMA/ManySeg",
        f"docs_dir: {SRC}",
        f"site_dir: {OUT}",
        "theme:",
        "  name: material",
        "  logo: docs/assets/icon-light.svg",
        "  features: [navigation.sections, content.code.copy, search.highlight]",
        "  palette:",
        "    - scheme: default",
        "      toggle: {icon: material/brightness-7, name: Dark mode}",
        "    - scheme: slate",
        "      toggle: {icon: material/brightness-4, name: Light mode}",
        "markdown_extensions: [tables, admonition, toc, pymdownx.highlight, pymdownx.superfences]",
        "plugins:",
        "  - search",
        "  - mkdocs-jupyter: {execute: false, include_source: true}",
        "  - mkdocstrings:",
        "      default_handler: python",
        f"      handlers: {{python: {json.dumps(HANDLER)}}}",  # JSON is valid YAML
        "nav:", *nav, ""]))
    # mkdocs 2.0 drops the plugin/theme system Material needs: stay on 1.x.
    if RTD_OUT or shutil.which("uvx") is None:  # mkdocs already installed (pip)
        cmd = [sys.executable, "-m", "mkdocs", "build", "-q"]
    else:
        cmd = ["uvx", "--from", "mkdocs<2", "--with-requirements", str(REQUIREMENTS), "mkdocs",
               "build", "-q"]
    return subprocess.call(cmd + ["-f", str(cfg)])


if __name__ == "__main__":
    sys.exit(main())
