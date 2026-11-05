"""Build the documentation site (MkDocs + Material) from the repo's existing Markdown.

Assembles README.md (home), CHANGELOG.md, CONTRIBUTING.md, docs/*.md, the model builder page
(docs/builder), examples/ (README and scripts), the notebooks (with their saved outputs; nothing
is executed), every experiments/E*/README.md and an API reference generated from the docstrings
(mkdocstrings, one page per public module in REFERENCE) into build/site_src (gitignored), writes
a generated MkDocs config with the nav, and builds static HTML into build/site. The build is
strict: a broken link, image or anchor fails it. No new runtime dependency: the docs tools are
listed in docs/requirements.txt, and mkdocstrings reads the source statically (griffe), so the
build needs neither torch nor the optional extras, a GPU or the network (once the tools are
installed). Run with
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

DOCS = [("API design", "api_design.md"), ("Benchmark design", "benchmark_design.md"),
        ("Testing strategy", "testing_strategy.md"),
        ("Backbone compatibility", "backbone_compatibility.md"), ("Roadmap", "roadmap.md"),
        ("Architecture review", "architecture_review.md")]

# API reference: (section, modules), one page per module. A page lists the module's __all__, or
# without one its public names defined there (imports left out). Every public subpackage
# (an __init__ with __all__) is listed; tests/unit/test_docs_reference.py checks that.
REFERENCE = [
    ("Models", ["manyseg", "manyseg.weights", "manyseg.io", "manyseg.hub"]),
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
            title = re.sub(r"\s*\(pre-registered[^)]*\)\s*$", "", line[2:].strip())
            return re.sub(r"; pre-registered\)\s*$", ")", title)
    return md.parent.name


def _site_links(text: str) -> str:
    """Point links written for the repo layout at the pages they become on the site."""
    text = re.sub(r"\]\(((?:\.\./)*)(experiments|examples|notebooks)/\)", r"](\1\2/index.md)", text)
    text = re.sub(r"\]\(((?:\.\./)*)examples/README\.md", r"](\1examples/index.md", text)
    text = re.sub(r"\]\(((?:\.\./)*)experiments/(E[^/)]+)/README\.md",
                  r"](\1experiments/\2.md", text)
    return re.sub(r"\]\(((?:\.\./)*)docs/builder/\)", r"](\1docs/builder/index.html)", text)


def _copy_page(src: Path, dst: Path) -> None:
    dst.write_text(_site_links(src.read_text()))


def _copy_experiment(readme: Path, dst: Path) -> None:
    """Write-up to experiments/<name>.md, the files it links to under experiments/<name>/."""

    def local(match: re.Match) -> str:
        target = readme.parent / match[1]
        if not target.is_file():
            return match[0]
        copy = dst.with_suffix("") / match[1]
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(target, copy)
        return f"]({dst.stem}/{match[1]})"

    dst.write_text(re.sub(r"\]\(([^)#:\s]+)\)", local, _site_links(readme.read_text())))


def main() -> int:
    shutil.rmtree(SRC, ignore_errors=True)
    (SRC / "experiments").mkdir(parents=True)
    shutil.copytree(ROOT / "docs" / "assets", SRC / "docs" / "assets")
    # The model builder is a static page (HTML + JS); MkDocs copies it through unchanged.
    shutil.copytree(ROOT / "docs" / "builder", SRC / "docs" / "builder")
    # README links point at docs/ and examples/ relative to the repo root; keep that layout.
    _copy_page(ROOT / "README.md", SRC / "index.md")
    _copy_page(ROOT / "CHANGELOG.md", SRC / "CHANGELOG.md")  # linked from README
    _copy_page(ROOT / "CONTRIBUTING.md", SRC / "CONTRIBUTING.md")  # linked from README
    for _, name in DOCS:
        _copy_page(ROOT / "docs" / name, SRC / "docs" / name)
    (SRC / "examples").mkdir()
    _copy_page(ROOT / "examples" / "README.md", SRC / "examples" / "index.md")
    for script in (ROOT / "examples").glob("*.py"):  # linked from the examples page
        shutil.copy(script, SRC / "examples" / script.name)
    (SRC / "notebooks").mkdir()
    _copy_page(ROOT / "notebooks" / "README.md", SRC / "notebooks" / "index.md")
    notebooks = []
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy(nb, SRC / "notebooks" / nb.name)
        cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
        title = next(c for c in cells if c["cell_type"] == "markdown")
        notebooks.append((title["source"][0].lstrip("# ").strip(), f"notebooks/{nb.name}"))
    (SRC / "reference").mkdir()
    for mod in PAGES:
        (SRC / "reference" / f"{mod}.md").write_text(f"::: {mod}\n")
    exps = []
    for d in sorted((ROOT / "experiments").glob("E*/README.md")):
        dst = SRC / "experiments" / f"{d.parent.name}.md"
        _copy_experiment(d, dst)
        exps.append((f"{d.parent.name.split('_')[0]}: {_title(d).split(':', 1)[-1].strip()}",
                     f"experiments/{dst.name}"))
    (SRC / "experiments" / "index.md").write_text("\n".join([
        "# Experiments", "",
        "The evidence behind the defaults and tips, negative results included.", "",
        *[f"* [{t}]({p.removeprefix('experiments/')})" for t, p in exps], ""]))
    nav = ["  - Home: index.md", "  - Documentation:"]
    nav += [f"      - {t}: docs/{n}" for t, n in DOCS]
    nav += ["      - Changelog: CHANGELOG.md", "      - Contributing: CONTRIBUTING.md"]
    nav += ["  - Model builder: docs/builder/index.html"]
    nav += ["  - Examples: examples/index.md"]
    nav += ["  - Notebooks:", "      - Overview: notebooks/index.md"]
    nav += [f"      - {json.dumps(t)}: {p}" for t, p in notebooks]
    nav += ["  - API reference:"]
    for section, mods in REFERENCE:
        nav += [f"      - {section}:"] + [f"          - {m}: reference/{m}.md" for m in mods]
    nav += ["  - Experiments:", "      - Overview: experiments/index.md"]
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
        "validation:",
        "  links: {unrecognized_links: warn, anchors: warn}",
        "plugins:",
        "  - search",
        "  - mkdocs-jupyter: {execute: false, include_source: true}",
        "  - mkdocstrings:",
        "      default_handler: python",
        f"      handlers: {{python: {json.dumps(HANDLER)}}}",  # JSON is valid YAML
        "nav:", *nav, ""]))
    # mkdocs 2.0 drops the plugin/theme system Material needs: stay on 1.x.
    if RTD_OUT or shutil.which("uvx") is None:  # mkdocs already installed (pip)
        cmd = [sys.executable, "-m", "mkdocs", "build", "--strict"]
    else:
        cmd = ["uvx", "--from", "mkdocs<2", "--with-requirements", str(REQUIREMENTS), "mkdocs",
               "build", "--strict"]
    return subprocess.call(cmd + ["-f", str(cfg)])


if __name__ == "__main__":
    sys.exit(main())
