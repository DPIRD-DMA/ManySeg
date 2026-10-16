"""Build the documentation site (MkDocs + Material) from the repo's existing Markdown.

Assembles README.md (home), docs/*.md, the model builder page (docs/builder), examples/README.md
and every experiments/E*/README.md
into build/site_src (gitignored), writes a generated MkDocs config with the nav, and builds
static HTML into build/site. No new runtime dependency: run with
    uvx --with mkdocs-material mkdocs --version   # once, to check the tool
    uv run python scripts/build_docs.py            # or: python scripts/build_docs.py
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
OUT = Path(RTD_OUT) / "html" if RTD_OUT else ROOT / "build" / "site"

DOCS = [("Architecture review", "architecture_review.md"), ("API design", "api_design.md"),
        ("Benchmark design", "benchmark_design.md"), ("Testing strategy", "testing_strategy.md"),
        ("Backbone compatibility", "backbone_compatibility.md"), ("Roadmap", "roadmap.md")]


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
    for _, name in DOCS:
        shutil.copy(ROOT / "docs" / name, SRC / "docs" / name)
    (SRC / "examples").mkdir()
    shutil.copy(ROOT / "examples" / "README.md", SRC / "examples" / "index.md")
    exps = []
    for d in sorted((ROOT / "experiments").glob("E*/README.md")):
        dst = SRC / "experiments" / f"{d.parent.name}.md"
        shutil.copy(d, dst)
        exps.append((f"{d.parent.name.split('_')[0]}: {_title(d).split(':', 1)[-1].strip()}",
                     f"experiments/{dst.name}"))
    nav = ["  - Home: index.md", "  - Documentation:"]
    nav += [f"      - {t}: docs/{n}" for t, n in DOCS]
    nav += ["  - Model builder: docs/builder/index.html"]
    nav += ["  - Examples: examples/index.md", "  - Experiments:"]
    nav += [f"      - {json.dumps(t)}: {p}" for t, p in exps]  # JSON strings are valid YAML
    cfg = ROOT / "build" / "mkdocs.gen.yml"
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
        "markdown_extensions: [tables, admonition, toc, pymdownx.superfences]",
        "nav:", *nav, ""]))
    # mkdocs 2.0 drops the plugin/theme system Material needs: stay on 1.x.
    if RTD_OUT or shutil.which("uvx") is None:  # mkdocs already installed (pip)
        cmd = [sys.executable, "-m", "mkdocs", "build", "-q"]
    else:
        cmd = ["uvx", "--from", "mkdocs<2", "--with", "mkdocs-material<10", "mkdocs", "build",
               "-q"]
    return subprocess.call(cmd + ["-f", str(cfg)])


if __name__ == "__main__":
    sys.exit(main())
