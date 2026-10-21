"""Write packages/manyseg/README.md (the PyPI page) from the repository README.

PyPI cannot resolve relative paths, so links become absolute GitHub URLs: images point at
raw.githubusercontent.com, everything else at the repository's blob view.

    python scripts/sync_readme.py          # regenerate
    python scripts/sync_readme.py --check  # exit 1 if out of date (CI)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
TARGET = ROOT / "packages" / "manyseg" / "README.md"
REPO = "DPIRD-DMA/ManySeg"
BLOB = f"https://github.com/{REPO}/blob/main/"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/"
HEADER = "<!-- Generated from the repository README by scripts/sync_readme.py; edit that. -->\n"


def _absolute(path: str, base: str) -> str:
    if re.match(r"[a-z][a-z0-9+.-]*:|#|/", path):
        return path
    return base + path


def render(text: str) -> str:
    text = re.sub(r'\b(src|srcset)="([^"]+)"', lambda m: f'{m[1]}="{_absolute(m[2], RAW)}"', text)
    text = re.sub(r'\bhref="([^"]+)"', lambda m: f'href="{_absolute(m[1], BLOB)}"', text)
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)",
                  lambda m: f"{m[1]}[{m[2]}]({_absolute(m[3], RAW if m[1] else BLOB)})", text)
    return HEADER + text


def main() -> int:
    expected = render(SOURCE.read_text(encoding="utf-8"))
    if "--check" in sys.argv[1:]:
        if TARGET.read_text(encoding="utf-8") != expected:
            print(f"{TARGET.relative_to(ROOT)} is out of date: run python scripts/sync_readme.py")
            return 1
        return 0
    TARGET.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
