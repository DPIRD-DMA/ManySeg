"""Griffe extension for the API reference: ManySeg's docstrings are Google style (``Args:``,
``Returns:``) but use a few reStructuredText constructs that Markdown does not know. Before
mkdocstrings renders a docstring, this rewrites:

- Sphinx roles (``:func:`decode_scene```, ``:class:`~manyseg.weights.Weights```, ...) into
  mkdocstrings cross-references when the target is a documented ManySeg object, else into inline
  code;
- literal blocks (a paragraph ending in ``::`` followed by an indented block) into fenced code;
- directives (``.. warning:: text``) into Material admonitions;
- the first paragraph into one line, so summary tables show the whole first sentence.

Double-backtick literals need no change: Markdown renders ````x```` as inline code too.
Loaded by the config scripts/build_docs.py writes; the conversion itself is plain Python
(:func:`convert`) so it can be tested without mkdocs.
"""

from __future__ import annotations

import re
from collections.abc import Callable

ROLE = re.compile(r":(?:py:)?(?:func|class|mod|meth|attr|data|obj|exc|const):`([^`]+)`")
DIRECTIVE = re.compile(r"^(\s*)\.\. (\w+)::\s*(.*)$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _roles(text: str, link: Callable[[str], str | None]) -> str:
    def sub(m: re.Match) -> str:
        target = m.group(1).strip()
        shown = target.lstrip("~.")
        if target.startswith("~"):
            shown = shown.rsplit(".", 1)[-1]
        path = link(target.lstrip("~."))
        return f"[`{shown}`][{path}]" if path else f"`{shown}`"

    # reST's escaped space joins a suffix to a literal (``Polygon``\\ s); Markdown needs none
    return ROLE.sub(sub, text).replace("`\\ ", "`")


def _blocks(lines: list[str]) -> list[str]:
    """Literal blocks to fenced code and directives to admonitions, line by line."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        d = DIRECTIVE.match(line)
        if d:  # ".. warning:: first line" + lines indented deeper than the directive
            ind, kind, first = d.groups()
            body = [first] if first else []
            i += 1
            while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > len(ind)):
                body.append(lines[i].strip())
                i += 1
            while body and not body[-1]:
                body.pop()
            out.append(f"{ind}!!! {kind.lower()}")
            out += [f"{ind}    {b}" if b else "" for b in body]
            out.append("")
            continue
        if line.rstrip().endswith("::") and not line.strip().startswith(".."):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            base = _indent(line)
            if j < len(lines) and _indent(lines[j]) > base:
                block, k = [], j
                while k < len(lines) and (not lines[k].strip() or _indent(lines[k]) > base):
                    block.append(lines[k])
                    k += 1
                while block and not block[-1].strip():
                    block.pop()
                strip = min(_indent(b) for b in block if b.strip())
                head = line.rstrip()[:-2].rstrip()
                if head.strip():
                    out.append(head + ":")
                pad = " " * base
                out += ["", pad + "```python"]
                out += [pad + b[strip:] if b.strip() else "" for b in block]
                out += [pad + "```", ""]
                i = k
                continue
        out.append(line)
        i += 1
    return out


SECTION = re.compile(r"^[A-Z][\w ]*:$")


def _summary(lines: list[str]) -> list[str]:
    """Join the first paragraph into one line: Markdown renders it the same, and the summary
    tables (which show a docstring's first line) then show the whole first sentence."""
    n = 0
    while n < len(lines) and lines[n].strip() and not SECTION.match(lines[n].strip()) \
            and not lines[n].startswith((" ", "```", "!!!", "- ", "* ")):
        n += 1
    return [" ".join(x.strip() for x in lines[:n])] + lines[n:] if n > 1 else lines


def convert(text: str, link: Callable[[str], str | None] = lambda _: None) -> str:
    """One docstring from reST constructs to Markdown; ``link(target)`` returns the identifier to
    cross-reference for a role's target, or None for inline code."""
    return "\n".join(_summary(_blocks(_roles(text, link).split("\n"))))


def _lookup(root, path: str):
    """The object at dotted ``path`` under the package ``root``, following re-exports; None if
    it does not exist."""
    parts = path.split(".")
    if parts[0] != root.name:
        return None
    obj = root
    for name in parts[1:]:
        try:
            obj = obj.members[name]
        except Exception:  # missing member, or an alias that does not resolve
            return None
    return obj


def _canonical(obj) -> str | None:
    try:
        return obj.canonical_path
    except Exception:  # an alias to something outside the package
        return None


def _shown(page, name: str, member) -> bool:
    """Whether a module page lists ``member``: in ``__all__`` when the module has one, else
    public and defined there (not imported)."""
    exports = page.exports
    if exports is not None:
        return name in exports
    return not name.startswith("_") and not member.is_alias


def rendered(root, pages) -> dict[str, str]:
    """canonical path -> identifier on the reference pages, for every object the pages show:
    the page modules, their listed members and the public members of listed classes."""
    index: dict[str, str] = {}

    def add(obj, ident: str) -> None:
        canonical = _canonical(obj)
        if canonical is None:
            return
        index.setdefault(canonical, ident)
        index.setdefault(ident, ident)
        try:
            kind = obj.kind.value
            members = obj.members if kind == "class" else {}
        except Exception:
            return
        for name, m in members.items():
            if not name.startswith("_") and not m.is_alias:
                add(m, f"{ident}.{name}")

    for page_path in pages:
        page = _lookup(root, page_path)
        if page is None:
            continue
        index[page_path] = page_path
        for name, member in page.members.items():
            if _shown(page, name, member) and not (member.is_alias and _canonical(member) is None):
                add(member, f"{page_path}.{name}")
    return index


def _plumbing(cls, name: str, member) -> bool:
    """An undocumented class member the reference leaves out: ``forward`` / ``extra_repr`` (the
    torch.nn.Module protocol), special methods, and attributes assigned inside a method (the
    layers a module's ``__init__`` builds), but not fields declared in the class body."""
    if name in ("forward", "extra_repr"):
        return True
    if name.startswith("__") and name.endswith("__"):
        return name not in ("__init__", "__call__")  # the constructor and call signatures
    if not member.is_attribute:
        return False
    return any(m.is_function and m.lineno <= member.lineno <= (m.endlineno or m.lineno)
               for m in cls.members.values() if not m.is_alias)


try:
    import griffe
except ImportError:  # the conversion is usable (and testable) without griffe
    griffe = None

if griffe is not None:

    class RstToMarkdown(griffe.Extension):
        """Prepare the loaded package for the reference pages: rewrite every docstring with
        :func:`convert`, export functions that share their submodule's name, and drop
        undocumented torch plumbing (:func:`_plumbing`)."""

        def __init__(self, pages: list[str] | None = None) -> None:
            super().__init__()
            self.pages = set(pages or ())

        def on_package(self, *, pkg, **kwargs) -> None:
            detached = self._unshadow(pkg)
            self.index = rendered(pkg, self.pages)
            for obj in [pkg, *detached]:
                self._walk(pkg, obj)

        def _unshadow(self, mod) -> list:
            """``from .road_targets import road_targets`` in an ``__init__`` rebinds the name to
            the function at runtime, but griffe keeps the submodule: point the exported name
            at the function, as Python does. Returns the submodules taken out of the tree."""
            detached = []
            for name, member in list(mod.members.items()):
                if member.is_alias or not member.is_module:
                    continue
                inner = member.members.get(name)
                if (mod.exports and name in mod.exports and inner is not None
                        and not inner.is_module):
                    mod.set_member(name, griffe.Alias(name, inner, parent=mod))
                    detached.append(member)
                detached += self._unshadow(member)
            return detached

        def _walk(self, root, obj) -> None:
            if obj.docstring is not None:
                obj.docstring.value = convert(obj.docstring.value,
                                              lambda t, o=obj: self._link(root, o, t))
            for name, member in list(obj.members.items()):
                if member.is_alias:
                    continue
                if obj.is_class and member.docstring is None and _plumbing(obj, name, member):
                    del obj.members[name]
                    continue
                self._walk(root, member)

        def _link(self, root, obj, target: str) -> str | None:
            scopes, scope = [], obj
            while scope is not None:  # the object, its class, its module, parent packages
                scopes.append(scope.path)
                scope = scope.parent
            for path in [target] + [f"{s}.{target}" for s in scopes]:
                found = _lookup(root, path)
                if found is not None:
                    return self.index.get(_canonical(found) or "")
            return None
