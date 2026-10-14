from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache
from typing import Any


class Registry(dict):
    """Name -> constructor mapping with helpful errors."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def register(self, name: str) -> Callable[[Any], Any]:
        def deco(obj):
            if name in self:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self[name] = obj
            return obj

        return deco

    def get_or_raise(self, name: str):
        if name not in self:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {sorted(self)}")
        return self[name]


@cache
def reserved_names() -> frozenset[str]:
    """Attribute names of ``nn.ModuleDict`` / ``nn.ParameterDict`` (``half``, ``train``,
    ``to``, ``keys``, ...): they cannot be used as keys of those containers."""
    from torch import nn

    return frozenset(dir(nn.ModuleDict())) | frozenset(dir(nn.ParameterDict()))


def check_module_names(kind: str, names: Iterable[str]) -> None:
    """Reject user-chosen names (heads, inputs, metadata fields) that would become
    ``ModuleDict`` keys but collide with ``nn.Module`` attributes, contain ``.`` or are empty."""
    for name in names:
        if (not isinstance(name, str) or not name or "." in name or name.startswith("_")
                or name in reserved_names()):
            public = sorted(n for n in reserved_names() if not n.startswith("_"))
            raise ValueError(
                f"{kind} name {name!r} is not allowed: names must be non-empty, contain no '.', "
                f"not start with '_' and not be one of the reserved nn.Module attribute names "
                f"{public}")
