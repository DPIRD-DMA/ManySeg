from __future__ import annotations

from collections.abc import Callable
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
