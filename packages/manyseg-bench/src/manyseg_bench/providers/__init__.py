"""Provider registry. Providers import their library lazily, so manyseg-bench works with any
subset of {manyseg, smp, fastai} installed."""

from .base import BuiltModel, ModelSpec, Provider, fingerprint_module
from .fastai_provider import FastaiProvider
from .manyseg_provider import ManySegProvider
from .smp_provider import SMPProvider

PROVIDERS: dict[str, type[Provider]] = {
    p.name: p for p in (ManySegProvider, SMPProvider, FastaiProvider)
}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; available: {sorted(PROVIDERS)}")
    return PROVIDERS[name]()


def register_provider(cls: type[Provider]) -> type[Provider]:
    PROVIDERS[cls.name] = cls
    return cls


__all__ = ["PROVIDERS", "BuiltModel", "ModelSpec", "Provider", "fingerprint_module",
           "get_provider", "register_provider"]
