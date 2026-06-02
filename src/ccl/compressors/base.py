"""The Compressor interface and registry.

A compressor takes text and returns shorter text. Registering one with
``@register`` makes it appear automatically in the benchmark, that pluggability
is the backbone of this as a *continuous* research lab: adding a method is one
class and one decorator.

Categories follow the survey taxonomy (arXiv 2410.12388): the compressors here
are all "hard-prompt" methods (they operate on the surface text). Soft-prompt
methods, which compress into learned vectors and require training, are surveyed
in LITERATURE.md rather than implemented.
"""

from __future__ import annotations

_REGISTRY: dict[str, Compressor] = {}


class Compressor:
    """Base class. Subclasses set `name`/`category` and implement `compress`."""

    name: str = "base"
    category: str = "uncategorized"
    requires_extra: str | None = None  # optional-dependency tag, for reporting

    def compress(self, text: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this compressor can run now (optional deps installed, etc.)."""
        return True


def register(cls: type[Compressor]) -> type[Compressor]:
    """Class decorator: instantiate and add to the global registry."""
    instance = cls()
    if not instance.name or instance.name == "base":
        raise ValueError(f"{cls.__name__} must set a unique `name`.")
    if instance.name in _REGISTRY:
        raise ValueError(f"Duplicate compressor name: {instance.name}")
    _REGISTRY[instance.name] = instance
    return cls


def get(name: str) -> Compressor:
    return _REGISTRY[name]


def available() -> list[Compressor]:
    """All registered compressors, baseline first then alphabetical."""
    return sorted(_REGISTRY.values(), key=lambda c: (c.name != "identity", c.name))
