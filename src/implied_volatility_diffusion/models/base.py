"""Registry of named volatility-model factories."""

from typing import Any, Callable, Iterable, Mapping

from implied_volatility_diffusion.core.protocols import VolModel

ModelFactory = Callable[[Mapping[str, Any]], VolModel]

_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register a model factory under ``name``.

    ``factory(cfg)`` must return an object implementing :class:`VolModel`.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("model name must be non-empty")
    _REGISTRY[key] = factory


def get_model_factory(name: str) -> ModelFactory:
    """Look up a registered model factory by name."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown model {name!r}; registered: {available}")
    return _REGISTRY[key]


def iter_model_names() -> Iterable[str]:
    """Yield the names of all registered models (sorted)."""
    return sorted(_REGISTRY)
