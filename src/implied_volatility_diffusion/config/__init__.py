"""YAML config loading, deep merge, and typed recipe objects.

Public, stable surface area:

* :func:`load_config`, :func:`merge_config`, :func:`merge_config_files`
* :class:`SurfaceRecipe` — a thin dataclass wrapper around a nested config dict
  that validates the ``market`` / ``grid`` / model-ranges / ``lhs`` keys used
  by the synthetic-surface recipes.
"""

from implied_volatility_diffusion.config.loader import (
    load_config,
    merge_config,
    merge_config_files,
)
from implied_volatility_diffusion.config.schema import SurfaceRecipe

__all__ = [
    "SurfaceRecipe",
    "load_config",
    "merge_config",
    "merge_config_files",
]
