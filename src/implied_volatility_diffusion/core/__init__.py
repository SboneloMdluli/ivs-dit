"""Core numerics and domain types (NumPy-first).

Submodules:

* :mod:`.protocols` — :class:`VolModel`, :class:`ImpliedVolInverter`, ...
* :mod:`.grid` — ``grid_axes``, ``build_grid_axis``.
* :mod:`.lhs` — Latin Hypercube sampling over config ranges.
* :mod:`.types` — :class:`SurfaceBatch`, :class:`MarketState`.
"""

from implied_volatility_diffusion.core.grid import build_grid_axis, grid_axes
from implied_volatility_diffusion.core.lhs import (
    lhs_params_from_config,
    lhs_params_multi_batch_from_config,
)
from implied_volatility_diffusion.core.protocols import (
    ImpliedVolInverter,
    ModelCallPricer,
    VolModel,
)
from implied_volatility_diffusion.core.types import MarketState, SurfaceBatch

__all__ = [
    "ImpliedVolInverter",
    "MarketState",
    "ModelCallPricer",
    "SurfaceBatch",
    "VolModel",
    "build_grid_axis",
    "grid_axes",
    "lhs_params_from_config",
    "lhs_params_multi_batch_from_config",
]
