"""Heston stochastic-volatility model (NumPy backend).

The implementation is organised as:

* :mod:`.cos` — Fang-Oosterlee COS pricer (vectorized over strikes).
* :mod:`.simulation` — scalar full-truncation Milstein discretization.
* :mod:`.model` — :class:`HestonModel` bundling the above behind
  :class:`implied_volatility_diffusion.core.protocols.VolModel`.
"""

from implied_volatility_diffusion.models.heston.heston_cos import _heston_cf as heston_cf
from implied_volatility_diffusion.models.heston.heston_cos import (
    heston_call_cos,
)
from implied_volatility_diffusion.models.heston.model import HESTON_PARAM_ORDER, HestonModel
from implied_volatility_diffusion.models.heston.simulation import (
    feller_index,
    is_feller_satisfied,
    milstein_step,
)

__all__ = [
    "HESTON_PARAM_ORDER",
    "HestonModel",
    "feller_index",
    "heston_call_cos",
    "heston_cf",
    "is_feller_satisfied",
    "milstein_step",
]
