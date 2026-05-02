"""Model-agnostic surface builders on top of :class:`VolModel`.

These helpers take any object implementing the :class:`VolModel` protocol
and return a :class:`SurfaceBatch`. They remove the need for recipe-specific
orchestrators to live in the same module as model-specific maths.
"""

from typing import Any, Mapping

import numpy as np

from implied_volatility_diffusion.core.grid import grid_axes
from implied_volatility_diffusion.core.protocols import VolModel
from implied_volatility_diffusion.core.types import MarketState, SurfaceBatch


def _market_state(cfg: Mapping[str, Any]) -> MarketState:
    market = cfg["market"]
    return MarketState(
        spot=float(market["spot"]),
        rate=float(market.get("r", market.get("rate", market.get("risk_free_rate", 0.0)))),
        dividend_yield=float(market.get("dividend_yield", 0.0)),
    )


def build_surfaces(
    model: VolModel,
    cfg: Mapping[str, Any],
    params: np.ndarray,
    *,
    spot_override: float | None = None,
    inst_var_override: np.ndarray | None = None,
) -> SurfaceBatch:
    """Build a :class:`SurfaceBatch` by invoking ``model.implied_vol_surface``.

    The ``market`` block of ``cfg`` supplies spot / rate / dividend yield;
    ``rate`` may also come from a per-row column inside ``params`` depending
    on the model (e.g. Heston uses ``params[:, -1]``).
    """
    params_t = np.asarray(params, dtype=float)
    m_t, tau_t = grid_axes(cfg)
    market = _market_state(cfg)
    spot = float(spot_override if spot_override is not None else market.spot)

    iv = model.implied_vol_surface(
        params_t,
        spot=spot,
        moneyness=m_t,
        tau=tau_t,
        rate=market.rate,
        dividend_yield=market.dividend_yield,
        **({"inst_var": inst_var_override} if inst_var_override is not None else {}),
    )
    return SurfaceBatch(params=params_t, moneyness=m_t, tau=tau_t, iv=iv)


def build_surface(
    model: VolModel,
    cfg: Mapping[str, Any],
    params: np.ndarray,
    *,
    spot_override: float | None = None,
    inst_var_override: float | None = None,
) -> SurfaceBatch:
    """Convenience wrapper for a single parameter vector."""
    params_t = np.asarray(params, dtype=float)
    if params_t.ndim == 1:
        params_t = params_t.reshape(1, -1)
    iv_override = None
    if inst_var_override is not None:
        iv_override = np.array([float(inst_var_override)], dtype=float)
    return build_surfaces(
        model,
        cfg,
        params_t,
        spot_override=spot_override,
        inst_var_override=iv_override,
    )
