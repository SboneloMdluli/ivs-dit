"""Historic-quotes SABR interpolation wrapper.

Glue between :func:`filter_day_for_surface` (cleaned parquet quotes on a
single ``quote_date``) and the per-expiry SABR calibration helpers in
:mod:`implied_volatility_diffusion.models.sabr.calibration`.

Given a single day's DataFrame with columns ``k`` (log-moneyness), ``tau``,
``iv``, ``vega``, ``underlying_last`` (and ``quote_date``), this module:

1. filters the day to a clean ``(k, tau, iv, vega)`` subset,
2. groups rows by distinct expiry ``tau``,
3. calibrates one SABR smile per expiry with fixed ``beta``, and
4. evaluates the Hagan lognormal SABR vol on the user-provided
   ``(k_grid, tau_grid)`` evaluation grid with
   :func:`implied_vol_surface_from_calibrated_slices`
   (moneyness axis is ``m = exp(k_grid)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import (
    filter_day_for_surface,
)
from implied_volatility_diffusion.models.sabr.calibration import (
    calibrate_params_for_expiries,
    implied_vol_surface_from_calibrated_slices,
)


@dataclass(frozen=True)
class HistoricalSabrSurface:
    """Output of :func:`build_historical_sabr_surface`.

    Attributes:
        surface: Dense IV surface, shape ``(len(k_grid), len(tau_grid))``.
        day_sub: Filtered DataFrame actually used for calibration.
        spot: Spot used for forward and strike reconstruction.
        expiry_taus: Expiry tenors (years) calibrated, shape ``(n_expiries,)``.
        calibrated_params: Per-expiry ``(alpha, rho, nu)``, shape ``(n_expiries, 3)``.
    """

    surface: np.ndarray
    day_sub: pd.DataFrame
    spot: float
    expiry_taus: np.ndarray
    calibrated_params: np.ndarray


def build_historical_sabr_surface(
    day_df: pd.DataFrame,
    k_grid: np.ndarray,
    tau_grid: np.ndarray,
    *,
    r: float,
    q: float,
    beta: float,
    k_range: tuple[float, float] = (-0.5, 0.5),
    tau_range: tuple[float, float] = (0.01, 2.0),
    iv_range: tuple[float, float] = (0.01, 2.0),
    min_points_per_expiry: int = 3,
) -> HistoricalSabrSurface:
    """Calibrate SABR smiles per expiry and evaluate on ``(k_grid, tau_grid)``.

    Args:
        day_df: Cleaned quotes for a single ``quote_date`` with columns
            ``k``, ``tau``, ``iv``, ``vega``, ``underlying_last``.
        k_grid: Log-moneyness evaluation axis.
        tau_grid: Maturity evaluation axis (years).
        r: Risk-free rate used for the SABR forward.
        q: Dividend yield used for the SABR forward.
        beta: SABR elasticity held fixed during calibration.
        k_range, tau_range, iv_range: Passed through to
            :func:`filter_day_for_surface`.
        min_points_per_expiry: Minimum number of valid strikes required to
            attempt a SABR calibration on a given expiry slice.

    Returns:
        :class:`HistoricalSabrSurface`.
    """
    day_sub = filter_day_for_surface(day_df, k_range=k_range, tau_range=tau_range, iv_range=iv_range)
    if day_sub.empty:
        raise ValueError("no rows survive filter_day_for_surface for this day")
    if "underlying_last" not in day_sub.columns:
        raise ValueError("day_df must include 'underlying_last' for spot reconstruction")

    spot = float(np.nanmedian(day_sub["underlying_last"].to_numpy(dtype=float)))
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("invalid spot derived from day_df['underlying_last']")

    expiry_taus: list[float] = []
    strikes_per_expiry: list[np.ndarray] = []
    ivs_per_expiry: list[np.ndarray] = []

    for tau_value, grp in day_sub.groupby("tau", sort=True):
        k_vals = grp["k"].to_numpy(dtype=float)
        iv_vals = grp["iv"].to_numpy(dtype=float)
        strikes = np.exp(k_vals) * spot
        mask = np.isfinite(strikes) & np.isfinite(iv_vals) & (strikes > 0.0) & (iv_vals > 0.0)
        if int(mask.sum()) < int(min_points_per_expiry):
            continue
        expiry_taus.append(float(tau_value))
        strikes_per_expiry.append(strikes[mask])
        ivs_per_expiry.append(iv_vals[mask])

    if len(expiry_taus) < 2:
        raise ValueError(
            f"need at least two expiries with >= {min_points_per_expiry} valid points to build a SABR surface"
        )

    expiry_tau_arr = np.asarray(expiry_taus, dtype=float)
    calibrated_params, _ = calibrate_params_for_expiries(
        spot,
        float(r),
        float(q),
        expiry_tau_arr,
        strikes_per_expiry,
        ivs_per_expiry,
        beta=float(beta),
    )

    k_arr = np.asarray(k_grid, dtype=float).ravel()
    tau_arr = np.asarray(tau_grid, dtype=float).ravel()
    moneyness = np.exp(k_arr)

    surface = implied_vol_surface_from_calibrated_slices(
        spot,
        float(r),
        float(q),
        moneyness,
        tau_arr,
        expiry_tau_arr,
        calibrated_params,
        beta=float(beta),
    )

    return HistoricalSabrSurface(
        surface=surface,
        day_sub=day_sub,
        spot=spot,
        expiry_taus=expiry_tau_arr,
        calibrated_params=calibrated_params,
    )


__all__ = ["HistoricalSabrSurface", "build_historical_sabr_surface"]
