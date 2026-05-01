"""SABR calibration and per-expiry market-fit helpers."""

from typing import Any

import numpy as np
import QuantLib as ql
from scipy import optimize


def forward_from_spot(spot: float, tau: float, r: float, q: float) -> float:
    """Risk-neutral forward for tenor ``tau``."""
    return float(spot * np.exp((r - q) * tau))


def _sabr_black_vol(
    forward: float,
    strike: float,
    tau: float,
    *,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """QuantLib SABR Black volatility with safe fallback."""
    try:
        return float(
            ql.sabrVolatility(
                strike,
                forward,
                tau,
                alpha,
                beta,
                nu,
                rho,
            )
        )
    except RuntimeError:
        return float("nan")


def calibrate_sabr_to_implied_vols(
    forward: float,
    tau: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    *,
    beta: float,
    initial_guess: tuple[float, float, float] | None = None,
    iv_floor: float = 1e-6,
) -> tuple[float, float, float, Any]:
    """Bounded nonlinear LS calibration of ``(alpha, rho, nu)`` to one expiry smile."""
    strikes_arr = np.asarray(strikes, dtype=float).ravel()
    market_iv_arr = np.asarray(market_ivs, dtype=float).ravel()
    if strikes_arr.shape != market_iv_arr.shape:
        raise ValueError("strikes and market_ivs must have the same shape")
    if tau <= 0.0 or forward <= 0.0:
        raise ValueError("forward and tau must be positive for calibration")
    valid = np.isfinite(strikes_arr) & np.isfinite(market_iv_arr) & (strikes_arr > 0.0) & (market_iv_arr > iv_floor)
    fit_strikes = strikes_arr[valid]
    fit_ivs = market_iv_arr[valid]
    if fit_strikes.size < 3:
        raise ValueError("need at least three valid strikes to calibrate SABR")

    if initial_guess is None:
        iv_atm = float(fit_ivs[np.argmin(np.abs(fit_strikes - forward))])
        alpha0 = max(iv_atm * (forward ** max(1.0 - beta, 0.0)), 1e-4)
        rho0, nu0 = 0.0, 0.3
    else:
        alpha0, rho0, nu0 = initial_guess

    x0 = np.array(
        [
            max(alpha0, 1e-4),
            max(min(rho0, 0.9999), -0.9999),
            max(nu0, 1e-4),
        ],
        dtype=float,
    )

    def residuals(x: np.ndarray) -> np.ndarray:
        alpha_x, rho_x, nu_x = float(x[0]), float(x[1]), float(x[2])
        model = np.array(
            [
                _sabr_black_vol(
                    forward=forward,
                    strike=float(k),
                    tau=tau,
                    alpha=alpha_x,
                    beta=beta,
                    rho=rho_x,
                    nu=nu_x,
                )
                for k in fit_strikes
            ],
            dtype=float,
        )
        bad = ~np.isfinite(model)
        err = model - fit_ivs
        err[bad] = 10.0
        return err

    result = optimize.least_squares(
        residuals,
        x0,
        method="trf",
        bounds=([1e-4, -0.9999, 1e-4], [np.inf, 0.9999, np.inf]),
        max_nfev=500,
    )
    alpha, rho, nu = float(result.x[0]), float(result.x[1]), float(result.x[2])
    return alpha, rho, nu, result


def calibrate_params_for_expiries(
    spot: float,
    r: float,
    q: float,
    expiry_taus: np.ndarray,
    strikes_per_expiry: list[np.ndarray],
    ivs_per_expiry: list[np.ndarray],
    *,
    beta: float,
) -> tuple[np.ndarray, list[Any]]:
    """Calibrate one SABR smile per expiry; returns ``(params, details)``."""
    taus = np.asarray(expiry_taus, dtype=float).ravel()
    if not (len(strikes_per_expiry) == len(ivs_per_expiry) == int(taus.size)):
        raise ValueError("expiry_taus, strikes_per_expiry, and ivs_per_expiry lengths must match")
    rows: list[list[float]] = []
    details: list[Any] = []
    for tau, expiry_strikes, expiry_ivs in zip(taus, strikes_per_expiry, ivs_per_expiry, strict=True):
        fwd = forward_from_spot(spot, float(tau), r, q)
        alpha, rho, nu, res = calibrate_sabr_to_implied_vols(
            fwd,
            float(tau),
            expiry_strikes,
            expiry_ivs,
            beta=beta,
        )
        rows.append([alpha, rho, nu])
        details.append(res)
    return np.asarray(rows, dtype=float), details


def implied_vol_surface_from_calibrated_slices(
    spot: float,
    r: float,
    q: float,
    moneyness: np.ndarray,
    tau_axis: np.ndarray,
    expiry_taus: np.ndarray,
    calibrated_params: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Fill a moneyness x maturity IV grid via interpolated/extrapolated SABR params."""
    m = np.asarray(moneyness, dtype=float).ravel()
    taus_grid = np.asarray(tau_axis, dtype=float).ravel()
    taus_exp = np.asarray(expiry_taus, dtype=float).ravel()
    p = np.asarray(calibrated_params, dtype=float)
    if p.shape[0] != taus_exp.size or p.shape[1] != 3:
        raise ValueError("calibrated_params must have shape (n_expiries, 3)")
    if taus_exp.size < 2:
        raise ValueError("need at least two calibrated expiries for interpolation/extrapolation")

    order = np.argsort(taus_exp)
    taus_exp_sorted = taus_exp[order]
    p_sorted = p[order]
    alpha_exp = p_sorted[:, 0]
    rho_exp = p_sorted[:, 1]
    nu_exp = p_sorted[:, 2]

    # Interpolate in maturity and linearly extrapolate using edge slopes.
    alpha_tau = np.interp(taus_grid, taus_exp_sorted, alpha_exp)
    rho_tau = np.interp(taus_grid, taus_exp_sorted, rho_exp)
    nu_tau = np.interp(taus_grid, taus_exp_sorted, nu_exp)

    lo_mask = taus_grid < taus_exp_sorted[0]
    hi_mask = taus_grid > taus_exp_sorted[-1]
    if np.any(lo_mask):
        dt = max(taus_exp_sorted[1] - taus_exp_sorted[0], 1e-12)
        alpha_slope = (alpha_exp[1] - alpha_exp[0]) / dt
        rho_slope = (rho_exp[1] - rho_exp[0]) / dt
        nu_slope = (nu_exp[1] - nu_exp[0]) / dt
        d = taus_grid[lo_mask] - taus_exp_sorted[0]
        alpha_tau[lo_mask] = alpha_exp[0] + alpha_slope * d
        rho_tau[lo_mask] = rho_exp[0] + rho_slope * d
        nu_tau[lo_mask] = nu_exp[0] + nu_slope * d
    if np.any(hi_mask):
        dt = max(taus_exp_sorted[-1] - taus_exp_sorted[-2], 1e-12)
        alpha_slope = (alpha_exp[-1] - alpha_exp[-2]) / dt
        rho_slope = (rho_exp[-1] - rho_exp[-2]) / dt
        nu_slope = (nu_exp[-1] - nu_exp[-2]) / dt
        d = taus_grid[hi_mask] - taus_exp_sorted[-1]
        alpha_tau[hi_mask] = alpha_exp[-1] + alpha_slope * d
        rho_tau[hi_mask] = rho_exp[-1] + rho_slope * d
        nu_tau[hi_mask] = nu_exp[-1] + nu_slope * d

    alpha_tau = np.maximum(alpha_tau, 1e-8)
    nu_tau = np.maximum(nu_tau, 1e-8)
    rho_tau = np.clip(rho_tau, -0.9999, 0.9999)

    out = np.empty((m.size, taus_grid.size), dtype=float)
    for j, tj in enumerate(taus_grid):
        if tj <= 0:
            out[:, j] = np.nan
            continue
        alpha = float(alpha_tau[j])
        rho = float(rho_tau[j])
        nu = float(nu_tau[j])
        fwd = forward_from_spot(spot, float(tj), r, q)
        strikes = m * spot
        for i, kk in enumerate(strikes):
            out[i, j] = _sabr_black_vol(
                forward=fwd,
                strike=float(kk),
                tau=float(tj),
                alpha=alpha,
                beta=beta,
                rho=rho,
                nu=nu,
            )
    return out
