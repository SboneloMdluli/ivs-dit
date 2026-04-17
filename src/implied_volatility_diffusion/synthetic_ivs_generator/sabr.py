"""SABR model: Hagan lognormal (Black) implied volatility and per-slice calibration.

Reference: Hagan, Kumar, Lesniewski, Woodward, "Managing Smile Risk", Wilmott (2002).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import optimize


def sabr_hagan_lognormal_iv(
    forward: float,
    strike: float,
    tau: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Return SABR lognormal implied volatility (Hagan 2002).

    Args:
        forward: Forward level at expiry ``tau``.
        strike: Strike.
        tau: Year fraction.
        alpha: SABR initial volatility level (>= 0).
        beta: CEV exponent in [0, 1] typically.
        rho: Correlation between asset and volatility.
        nu: Volatility of volatility (>= 0).
    """
    if strike <= 0.0 or forward <= 0.0 or tau <= 0.0:
        return 0.0
    if alpha <= 0.0 or nu < 0.0 or not (-0.9999 <= rho <= 0.9999):
        return float("nan")

    k = float(strike)
    f = float(forward)
    t = float(tau)
    one_m_beta = 1.0 - beta
    log_fk = math.log(f / k)
    fk_beta = (f * k) ** one_m_beta
    if fk_beta <= 0.0:
        return float("nan")

    a = (one_m_beta**2) * (alpha**2) / (24.0 * fk_beta)
    b = 0.25 * rho * beta * nu * alpha / math.sqrt(fk_beta)
    c = ((2.0 - 3.0 * rho * rho) * (nu**2)) / 24.0
    d = math.sqrt(fk_beta)
    v = (one_m_beta**2) * (log_fk**2) / 24.0
    w = (one_m_beta**4) * (log_fk**4) / 1920.0
    z = nu * d * log_fk / alpha

    eps = 1e-7
    base = alpha * (1.0 + (a + b + c) * t) / (d * (1.0 + v + w))
    if abs(z) <= eps:
        return float(max(base, 1e-16))

    xz = _x(rho, z)
    if not np.isfinite(xz) or abs(xz) < 1e-16:
        return float(max(base, 1e-16))
    return float(max(base * (z / xz), 1e-16))


def _x(rho: float, z: float) -> float:
    """Helper ``x(rho, z)`` used in Hagan 2002."""
    a = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 0.0)) + z - rho
    b = 1.0 - rho
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    return math.log(a / b)


def sabr_hagan_lognormal_iv_array(
    forward: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Element-wise Hagan IV for broadcastable ``forward``, ``strike``, ``tau``."""
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(tau, dtype=float)
    out = np.empty(np.broadcast_shapes(f.shape, k.shape, t.shape), dtype=float)
    it = np.nditer(
        [f, k, t, out],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
    )
    for fv, kv, tv, ov in it:
        ov[...] = sabr_hagan_lognormal_iv(
            float(fv), float(kv), float(tv), alpha, beta, rho, nu
        )
    return out


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
    """Calibrate ``(alpha, rho, nu)`` to one expiry smile.

    Calibration uses bounded nonlinear least-squares on SABR IV residuals.

    Args:
        forward: Forward for this slice.
        tau: Time to expiry (years).
        strikes: One-dimensional positive strikes.
        market_ivs: One-dimensional market implied vols.
        beta: Fixed SABR beta (common choices: 0, 0.5, 1 for indices).
        initial_guess: Optional initial guess ``(alpha, rho, nu)``.
        iv_floor: Ignore quotes with IV below this threshold.

    Returns:
        Tuple ``(alpha, rho, nu, result)`` where ``result`` is the SciPy optimizer output.
    """
    strikes_arr = np.asarray(strikes, dtype=float).ravel()
    market_iv_arr = np.asarray(market_ivs, dtype=float).ravel()
    if strikes_arr.shape != market_iv_arr.shape:
        raise ValueError("strikes and market_ivs must have the same shape")
    if tau <= 0.0 or forward <= 0.0:
        raise ValueError("forward and tau must be positive for calibration")
    valid = (
        np.isfinite(strikes_arr)
        & np.isfinite(market_iv_arr)
        & (strikes_arr > 0.0)
        & (market_iv_arr > iv_floor)
    )
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
                sabr_hagan_lognormal_iv(
                    forward,
                    float(k),
                    tau,
                    alpha_x,
                    beta,
                    rho_x,
                    nu_x,
                )
                for k in fit_strikes
            ],
            dtype=float,
        )
        bad = ~np.isfinite(model)
        err = model - fit_ivs
        # Penalize invalid formula points strongly to steer optimizer back in-domain.
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
