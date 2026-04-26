"""NumPy no-arbitrage checks for implied-vol surfaces."""

from __future__ import annotations

import numpy as np

from implied_volatility_diffusion.arbitrage_checks.report import ArbitrageReport
from implied_volatility_diffusion.pricing.black_scholes import bs_call_price


def _bs_call_grid(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Black-Scholes call prices on the ``(M, T)`` grid. NaN where inputs are invalid."""
    iv = np.asarray(iv, dtype=float)
    strikes = (moneyness * spot).reshape(-1, 1)  # (M, 1)
    tau_b = tau.reshape(1, -1)  # (1, T)
    safe_iv = np.where(np.isfinite(iv) & (iv > 0.0), iv, 1.0)
    safe_tau = np.where(tau_b > 0.0, tau_b, 1.0)
    price = bs_call_price(float(spot), strikes, safe_tau, float(rate), safe_iv, float(dividend_yield))
    invalid = ~np.isfinite(iv) | (iv <= 0.0) | np.broadcast_to(tau_b <= 0.0, iv.shape)
    return np.where(invalid, np.nan, price)


def check_iv_surface_arbitrage(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
) -> ArbitrageReport:
    """Check Roper / Gatheral no-arbitrage conditions for one IV surface."""
    iv_t = np.asarray(iv, dtype=float)
    if iv_t.ndim != 2:
        raise ValueError(f"iv must be 2D (n_moneyness, n_tau); got shape {tuple(iv_t.shape)}")
    m_t = np.asarray(moneyness, dtype=float).reshape(-1)
    t_t = np.asarray(tau, dtype=float).reshape(-1)
    if iv_t.shape != (m_t.size, t_t.size):
        raise ValueError(
            f"iv shape {tuple(iv_t.shape)} does not match (len(moneyness)={int(m_t.size)}, len(tau)={int(t_t.size)})"
        )
    if bool(np.any(np.diff(m_t) <= 0.0)):
        raise ValueError("moneyness must be strictly increasing")
    if bool(np.any(np.diff(t_t) <= 0.0)):
        raise ValueError("tau must be strictly increasing")

    c = _bs_call_grid(iv_t, m_t, t_t, spot=spot, rate=rate, dividend_yield=dividend_yield)
    k_t = m_t * float(spot)
    upper = float(spot) * np.exp(-float(dividend_yield) * t_t)
    lower = np.maximum(
        float(spot) * np.exp(-float(dividend_yield) * t_t).reshape(1, -1)
        - k_t.reshape(-1, 1) * np.exp(-float(rate) * t_t).reshape(1, -1),
        0.0,
    )
    finite = np.isfinite(c)

    upper_slack = upper.reshape(1, -1) - c
    lower_slack = c - lower
    bounds_slack = np.minimum(upper_slack, lower_slack)
    bounds_slack_f = np.where(finite, bounds_slack, np.inf)
    worst_bound = float(np.min(bounds_slack_f))
    n_bound_violations = int(np.sum(bounds_slack_f < -tol))
    bounds_ok = worst_bound >= -tol

    if m_t.size >= 2:
        d_c_d_k = np.diff(c, axis=0) / np.diff(k_t).reshape(-1, 1)
        finite_mono = np.isfinite(d_c_d_k)
        worst_mono = float(np.max(np.where(finite_mono, d_c_d_k, -np.inf)))
    else:
        worst_mono = 0.0

    if m_t.size >= 3:
        dk_l = (k_t[1:-1] - k_t[:-2]).reshape(-1, 1)
        dk_r = (k_t[2:] - k_t[1:-1]).reshape(-1, 1)
        denom = dk_l * dk_r * (dk_l + dk_r) / 2.0
        d2 = (dk_l * c[2:, :] - (dk_l + dk_r) * c[1:-1, :] + dk_r * c[:-2, :]) / denom
        finite_d2 = np.isfinite(d2)
        worst_butterfly = float(np.min(np.where(finite_d2, d2, np.inf)))
        n_butterfly = int(np.sum(finite_d2 & (d2 < -tol)))
    else:
        worst_butterfly = 0.0
        n_butterfly = 0

    butterfly_ok = (worst_butterfly >= -tol) and (worst_mono <= tol)
    if t_t.size >= 2:
        w = (iv_t * iv_t) * t_t.reshape(1, -1)
        dw = np.diff(w, axis=1)
        finite_dw = np.isfinite(dw)
        worst_calendar = float(np.min(np.where(finite_dw, dw, np.inf)))
        n_calendar = int(np.sum(finite_dw & (dw < -tol)))
    else:
        worst_calendar = 0.0
        n_calendar = 0
    calendar_ok = worst_calendar >= -tol

    return ArbitrageReport(
        butterfly_ok=butterfly_ok,
        calendar_ok=calendar_ok,
        bounds_ok=bounds_ok,
        arbitrage_free=butterfly_ok and calendar_ok and bounds_ok,
        worst_butterfly=worst_butterfly,
        worst_monotonicity=worst_mono,
        worst_calendar=worst_calendar,
        worst_bound=worst_bound,
        n_butterfly_violations=n_butterfly,
        n_calendar_violations=n_calendar,
        n_bound_violations=n_bound_violations,
    )


def check_iv_surfaces_arbitrage(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
) -> list[ArbitrageReport]:
    """Run :func:`check_iv_surface_arbitrage` over every leading-axis slice."""
    iv_t = np.asarray(iv, dtype=float)
    if iv_t.ndim < 2:
        raise ValueError(f"iv must have at least 2 dims; got shape {tuple(iv_t.shape)}")
    m_t = np.asarray(moneyness, dtype=float).reshape(-1)
    t_t = np.asarray(tau, dtype=float).reshape(-1)
    leading = iv_t.shape[:-2]
    flat = iv_t.reshape(-1, int(m_t.size), int(t_t.size))
    n = int(np.prod(leading)) if leading else 1
    return [
        check_iv_surface_arbitrage(
            flat[i],
            m_t,
            t_t,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
            tol=tol,
        )
        for i in range(n)
    ]
