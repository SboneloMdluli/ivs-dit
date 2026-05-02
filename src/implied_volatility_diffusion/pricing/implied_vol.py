"""Implied-vol inversion with Newton and scalar fallback."""

import math

import numpy as np
from py_lets_be_rational.exceptions import AboveMaximumException, BelowIntrinsicException
from py_vollib.black_scholes_merton import black_scholes_merton as _bsm_price
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as _bsm_implied_volatility
from scipy import optimize

from implied_volatility_diffusion.pricing.black_scholes import bs_call_price, bs_call_vega


def _call_price_scalar(spot: float, strike: float, tau: float, rate: float, sigma: float, q: float) -> float:
    s, k, t = float(spot), float(strike), float(tau)
    r, qd, sig = float(rate), float(q), float(sigma)
    if t <= 0.0:
        return float(max(s - k, 0.0))
    if sig <= 0.0:
        return float(max(s * math.exp(-qd * t) - k * math.exp(-r * t), 0.0))
    return float(_bsm_price("c", s, k, t, r, sig, qd))


def _jackel_rational_sigma0(
    market_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
) -> float | None:
    try:
        sigma = float(_bsm_implied_volatility(market_price, spot, strike, tau, rate, dividend_yield, "c"))
    except (BelowIntrinsicException, AboveMaximumException, ArithmeticError, ValueError):
        return None
    if not math.isfinite(sigma):
        return None
    return sigma


def _brenner_subrahmanyam_guess(
    spot: float, tau: float, dividend_yield: float, intrinsic: float, market_price: float, tol_price: float
) -> float:
    fwd_price = spot * math.exp(-dividend_yield * tau)
    time_value = max(market_price - intrinsic, tol_price)
    return math.sqrt(2.0 * math.pi / tau) * time_value / max(fwd_price, 1e-16)


def _implied_vol_brent(
    market_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    sigma_lo: float,
    sigma_hi: float,
    xtol: float,
) -> float:
    def objective(sig: float) -> float:
        return _call_price_scalar(spot, strike, tau, rate, sig, dividend_yield) - market_price

    lo, hi = sigma_lo, sigma_hi
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0.0:
        for _ in range(60):
            hi *= 1.35
            f_hi = objective(hi)
            if f_lo * f_hi <= 0.0:
                break
            lo = max(lo * 0.85, sigma_lo * 1e-6)
            f_lo = objective(lo)
            if f_lo * f_hi <= 0.0:
                break
        else:
            raise ValueError("could not bracket implied volatility root")
    return float(optimize.brentq(objective, lo, hi, xtol=xtol, full_output=False))


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float = 0.0,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 10.0,
    newton_refinement_steps: int = 3,
    newton_tol: float = 1e-10,
    brent_xtol: float = 1e-8,
    vega_floor_scale: float = 1e-14,
    intrinsic_nudge_scale: float = 1e-10,
    **_: object,
) -> float:
    """Scalar Black-Scholes-Merton implied volatility (Jäckel + Newton + Brent)."""
    s, k, t = spot, strike, tau
    r, q = rate, dividend_yield
    price = market_price

    if t <= 0.0:
        raise ValueError("tau must be positive for implied volatility")

    intrinsic = max(s * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
    tol_price = max(1e-12 * s, 1e-14)
    nudge = max(intrinsic_nudge_scale * s, 1e-14)
    if price < intrinsic:
        if intrinsic - price > max(nudge * 1e3, 1e-6 * max(s, 1.0)):
            raise ValueError("market_price is materially below discounted intrinsic value")
        price = intrinsic + nudge

    upper = _call_price_scalar(s, k, t, r, sigma_hi, q)
    if price > upper + tol_price:
        raise ValueError("market_price above model upper bound at sigma_hi")

    sigma0 = _jackel_rational_sigma0(price, s, k, t, r, q)
    if sigma0 is None:
        sigma0 = _brenner_subrahmanyam_guess(s, t, q, intrinsic, price, tol_price)

    sigma = min(max(sigma0, sigma_lo), sigma_hi)
    min_vega = max(vega_floor_scale * s, 1e-16)

    for _ in range(max(0, int(newton_refinement_steps))):
        diff = _call_price_scalar(s, k, t, r, sigma, q) - price
        if abs(diff) < newton_tol:
            return float(sigma)
        vg = float(bs_call_vega(s, k, t, r, sigma, q).item())
        if vg < min_vega:
            break
        step = diff / vg
        sigma_next = sigma - step
        if not math.isfinite(sigma_next):
            break
        sigma = min(max(sigma_next, sigma_lo), sigma_hi)
        if abs(step) < 1e-15:
            return float(sigma)

    diff = _call_price_scalar(s, k, t, r, sigma, q) - price
    if abs(diff) < max(newton_tol * 50.0, tol_price):
        return float(sigma)

    sigma = _implied_vol_brent(price, s, k, t, r, q, sigma_lo, sigma_hi, brent_xtol)
    return float(min(max(sigma, sigma_lo), sigma_hi))


def implied_vol_from_prices(
    prices: np.ndarray | float,
    *,
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    tau: np.ndarray | float,
    rate: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 10.0,
    max_newton_steps: int = 32,
    newton_tol: float = 1e-10,
    initial_sigma: float = 0.3,
    fallback_scalar: bool = True,
) -> np.ndarray:
    """Batch-invert call prices to Black-Scholes IV using NumPy broadcasting."""
    p_b, s_b, k_b, t_b, r_b, q_b = np.broadcast_arrays(
        np.asarray(prices, dtype=float),
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )
    sigma = np.full(p_b.shape, float(initial_sigma), dtype=float)
    converged = np.zeros(p_b.shape, dtype=bool)
    min_vega = 1e-16
    for _ in range(max_newton_steps):
        diff = bs_call_price(s_b, k_b, t_b, r_b, sigma, q_b) - p_b
        converged |= np.abs(diff) < newton_tol

        if bool(np.all(converged)):
            break

        vega = bs_call_vega(s_b, k_b, t_b, r_b, sigma, q_b)
        safe_vega = np.where(vega > min_vega, vega, min_vega)
        sigma_next = np.clip(sigma - diff / safe_vega, sigma_lo, sigma_hi)
        sigma = np.where(converged, sigma, sigma_next)

    residual = np.abs(bs_call_price(s_b, k_b, t_b, r_b, sigma, q_b) - p_b)
    bad = residual > max(newton_tol * 50.0, 1e-8)
    if fallback_scalar and bool(np.any(bad)):
        for idx in np.argwhere(bad):
            tup = tuple(int(i) for i in idx)
            try:
                sigma[tup] = implied_volatility(
                    p_b[tup],
                    s_b[tup],
                    k_b[tup],
                    t_b[tup],
                    r_b[tup],
                    q_b[tup],
                    sigma_lo=sigma_lo,
                    sigma_hi=sigma_hi,
                )
            except ValueError:
                sigma[tup] = float("nan")
    return sigma
