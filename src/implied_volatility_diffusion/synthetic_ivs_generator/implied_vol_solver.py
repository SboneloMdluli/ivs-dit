"""Implied volatility solver for European call options."""

import math

from py_lets_be_rational.exceptions import (
    AboveMaximumException,
    BelowIntrinsicException,
)
from py_vollib.black_scholes_merton import black_scholes_merton as _bsm_price
from py_vollib.black_scholes_merton.greeks.analytical import vega as _bsm_vega
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as _bsm_implied_volatility
from scipy import optimize


def call_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Black–Scholes–Merton price of a European call (continuous yield)."""
    s, k, t = float(spot), float(strike), float(tau)
    r, q, sig = float(rate), float(dividend_yield), float(vol)

    if t <= 0.0:
        return float(max(s - k, 0.0))
    if sig <= 0.0:
        return float(max(s * math.exp(-q * t) - k * math.exp(-r * t), 0.0))
    # Black–Scholes–Merton price of a European call
    return float(_bsm_price("c", s, k, t, r, sig, q))


def call_vega(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Black–Scholes–Merton vega for a European call (per unit volatility)."""
    s, k, t = float(spot), float(strike), float(tau)
    r, q, sig = float(rate), float(dividend_yield), float(vol)
    if t <= 0.0 or sig <= 0.0 or s <= 0.0 or k <= 0.0:
        return 0.0
    # py_vollib reports vega per 1% vol move; rescale to per unit volatility.
    return float(_bsm_vega("c", s, k, t, r, sig, q)) * 100.0


def _jackel_rational_sigma0(
    market_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
) -> float | None:
    """Run py_vollib's lets_be_rational seed; return ``None`` if it cannot converge."""
    try:
        sigma = float(
            _bsm_implied_volatility(
                market_price,
                spot,
                strike,
                tau,
                rate,
                dividend_yield,
                "c",
            )
        )
    except (BelowIntrinsicException, AboveMaximumException, ArithmeticError, ValueError):
        return None
    if not math.isfinite(sigma):
        return None
    return sigma


def _brenner_subrahmanyam_guess(
    spot: float,
    tau: float,
    dividend_yield: float,
    intrinsic: float,
    market_price: float,
    tol_price: float,
) -> float:
    """ATM-style time-value guess used when the rational seed is unavailable."""
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
    """Brent's root-finding over ``call_price(sigma) - market_price``."""

    def objective(sig: float) -> float:
        return call_price(spot, strike, tau, rate, sig, dividend_yield) - market_price

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
    """Black–Scholes–Merton implied volatility via Jäckel + Newton + Brent.

    Args:
        market_price: Discounted European call price
        spot: Spot level.
        strike: Strike.
        tau: Time to expiry in years
        rate: risk-free rate.
        dividend_yield: dividend yield.
        sigma_lo: Lower clamp on the returned volatility.
        sigma_hi: Upper clamp on the returned volatility.
        newton_refinement_steps: Max Newton iterations
        newton_tol: Absolute price residual to treat Newton as converged.
        brent_xtol: Brent tolerance
        vega_floor_scale: Stop Newton when ``vega < vega_floor_scale * spot``
        intrinsic_nudge_scale: Nudge price below intrinsic value to avoid Jäckel's rational seed raising

    Returns:
        Implied volatility in range [sigma_lo, sigma_hi].

    Raises:
        ValueError: tau <= 0 or the price violates no-arbitrage bounds beyond
            the nudge / upper-bound tolerance.
    """
    s, k, t = spot, strike, tau
    r, q = rate, dividend_yield
    price = market_price

    if t <= 0.0:
        raise ValueError("tau must be positive for implied volatility")

    intrinsic = max(s * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
    tol_price = max(1e-12 * s, 1e-14)
    nudge = max(intrinsic_nudge_scale * s, 1e-14)

    # check if the Hestonprice is below the intrinsic value
    if price < intrinsic:
        if intrinsic - price > max(nudge * 1e3, 1e-6 * max(s, 1.0)):
            raise ValueError("market_price is materially below discounted intrinsic value")
        price = intrinsic + nudge

    upper = call_price(s, k, t, r, sigma_hi, q)
    # check if the Heston price is above the upper bound
    if price > upper + tol_price:
        raise ValueError("market_price above model upper bound at sigma_hi")

    sigma0 = _jackel_rational_sigma0(price, s, k, t, r, q)

    # fallback to the Brenner-Subrahmanyam
    if sigma0 is None:
        sigma0 = _brenner_subrahmanyam_guess(s, t, q, intrinsic, price, tol_price)

    sigma = min(max(sigma0, sigma_lo), sigma_hi)
    min_vega = max(vega_floor_scale * s, 1e-16)

    # Newton refinement
    for _ in range(max(0, int(newton_refinement_steps))):
        diff = call_price(s, k, t, r, sigma, q) - price
        if abs(diff) < newton_tol:
            return float(sigma)
        vg = call_vega(s, k, t, r, sigma, q)
        if vg < min_vega:
            break
        step = diff / vg
        sigma_next = sigma - step
        if not math.isfinite(sigma_next):
            break
        sigma = min(max(sigma_next, sigma_lo), sigma_hi)
        if abs(step) < 1e-15:
            return float(sigma)

    # final check if the Heston price is close to the market price
    diff = call_price(s, k, t, r, sigma, q) - price
    if abs(diff) < max(newton_tol * 50.0, tol_price):
        return float(sigma)

    # Brent fallback
    sigma = _implied_vol_brent(price, s, k, t, r, q, sigma_lo, sigma_hi, brent_xtol)
    return float(min(max(sigma, sigma_lo), sigma_hi))
