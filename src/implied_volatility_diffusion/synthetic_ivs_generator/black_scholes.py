"""Black–Scholes European call and implied volatility (Jäckel + Newton + Brent)."""

from __future__ import annotations

import math

import numpy as np
from py_lets_be_rational import (
    implied_volatility_from_a_transformed_rational_guess_with_limited_iterations as imp_vol_jackel,
)
from py_lets_be_rational.exceptions import (
    AboveMaximumException,
    BelowIntrinsicException,
)
from scipy import optimize, special


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def call_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Black–Scholes price of a European call (continuous yield)."""
    if tau <= 0.0:
        return float(max(spot - strike, 0.0))
    if vol <= 0.0:
        fwd = spot * math.exp(-dividend_yield * tau)
        df = math.exp(-rate * tau)
        return float(max(df * (fwd - strike), 0.0))

    sqrt_t = math.sqrt(tau)

    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * tau) / (
        vol * sqrt_t
    )

    d2 = d1 - vol * sqrt_t

    return float(
        spot * math.exp(-dividend_yield * tau) * special.ndtr(d1)
        - strike * math.exp(-rate * tau) * special.ndtr(d2)
    )


def call_vega(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Black–Scholes vega for a European call."""
    if tau <= 0.0 or vol <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * tau) / (
        vol * sqrt_t
    )
    return float(spot * math.exp(-dividend_yield * tau) * sqrt_t * _norm_pdf(d1))


def _undiscounted_forward_call(discounted_price: float, rate: float, tau: float) -> float:
    """Map discounted European call price to undiscounted forward call value."""
    return discounted_price * math.exp(rate * tau)


def _jackel_rational_sigma0(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    market_discounted: float,
    *,
    jackel_iterations: int = 0,
) -> float | None:
    """Peter Jäckel rational IV (LetsBeRational with limited internal iterations)."""
    if strike <= 0.0 or spot <= 0.0 or tau <= 0.0:
        return None
    forward = spot * math.exp((rate - dividend_yield) * tau)
    price_fwd = _undiscounted_forward_call(market_discounted, rate, tau)
    try:
        # calculate the implied volatility using the Jäckel rational approximation
        # http://www.jaeckel.org/LetsBeRational.pdf
        return float(
            imp_vol_jackel(
                price_fwd, forward, float(strike), tau, 1, int(jackel_iterations)
            )
        )
    except (BelowIntrinsicException, AboveMaximumException, ArithmeticError, ValueError):
        return None


def _brenner_subrahmanyam_guess(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    intrinsic: float,
    market_price: float,
    tol_price: float,
) -> float:
    """ATM-style time-value guess when Jäckel is unavailable."""
    fwd = spot * math.exp(-dividend_yield * tau)
    time_value = max(market_price - intrinsic, tol_price)
    return math.sqrt(2.0 * math.pi / tau) * time_value / max(fwd, 1e-16)


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
    """Brent's method for root-finding (robust tail / tiny vega)."""

    def objective(sig: float) -> float:
        return call_price(spot, strike, tau, rate, sig, dividend_yield) - market_price

    lo, hi = sigma_lo, sigma_hi
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        for _ in range(60):
            hi *= 1.35
            f_hi = objective(hi)
            if f_lo * f_hi <= 0:
                break
            lo = max(lo * 0.85, sigma_lo * 1e-6)
            f_lo = objective(lo)
            if f_lo * f_hi <= 0:
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
    sigma_hi: float = 5.0,
    xtol: float = 1e-8,
    *,
    newton_refinement_steps: int = 3,
    newton_tol: float = 1e-10,
    jackel_iterations: int = 0,
    vega_floor_scale: float = 1e-14,
) -> float:
    """Implied Black–Scholes volatility using a 3-level pipeline.

    Level 1 — Jäckel rational approximation (``py_lets_be_rational``, ``jackel_iterations``
    internal refinement; default 0 = rational stage only).

    Level 2 — a few Newton steps on discounted Black–Scholes vs market price.

    Level 3 — Brent only if Newton stalls (tiny vega, wings, etc.).

    Reference: https://arxiv.org/pdf/1901.08943

    Args:
        market_price: Model call price (e.g. COS), discounted.
        spot: Spot (or forward numeraire-consistent spot).
        strike: Strike.
        tau: Time to expiry in years.
        rate: Risk-free rate (continuous compounding).
        dividend_yield: Continuous dividend yield.
        sigma_lo: Lower search bound for implied vol.
        sigma_hi: Upper search bound for implied vol.
        xtol: Brent tolerance when Level 3 runs.
        newton_refinement_steps: Max Newton iterations after the Jäckel seed.
        newton_tol: Absolute price residual to treat Newton as converged.
        jackel_iterations: Passed to LetsBeRational (0–2 typical; 0 = fastest seed).
        vega_floor_scale: Stop Newton when vega < ``vega_floor_scale * spot``.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive for implied volatility")

    # S0*exp(-q*tau) - K*exp(-r*tau)
    intrinsic = max(
        spot * math.exp(-dividend_yield * tau) - strike * math.exp(-rate * tau),
        0.0,
    )
    tol_price = max(1e-12 * spot, 1e-14)
    arb_nudge = max(1e-10 * spot, 1e-12)

    if market_price < intrinsic:
        market_price = intrinsic + arb_nudge
    if market_price + tol_price < intrinsic:
        raise ValueError("market_price below discounted intrinsic value")

    upper = call_price(spot, strike, tau, rate, sigma_hi, dividend_yield)
    if market_price > upper + tol_price:
        raise ValueError("market_price above model upper bound at sigma_hi")

    sigma0 = _jackel_rational_sigma0(
        spot,
        strike,
        tau,
        rate,
        dividend_yield,
        market_price,
        jackel_iterations=jackel_iterations,
    )
    if sigma0 is None or not math.isfinite(sigma0):
        sigma0 = _brenner_subrahmanyam_guess(
            spot, strike, tau, rate, dividend_yield, intrinsic, market_price, tol_price
        )
    sigma = min(max(float(sigma0), sigma_lo), sigma_hi)

    min_vega = max(vega_floor_scale * spot, 1e-16)

    for _ in range(max(0, newton_refinement_steps)):
        diff = call_price(spot, strike, tau, rate, sigma, dividend_yield) - market_price
        if abs(diff) < newton_tol:
            return float(sigma)
        vg = call_vega(spot, strike, tau, rate, sigma, dividend_yield)
        if vg < min_vega:
            break
        step = diff / vg
        sigma_next = sigma - step
        if not math.isfinite(sigma_next):
            break
        sigma = min(max(sigma_next, sigma_lo), sigma_hi)
        if abs(step) < 1e-15:
            return float(sigma)

    diff = call_price(spot, strike, tau, rate, sigma, dividend_yield) - market_price
    if abs(diff) < max(newton_tol * 50.0, tol_price):
        return float(sigma)

    return _implied_vol_brent(
        market_price,
        spot,
        strike,
        tau,
        rate,
        dividend_yield,
        sigma_lo,
        sigma_hi,
        xtol,
    )


def implied_volatility_array(
    market_prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    taus: np.ndarray,
    rate: float,
    dividend_yield: float = 0.0,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
    xtol: float = 1e-8,
) -> np.ndarray:
    """Element-wise implied vol for broadcastable (strikes, taus) grids."""
    prices = np.asarray(market_prices, dtype=float)
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(taus, dtype=float)
    out = np.empty_like(prices, dtype=float)
    it = np.nditer(
        [prices, k, t, out],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
    )

    # compute the implied volatility for each element in the broadcasted grid
    for p, strike, tau, o in it:
        o[...] = implied_volatility(
            float(p),
            spot,
            float(strike),
            float(tau),
            rate,
            dividend_yield=dividend_yield,
            sigma_lo=sigma_lo,
            sigma_hi=sigma_hi,
            xtol=xtol,
        )
    return out
