"""Black-Scholes-Merton call price and vega helpers."""

from __future__ import annotations

import math

import numpy as np
from py_vollib.black_scholes_merton import black_scholes_merton as _bsm_price
from py_vollib.black_scholes_merton.greeks.analytical import vega as _bsm_vega_pct

_VEGA_PER_UNIT_SCALE = 100.0  # 100% per unit of volatility


def _price_scalar(s: float, k: float, t: float, r: float, sig: float, q: float) -> float:
    """Scalar BSM call price with the legacy guard-rails."""
    if not math.isfinite(s) or not math.isfinite(k):
        return float("nan")
    if t <= 0.0:
        return float(max(s - k, 0.0))
    if sig <= 0.0:
        return float(max(s * math.exp(-q * t) - k * math.exp(-r * t), 0.0))
    if s <= 0.0 or k <= 0.0:
        return float("nan")
    return float(_bsm_price("c", float(s), float(k), float(t), float(r), float(sig), float(q)))


def _vega_scalar(s: float, k: float, t: float, r: float, sig: float, q: float) -> float:
    """Scalar BSM call vega in per-1-unit-vol convention."""
    if not (math.isfinite(s) and math.isfinite(k) and math.isfinite(t) and math.isfinite(sig)):
        return 0.0
    if t <= 0.0 or sig <= 0.0 or s <= 0.0 or k <= 0.0:
        return 0.0
    return (
        float(_bsm_vega_pct("c", float(s), float(k), float(t), float(r), float(sig), float(q))) * _VEGA_PER_UNIT_SCALE
    )


_price_ufunc = np.frompyfunc(_price_scalar, 6, 1)
_vega_ufunc = np.frompyfunc(_vega_scalar, 6, 1)


def bs_call_price(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    tau: np.ndarray | float,
    rate: np.ndarray | float,
    sigma: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes-Merton call price via :mod:`py_vollib`."""
    s, k, t, r, sig, q = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )
    return np.asarray(_price_ufunc(s, k, t, r, sig, q), dtype=float)


def bs_call_vega(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    tau: np.ndarray | float,
    rate: np.ndarray | float,
    sigma: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes-Merton call vega via :mod:`py_vollib` (per 1 vol unit)."""
    s, k, t, r, sig, q = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )
    return np.asarray(_vega_ufunc(s, k, t, r, sig, q), dtype=float)


def bs_call_price_scalar(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Scalar convenience used by the arbitrage code path."""
    return _price_scalar(float(spot), float(strike), float(tau), float(rate), float(vol), float(dividend_yield))
