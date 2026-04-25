"""Heston COS pricer wrappers without torch utility dependencies."""

from __future__ import annotations

import numpy as np

from implied_volatility_diffusion.models.heston.heston_cos import heston_call_cos as _heston_call_cos_numpy


def heston_call_cos_batch(
    spot: float,
    strikes: np.ndarray,
    tau: float,
    rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    dividend_yield: float = 0.0,
    n_terms: int = 1024,
    truncation_L: float = 14.0,
) -> np.ndarray:
    """Batched Heston call price over NumPy strikes at one ``tau``."""
    k_arr = np.asarray(strikes, dtype=float).reshape(-1)
    return np.array(
        [
            _heston_call_cos_numpy(
                float(spot),
                float(kv),
                float(tau),
                float(rate),
                float(kappa),
                float(theta),
                float(sigma),
                float(rho),
                float(v0),
                dividend_yield=float(dividend_yield),
                n_terms=int(n_terms),
                truncation_L=float(truncation_L),
            )
            for kv in k_arr
        ],
        dtype=float,
    )


def heston_call_cos(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    dividend_yield: float = 0.0,
    n_terms: int = 1024,
    truncation_L: float = 14.0,
) -> float:
    """Scalar Heston call price via COS."""
    return float(
        _heston_call_cos_numpy(
            float(spot),
            float(strike),
            float(tau),
            float(rate),
            float(kappa),
            float(theta),
            float(sigma),
            float(rho),
            float(v0),
            dividend_yield=float(dividend_yield),
            n_terms=int(n_terms),
            truncation_L=float(truncation_L),
        )
    )
