"""Heston COS pricer wrappers without torch utility dependencies."""

import numpy as np

from implied_volatility_diffusion.models.heston.heston_cos import heston_call_cos


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
    truncation_l: float = 14.0,
) -> np.ndarray:
    """Batched Heston call price over NumPy strikes at one ``tau``."""
    k_arr = np.asarray(strikes, dtype=float).reshape(-1)
    return np.array(
        [
            heston_call_cos(
                spot,
                kv,
                tau,
                rate,
                kappa,
                theta,
                sigma,
                rho,
                v0,
                dividend_yield=float(dividend_yield),
                n_terms=int(n_terms),
                truncation_l=float(truncation_l),
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
    truncation_l: float = 14.0,
) -> float:
    """Scalar Heston call price via COS."""
    return float(
        heston_call_cos(
            spot=spot,
            strike=strike,
            tau=tau,
            rate=rate,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            v0=v0,
            dividend_yield=dividend_yield,
            n_terms=int(n_terms),
            truncation_l=float(truncation_l),
        )
    )
