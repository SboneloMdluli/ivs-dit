"""Scalar Euler discretization of the SABR SDE for sequential IV surfaces.

SABR dynamics (pricing measure, following Hagan et al. 2002) for the
underlying ``S`` and stochastic-vol level ``alpha``:

.. math::

   dS_t      &= (r - q) S_t \\, dt + \\alpha_t \\, S_t^{\\beta} \\, dW^1_t \\\\
   d\\alpha_t &= \\nu \\, \\alpha_t \\, dW^2_t \\\\
   dW^1_t \\, dW^2_t &= \\rho \\, dt

The ``alpha`` SDE is geometric Brownian motion with zero drift, so it is
integrated in closed form. The spot SDE is integrated with a log-Euler
step for numerical stability and to guarantee positivity regardless of
``beta``.

This mirrors :mod:`implied_volatility_diffusion.models.heston.simulation`
so the sequential SABR recipe has the same shape as the Heston one.
"""

from __future__ import annotations

import math

import numpy as np


def sabr_step(
    s: float,
    alpha: float,
    dt: float,
    r: float,
    q: float,
    beta: float,
    rho: float,
    nu: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One scalar Euler step of the SABR SDE.

    Args:
        s: Current spot level (must be positive).
        alpha: Current SABR level parameter (must be positive).
        dt: Time increment in years (must be positive).
        r: Risk-free rate.
        q: Continuous dividend yield.
        beta: SABR elasticity in ``[0, 1]``.
        rho: SABR correlation in ``[-1, 1]`` (clamped).
        nu: SABR vol-of-vol (must be non-negative).
        rng: NumPy :class:`~numpy.random.Generator` used for the two shocks.

    Returns:
        Tuple ``(s_next, alpha_next)``.
    """
    dt_f = float(dt)
    if dt_f <= 0.0:
        raise ValueError("dt must be positive")
    if float(s) <= 0.0:
        raise ValueError("spot must be positive")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")

    sqrt_dt = math.sqrt(dt_f)
    rho_c = max(-1.0, min(1.0, float(rho)))
    beta_c = max(0.0, min(1.0, float(beta)))
    nu_nn = max(0.0, float(nu))

    z1, z2 = rng.standard_normal(2)
    d_w1 = sqrt_dt * z1
    d_w2 = rho_c * sqrt_dt * z1 + math.sqrt(max(0.0, 1.0 - rho_c**2)) * sqrt_dt * z2

    # alpha: exact log-Euler for dalpha = nu alpha dW2.
    alpha_next = float(alpha) * math.exp(-0.5 * nu_nn**2 * dt_f + nu_nn * d_w2)

    # Spot: log-Euler under risk-neutral drift with local-vol diffusion
    # sigma_loc(S, alpha) = alpha * S^(beta-1) (so dS/S has diffusion alpha * S^(beta-1)).
    sigma_loc = float(alpha) * float(s) ** (beta_c - 1.0)
    drift = (float(r) - float(q) - 0.5 * sigma_loc**2) * dt_f
    log_s_next = math.log(float(s)) + drift + sigma_loc * d_w1
    return float(math.exp(log_s_next)), float(alpha_next)


__all__ = ["sabr_step"]
