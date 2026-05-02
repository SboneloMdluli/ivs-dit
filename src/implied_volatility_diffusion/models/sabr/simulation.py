"""SABR simulation step used for sequential surfaces."""

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
    """One scalar Euler step of the SABR SDE."""
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

    alpha_next = float(alpha) * math.exp(-0.5 * nu_nn**2 * dt_f + nu_nn * d_w2)

    sigma_loc = float(alpha) * float(s) ** (beta_c - 1.0)
    drift = (float(r) - float(q) - 0.5 * sigma_loc**2) * dt_f
    log_s_next = math.log(float(s)) + drift + sigma_loc * d_w1
    return float(math.exp(log_s_next)), float(alpha_next)


__all__ = ["sabr_step"]
