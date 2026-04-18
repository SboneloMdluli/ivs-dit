"""Time-discretization schemes for the risk-neutral Heston model simulation.

References:
- Gatheral, J., 2011. The volatility surface: a practitioner's guide. John Wiley & Sons.
- L. Andersen, "Efficient Simulation of the Heston Stochastic Volatility Model",
  Journal of Computational Finance, 2008.
"""

import math

import numpy as np

def is_feller_satisfied(kappa: float, theta: float, sigma_v: float, *, eps: float = 0.0) -> bool:
    """Return ``True`` when ``2 kappa theta >= sigma_v^2 + eps`` (Feller condition).

    A strict inequality ``2 kappa theta > sigma_v^2`` rules out the variance hitting
    zero in continuous time. Use a positive ``eps`` to enforce a numerical margin.
    """
    return 2.0 * float(kappa) * float(theta) >= float(sigma_v) * float(sigma_v) + float(eps)


def feller_index(kappa: float, theta: float, sigma_v: float) -> float:
    """Return ``2 kappa theta - sigma_v^2``; positive when Feller holds."""
    return 2.0 * float(kappa) * float(theta) - float(sigma_v) * float(sigma_v)


def _correlated_normals(rho: float, rng: np.random.Generator) -> tuple[float, float]:
    """Draw ``(Z_s, Z_v)`` with ``corr(Z_s, Z_v) = rho`` from two iid standard normals."""
    z_s, z_indep = rng.standard_normal(2)
    rho_c = max(-1.0, min(1.0, float(rho)))
    z_v = rho_c * z_s + math.sqrt(max(0.0, 1.0 - rho_c * rho_c)) * z_indep
    return float(z_s), float(z_v)


def milstein_step(
    s: float,
    v: float,
    dt: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One Milstein step for the joint ``(S, v)``

    Variance update (full-truncation Milstein):

    """
    dt_f = float(dt)
    if dt_f <= 0.0:
        raise ValueError("dt must be positive")
    sqrt_dt = math.sqrt(dt_f)
    z_s, z_v = _correlated_normals(rho, rng)

    v_pos = max(float(v), 0.0)
    sqrt_v = math.sqrt(v_pos)
    drift_v = float(kappa) * (float(theta) - v_pos) * dt_f
    diff_v = float(sigma_v) * sqrt_v * sqrt_dt * z_v
    milstein_corr = 0.25 * float(sigma_v) * float(sigma_v) * dt_f * (z_v * z_v - 1.0)
    v_next = max(v_pos + drift_v + diff_v + milstein_corr, 0.0)

    log_s_next = math.log(float(s)) + (float(r) - float(q) - 0.5 * v_pos) * dt_f + sqrt_v * sqrt_dt * z_s
    return float(math.exp(log_s_next)), float(v_next)


def heston_path_step(
    s: float,
    v: float,
    dt: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Dispatch one ``(spot, variance)`` step under the requested scheme."""
  
    return milstein_step(s, v, dt, r, q, kappa, theta, sigma_v, rho, rng)

