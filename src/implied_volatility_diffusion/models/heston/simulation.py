"""Full-truncation Milstein discretization of the risk-neutral Heston SDE.

Two entry points:

* :func:`milstein_step` — scalar NumPy path (used by sequential-IVS generator
  to keep seed-by-seed bit-compatibility with the legacy implementation).
"""

from __future__ import annotations

import math

import numpy as np


def is_feller_satisfied(kappa: float, theta: float, sigma_v: float, *, eps: float = 0.0) -> bool:
    """Return ``True`` when ``2 kappa theta >= sigma_v^2 + eps``."""
    return 2.0 * kappa * theta >= sigma_v**2 + eps


def feller_index(kappa: float, theta: float, sigma_v: float) -> float:
    """Return ``2 kappa theta - sigma_v^2``."""
    return 2.0 * kappa * theta - sigma_v**2


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
    """One scalar full-truncation Milstein step under the risk-neutral Heston SDE."""
    dt_f = float(dt)
    if dt_f <= 0.0:
        raise ValueError("dt must be positive")
    sqrt_dt = math.sqrt(dt_f)
    rho_c = max(-1.0, min(1.0, float(rho)))
    z1, z2 = rng.standard_normal(2)

    d_w1 = sqrt_dt * z1
    d_w2 = rho_c * sqrt_dt * z1 + math.sqrt(max(0.0, 1.0 - rho_c**2)) * sqrt_dt * z2

    v_pos = max(float(v), 0.0)
    sqrt_v = math.sqrt(v_pos)
    drift_v = kappa * (theta - v_pos) * dt_f
    diff_v = sigma_v * sqrt_v * d_w2
    milstein_corr = 0.25 * sigma_v**2 * (d_w2**2 - dt_f)
    v_next = max(v_pos + drift_v + diff_v + milstein_corr, 0.0)

    log_s_next = math.log(float(s)) + (float(r) - float(q) - 0.5 * v_pos) * dt_f + sqrt_v * d_w1
    return float(math.exp(log_s_next)), float(v_next)
