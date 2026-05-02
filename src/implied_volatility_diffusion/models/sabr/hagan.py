"""SABR lognormal (Hagan 2002) implied vol."""

import math

import numpy as np


def _x_scalar(rho: float, z: float) -> float:
    a = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 0.0)) + z - rho
    b = 1.0 - rho
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    return math.log(a / b)


def sabr_hagan_lognormal_iv(
    forward: float,
    strike: float,
    tau: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Scalar Hagan 2002 lognormal implied vol."""
    if strike <= 0.0 or forward <= 0.0 or tau <= 0.0:
        return 0.0
    if alpha <= 0.0 or nu < 0.0 or not (-0.9999 <= rho <= 0.9999):
        return float("nan")

    k = float(strike)
    f = float(forward)
    t = float(tau)
    one_m_beta = 1.0 - beta
    log_fk = math.log(f / k)
    fk_beta = (f * k) ** one_m_beta
    if fk_beta <= 0.0:
        return float("nan")

    a = (one_m_beta**2) * (alpha**2) / (24.0 * fk_beta)
    b = 0.25 * rho * beta * nu * alpha / math.sqrt(fk_beta)
    c = ((2.0 - 3.0 * rho * rho) * (nu**2)) / 24.0
    d = math.sqrt(fk_beta)
    v = (one_m_beta**2) * (log_fk**2) / 24.0
    w = (one_m_beta**4) * (log_fk**4) / 1920.0
    z = nu * d * log_fk / alpha

    eps = 1e-7
    base = alpha * (1.0 + (a + b + c) * t) / (d * (1.0 + v + w))
    if abs(z) <= eps:
        return float(max(base, 1e-16))

    xz = _x_scalar(rho, z)
    if not math.isfinite(xz) or abs(xz) < 1e-16:
        return float(max(base, 1e-16))
    return float(max(base * (z / xz), 1e-16))


def sabr_hagan_lognormal_iv_array(
    forward: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Element-wise Hagan IV for broadcastable ``forward``, ``strike``, ``tau``."""
    f, k, t = np.broadcast_arrays(
        np.asarray(forward, dtype=float), np.asarray(strike, dtype=float), np.asarray(tau, dtype=float)
    )
    out = np.empty(f.shape, dtype=float)
    it = np.nditer(
        [f, k, t, out], flags=["multi_index"], op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]]
    )
    for fv, kv, tv, ov in it:
        ov[...] = sabr_hagan_lognormal_iv(float(fv), float(kv), float(tv), alpha, beta, rho, nu)
    return out
