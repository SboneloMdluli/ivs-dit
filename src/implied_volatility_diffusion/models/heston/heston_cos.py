"""Heston model: characteristic function and European call price (COS method).

Reference: "A novel pricing method for European options based on Fourier-cosine
series expansions" (Fang & Oosterlee).
"""

import math

import numpy as np


def _heston_cf(
    u: np.ndarray,
    tau: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> np.ndarray:
    """Risk-neutral Heston characteristic function.

    Args:
        u: Fourier-cosine frequencies
        tau: Time to expiry
        spot: Spot level
        rate: Risk-free rate
        dividend_yield: Dividend yield
        kappa: Heston parameter
        theta: Heston parameter
        sigma: Heston parameter
        rho: Heston parameter
        v0: Heston parameter

    Returns:
        Risk-neutral Heston characteristic function
    Reference: Gatheral, J., 2011. The volatility surface: a practitioner’s guide. John Wiley & Sons.
    """
    iu = 1j * u
    rmq = rate - dividend_yield
    sig2 = sigma**2

    p = kappa - rho * sigma * iu
    q = sig2 * (iu + u * u)
    d = np.sqrt(p * p + q)
    d = np.where(np.abs(d) < 1e-14, 1e-14 + 0j, d)

    exp_mdt = np.exp(-d * tau)
    g = (p - d) / (p + d)
    g = np.where(np.abs(g) > 1e-12, g, 1e-12 + 0j)

    intr = 1.0 - g * exp_mdt
    denom = 1.0 - g

    C = rmq * iu * tau + (kappa * theta / sig2) * ((p - d) * tau - 2.0 * (np.log(intr) - np.log(denom)))
    D = ((p - d) / sig2) * ((1.0 - exp_mdt) / intr)
    return np.exp(C + D * v0 + iu * np.log(float(spot)))


def _heston_log_moments_truncation(
    tau: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    L: float,
) -> tuple[float, float, float]:
    """Truncation [a, b] for x = ln(S_T).

    Cumulant formulas follow Fang & Oosterlee for ln(S_T/S_0); the first cumulant of
    ln(S_T) adds ln(S_0) because Var(ln S_T) = Var(ln(S_T/S_0)).
    Reference: https://mpra.ub.uni-muenchen.de/8914/4/MPRA_paper_8914.pdf
    """
    rmq = rate - dividend_yield
    exp_k = np.exp(-kappa * tau)
    c1_rel = rmq * tau + (1.0 - exp_k) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * tau
    c1 = c1_rel + np.log(float(spot))

    k2 = kappa * kappa
    k3 = k2 * kappa
    term_bracket = np.exp(-kappa * tau) * (
        4.0 * kappa * v0 * np.exp(-kappa * tau)
        - 4.0 * kappa * (theta - 2.0 * v0) * np.exp(-kappa * tau)
        + k2 * theta * tau
        - 4.0 * kappa * theta
        - 4.0 * kappa * v0
        + 4.0 * theta * np.exp(-kappa * tau)
        + 4.0 * kappa * theta
    )

    c2_formula = (theta / (8.0 * k3)) * (term_bracket + theta * (2.0 * v0 - theta) * tau)
    integrated_var = theta * tau + (v0 - theta) * (1.0 - np.exp(-kappa * tau)) / kappa
    c2_real = float(np.real(c2_formula))
    c2_lo = max(float(integrated_var), 1e-16)
    c2_hi = max(c2_lo * 2.0, c2_lo + 0.25)
    c2 = min(max(c2_real if np.isfinite(c2_real) else c2_lo, c2_lo), c2_hi)

    # [a, b] := [c1 - L * sqrt(c2 ), c1 + L * sqrt(c2))]
    width = L * np.sqrt(c2)
    a = float(c1 - width)
    b = float(c1 + width)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a = float(c1 - L * np.sqrt(theta * tau + 1e-16))
        b = float(c1 + L * np.sqrt(theta * tau + 1e-16))
    return a, b, float(integrated_var)


def _widen_truncation_for_strike(
    a: float,
    b: float,
    *,
    strike: float,
    integrated_var: float,
    v0: float,
    tau: float,
    truncation_l: float,
) -> tuple[float, float]:
    """Pad the COS log-spot truncation interval around the strike.

    Cumulant-only ``[a, b]`` can miss ``log(K)`` when τ is tiny, which underprices options
    and breaks implied-vol inversion. This widens the interval symmetrically in log space.
    """
    std_ln = math.sqrt(max(integrated_var, v0 * tau, 1e-20))
    pad = max(float(truncation_l), 6.0) * std_ln
    log_k = math.log(strike)
    a_w = min(a, log_k - pad)
    b_w = max(b, log_k + pad)
    if b_w <= a_w:
        eps = 1e-4
        a_w, b_w = log_k - eps, log_k + eps
    return a_w, b_w


def _call_cos_coefficients(
    strike: float,
    a: float,
    b: float,
    n_terms: int,
) -> np.ndarray:
    """Fourier–cosine coefficients V_k for v(x)=max(e^x-K,0) on x=ln S_T (Fang & Oosterlee)."""
    k = np.arange(n_terms, dtype=float)
    omega = k * np.pi / (b - a)
    log_k = np.log(strike)
    c = max(log_k, a)
    d = b
    if c >= d:
        return np.zeros(n_terms, dtype=float)

    coeff = np.zeros(n_terms, dtype=float)
    exp_c = np.exp(c)
    exp_d = np.exp(d)

    coeff[0] = (1.0 / (b - a)) * (exp_d - exp_c - strike * (d - c))

    mask = k > 0
    if np.any(mask):
        w = omega[mask]
        chi = np.real(np.exp(-1j * w * a) * (np.exp((1.0 + 1j * w) * d) - np.exp((1.0 + 1j * w) * c)) / (1.0 + 1j * w))
        psi = strike * (np.sin(w * (d - a)) - np.sin(w * (c - a))) / w
        coeff[mask] = (2.0 / (b - a)) * (chi - psi)

    return coeff


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
    """European call price under Heston via COS (Fang & Oosterlee)."""
    if tau <= 0.0:
        return float(max(spot - strike, 0.0))

    a, b, integrated_var = _heston_log_moments_truncation(
        tau, spot, rate, dividend_yield, kappa, theta, sigma, rho, v0, truncation_l
    )
    a, b = _widen_truncation_for_strike(
        a,
        b,
        strike=strike,
        integrated_var=integrated_var,
        v0=v0,
        tau=tau,
        truncation_l=truncation_l,
    )
    k = np.arange(n_terms, dtype=float)
    u = k * np.pi / (b - a)
    phi = _heston_cf(u, tau, spot, rate, dividend_yield, kappa, theta, sigma, rho, v0)
    exp_shift = np.exp(-1j * u * a)
    uk = _call_cos_coefficients(strike, a, b, n_terms)
    terms = np.real(phi * exp_shift) * uk

    # discount the price
    price = np.exp(-rate * tau) * np.sum(terms)
    return max(float(price), 0.0)
