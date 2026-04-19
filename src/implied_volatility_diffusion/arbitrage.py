"""No-arbitrage diagnostics for Black-Scholes implied-volatility surfaces.

Two checks are provided, following the standard Roper / Gatheral conditions:

* **Butterfly (strike) no-arbitrage**: at each maturity, the *undiscounted*
  European call price ``C(K, tau)`` must be a non-increasing, convex function
  of the strike ``K``, with values inside the model-free bounds
  ``max(S e^{-q tau} - K e^{-r tau}, 0) <= C <= S e^{-q tau}``.
  Equivalently the BS price built from the implied vol must satisfy

  .. math:: \\partial_K C \\le 0 \\quad\\text{and}\\quad \\partial^2_{K^2} C \\ge 0.

* **Calendar no-arbitrage**: at each fixed moneyness ``m = K / S_0`` the total
  implied variance ``w(m, tau) = sigma_imp(m, tau)^2 \\, tau`` must be
  non-decreasing in ``tau``. This is the standard form used in synthetic-IVS
  generators that work on a fixed (m, tau) grid and is exact in the absence of
  a deterministic forward drift.

The functions here take the implied-vol surface in the same shape produced by
:func:`implied_vol_surface_for_params` -- ``(n_moneyness, n_tau)`` -- or any
batch of such surfaces, and return per-surface flags plus the worst violation
margin, so they can be used both as soft diagnostics and inside ``assert``s.
"""

from dataclasses import dataclass

import numpy as np

from implied_volatility_diffusion.synthetic_ivs_generator.implied_vol_solver import call_price


@dataclass(frozen=True)
class ArbitrageReport:
    """Per-surface arbitrage diagnostics.

    Attributes:
        butterfly_ok: True iff every maturity slice is monotone non-increasing
            and convex in the strike (within ``tol``).
        calendar_ok: True iff total implied variance is non-decreasing in tau
            at every moneyness (within ``tol``).
        bounds_ok: True iff the BS call prices implied by the surface lie inside
            the model-free bounds ``[max(F - K, 0) e^{-r tau}, S e^{-q tau}]``.
        arbitrage_free: Logical AND of the three flags above.
        worst_butterfly: Most negative second-difference of ``C`` in K
            (``>= -tol`` means convex). Negative magnitude = violation depth.
        worst_monotonicity: Most positive forward difference of ``C`` in K
            (``<= tol`` means non-increasing). Positive magnitude = violation depth.
        worst_calendar: Most negative forward difference of total variance ``w``
            in tau (``>= -tol`` means non-decreasing).
        worst_bound: Most negative slack against the BS upper / intrinsic bounds.
        n_butterfly_violations: Number of strike triplets violating convexity.
        n_calendar_violations: Number of (m, tau)-pairs violating monotone w.
        n_bound_violations: Number of grid points outside the model-free bounds.
    """

    butterfly_ok: bool
    calendar_ok: bool
    bounds_ok: bool
    arbitrage_free: bool
    worst_butterfly: float
    worst_monotonicity: float
    worst_calendar: float
    worst_bound: float
    n_butterfly_violations: int
    n_calendar_violations: int
    n_bound_violations: int


def _bs_call_grid(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Black-Scholes call prices on the (moneyness, tau) grid implied by ``iv``."""
    out = np.empty_like(iv, dtype=float)
    for i, m_val in enumerate(moneyness):
        k = float(m_val) * float(spot)
        for j, t_val in enumerate(tau):
            sig = float(iv[i, j])
            if not np.isfinite(sig) or sig <= 0.0 or float(t_val) <= 0.0:
                out[i, j] = np.nan
                continue
            out[i, j] = call_price(spot, k, float(t_val), float(rate), sig, float(dividend_yield))
    return out


def check_iv_surface_arbitrage(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
) -> ArbitrageReport:
    """Check Roper / Gatheral no-arbitrage conditions for one IV surface.

    The strike grid ``K_i = moneyness_i * spot`` need not be uniform; second
    differences are formed from the standard non-uniform 3-point stencil.

    Args:
        iv: Implied-vol surface, shape ``(n_moneyness, n_tau)``.
        moneyness: Strictly increasing strike-over-spot grid.
        tau: Strictly increasing year-fraction grid.
        spot: Underlying spot used to translate moneyness to strike.
        rate: Risk-free rate used to price BS calls.
        dividend_yield: Continuous dividend yield.
        tol: Slack tolerance applied to every inequality.

    Returns:
        :class:`ArbitrageReport` summarising the surface.
    """
    iv = np.asarray(iv, dtype=float)
    if iv.ndim != 2:
        raise ValueError(f"iv must be 2D (n_moneyness, n_tau); got shape {iv.shape}")
    m = np.asarray(moneyness, dtype=float).ravel()
    t = np.asarray(tau, dtype=float).ravel()
    if iv.shape != (m.size, t.size):
        raise ValueError(
            f"iv shape {iv.shape} does not match (len(moneyness)={m.size}, len(tau)={t.size})"
        )
    if np.any(np.diff(m) <= 0.0):
        raise ValueError("moneyness must be strictly increasing")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("tau must be strictly increasing")

    c = _bs_call_grid(iv, m, t, spot=spot, rate=rate, dividend_yield=dividend_yield)
    k = m * float(spot)
    upper = float(spot) * np.exp(-float(dividend_yield) * t)
    lower = np.maximum(float(spot) * np.exp(-float(dividend_yield) * t)[None, :]
                       - k[:, None] * np.exp(-float(rate) * t)[None, :], 0.0)
    finite = np.isfinite(c)

    upper_slack = upper[None, :] - c
    lower_slack = c - lower
    bounds_slack = np.minimum(upper_slack, lower_slack)
    bounds_slack_finite = np.where(finite, bounds_slack, np.inf)
    worst_bound = float(np.min(bounds_slack_finite))
    n_bound_violations = int(np.sum(bounds_slack_finite < -tol))
    bounds_ok = worst_bound >= -tol

    if m.size >= 2:
        dC_dK = np.diff(c, axis=0) / np.diff(k)[:, None]
        finite_mono = np.isfinite(dC_dK)
        worst_mono = float(np.max(np.where(finite_mono, dC_dK, -np.inf)))
        n_mono_violations = int(np.sum(finite_mono & (dC_dK > tol)))
    else:
        worst_mono = 0.0
        n_mono_violations = 0

    if m.size >= 3:
        dk_l = (k[1:-1] - k[:-2])[:, None]
        dk_r = (k[2:] - k[1:-1])[:, None]
        denom = dk_l * dk_r * (dk_l + dk_r) / 2.0
        d2 = (
            dk_l * c[2:, :]
            - (dk_l + dk_r) * c[1:-1, :]
            + dk_r * c[:-2, :]
        ) / denom
        finite_d2 = np.isfinite(d2)
        worst_butterfly = float(np.min(np.where(finite_d2, d2, np.inf)))
        n_butterfly = int(np.sum(finite_d2 & (d2 < -tol)))
    else:
        worst_butterfly = 0.0
        n_butterfly = 0

    butterfly_ok = (worst_butterfly >= -tol) and (worst_mono <= tol)

    if t.size >= 2:
        w = (iv ** 2) * t[None, :]
        dw = np.diff(w, axis=1)
        finite_dw = np.isfinite(dw)
        worst_calendar = float(np.min(np.where(finite_dw, dw, np.inf)))
        n_calendar = int(np.sum(finite_dw & (dw < -tol)))
    else:
        worst_calendar = 0.0
        n_calendar = 0
    calendar_ok = worst_calendar >= -tol

    return ArbitrageReport(
        butterfly_ok=butterfly_ok,
        calendar_ok=calendar_ok,
        bounds_ok=bounds_ok,
        arbitrage_free=butterfly_ok and calendar_ok and bounds_ok,
        worst_butterfly=worst_butterfly,
        worst_monotonicity=worst_mono,
        worst_calendar=worst_calendar,
        worst_bound=worst_bound,
        n_butterfly_violations=n_butterfly,
        n_calendar_violations=n_calendar,
        n_bound_violations=n_bound_violations,
    )


def check_iv_surfaces_arbitrage(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
) -> list[ArbitrageReport]:
    """Run :func:`check_iv_surface_arbitrage` over every leading-axis slice.

    Accepts arrays of shape ``(..., n_moneyness, n_tau)`` (e.g. the LHS batch
    output ``(n_samples, n_m, n_tau)`` or the sequential-path output
    ``(n_paths, n_steps, n_m, n_tau)``). The returned list is flattened over
    all leading axes in row-major order.
    """
    iv = np.asarray(iv, dtype=float)
    if iv.ndim < 2:
        raise ValueError(f"iv must have at least 2 dims; got shape {iv.shape}")
    m = np.asarray(moneyness, dtype=float).ravel()
    t = np.asarray(tau, dtype=float).ravel()
    leading = iv.shape[:-2]
    flat = iv.reshape(-1, m.size, t.size)
    return [
        check_iv_surface_arbitrage(
            flat[i],
            m,
            t,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
            tol=tol,
        )
        for i in range(int(np.prod(leading)) if leading else 1)
    ]
