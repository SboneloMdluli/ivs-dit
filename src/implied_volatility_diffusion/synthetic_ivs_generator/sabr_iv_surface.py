"""SABR implied-vol surfaces: synthetic LHS and hooks for calibrated market slices."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from implied_volatility_diffusion.iv_surface import (
    grid_axes,
    implied_vol_surfaces_from_param_matrix,
    lhs_params_from_config,
    lhs_params_multi_batch_from_config,
)
from implied_volatility_diffusion.synthetic_ivs_generator.sabr import (
    calibrate_sabr_to_implied_vols,
    sabr_lognormal_iv,
)

SABR_PARAM_ORDER = ("alpha", "rho", "nu")


def forward_from_spot(spot: float, tau: float, r: float, q: float) -> float:
    """Risk-neutral forward for tenor ``tau``."""
    return float(spot * np.exp((r - q) * tau))


def implied_vol_surface_for_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implied-vol grid for one SABR parameter vector ``(alpha, rho, nu)``."""
    market = cfg["market"]
    spot = float(market["spot"])
    q = float(market.get("dividend_yield", 0.0))
    r = float(market.get("r", market.get("rate", market.get("risk_free_rate", 0.0))))
    sabr_cfg = cfg.get("sabr", {})
    beta = float(sabr_cfg.get("beta", 0.5))

    m, tau = grid_axes(cfg)
    alpha, rho, nu = (float(x) for x in params)

    out = np.empty((m.size, tau.size), dtype=float)
    for j, tj in enumerate(tau):
        if tj <= 0:
            out[:, j] = np.nan
            continue
        tjf = float(tj)
        fwd = forward_from_spot(spot, tjf, r, q)
        strikes = m * spot
        for i, k in enumerate(strikes):
            out[i, j] = sabr_lognormal_iv(fwd, float(k), tjf, alpha, beta, rho, nu)
    return m, tau, out


def lhs_sabr_params(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube sample over ``cfg['sabr_ranges']``; rows ``(alpha, rho, nu)``."""
    return lhs_params_from_config(
        cfg,
        param_order=SABR_PARAM_ORDER,
        ranges_key="sabr_ranges",
        n_samples=n_samples,
        seed=seed,
    )


def lhs_sabr_params_multi_batch(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> np.ndarray:
    """Several independent LHS batches over ``cfg['sabr_ranges']``."""
    return lhs_params_multi_batch_from_config(
        cfg,
        param_order=SABR_PARAM_ORDER,
        ranges_key="sabr_ranges",
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        seed_stride=seed_stride,
    )


def implied_vol_surfaces_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Latin-hypercube SABR draws and one implied-vol surface per draw."""
    lhs_cfg = cfg.get("lhs", {})
    nb = int(n_batches if n_batches is not None else lhs_cfg.get("n_batches", 1))
    if nb <= 1:
        params = lhs_sabr_params(cfg, n_samples=n_samples, seed=seed)
    else:
        params = lhs_sabr_params_multi_batch(
            cfg,
            n_samples=n_samples,
            n_batches=nb,
            seed=seed,
            seed_stride=seed_stride,
        )

    def build_surface(row: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
        _, _, surf = implied_vol_surface_for_params(row, c)
        return surf

    return implied_vol_surfaces_from_param_matrix(params, cfg, build_surface=build_surface)


def calibrate_params_for_expiries(
    spot: float,
    r: float,
    q: float,
    expiry_taus: np.ndarray,
    strikes_per_expiry: list[np.ndarray],
    ivs_per_expiry: list[np.ndarray],
    *,
    beta: float,
) -> tuple[np.ndarray, list[Any]]:
    """Calibrate one SABR smile per expiry.

    Args:
        spot: Spot at valuation time.
        r: Risk-free rate (continuous).
        q: Continuous dividend yield on the underlying.
        expiry_taus: 1D maturities (years), length ``E``.
        strikes_per_expiry: length-``E`` list of 1D strike arrays.
        ivs_per_expiry: length-``E`` list of 1D Black implied-vol arrays.
        beta: Fixed SABR beta.

    Returns:
        ``params`` with shape ``(E, 3)`` (columns ``alpha``, ``rho``, ``nu``) and
        a list of SciPy ``OptimizeResult`` diagnostics.
    """
    taus = np.asarray(expiry_taus, dtype=float).ravel()
    if not (len(strikes_per_expiry) == len(ivs_per_expiry) == int(taus.size)):
        raise ValueError("expiry_taus, strikes_per_expiry, and ivs_per_expiry lengths must match")
    rows: list[list[float]] = []
    details: list[Any] = []
    for tau, expiry_strikes, expiry_ivs in zip(taus, strikes_per_expiry, ivs_per_expiry, strict=True):
        fwd = forward_from_spot(spot, float(tau), r, q)
        alpha, rho, nu, res = calibrate_sabr_to_implied_vols(
            fwd,
            float(tau),
            expiry_strikes,
            expiry_ivs,
            beta=beta,
        )
        rows.append([alpha, rho, nu])
        details.append(res)
    return np.asarray(rows, dtype=float), details


def implied_vol_surface_from_calibrated_slices(
    spot: float,
    r: float,
    q: float,
    moneyness: np.ndarray,
    tau_axis: np.ndarray,
    expiry_taus: np.ndarray,
    calibrated_params: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Fill a moneyness × maturity IV grid using per-expiry calibrated SABR parameters.

    Each column ``tau_j`` uses the calibrated triple from the nearest market expiry
    in ``expiry_taus``.
    """
    m = np.asarray(moneyness, dtype=float).ravel()
    taus_grid = np.asarray(tau_axis, dtype=float).ravel()
    taus_exp = np.asarray(expiry_taus, dtype=float).ravel()
    p = np.asarray(calibrated_params, dtype=float)
    if p.shape[0] != taus_exp.size or p.shape[1] != 3:
        raise ValueError("calibrated_params must have shape (n_expiries, 3)")

    out = np.empty((m.size, taus_grid.size), dtype=float)
    for j, tj in enumerate(taus_grid):
        if tj <= 0:
            out[:, j] = np.nan
            continue
        idx = int(np.argmin(np.abs(taus_exp - tj)))
        alpha, rho, nu = (float(x) for x in p[idx])
        fwd = forward_from_spot(spot, float(tj), r, q)
        strikes = m * spot
        for i, k in enumerate(strikes):
            out[i, j] = sabr_lognormal_iv(fwd, float(k), float(tj), alpha, beta, rho, nu)
    return out


def plot_sabr_surface(
    moneyness: np.ndarray,
    tau_axis: np.ndarray,
    iv_surface: np.ndarray,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> Any:
    """Plot one implied-vol surface on a 3D moneyness x maturity grid.

    Args:
        moneyness: One-dimensional moneyness axis.
        tau_axis: One-dimensional maturity axis.
        iv_surface: Surface array with shape ``(len(moneyness), len(tau_axis))``.
        cfg: Optional config mapping. If provided, reads ``cfg['plot_surface']`` keys:
            ``zlim``, ``elev``, ``azim``, ``cmap``.

    Returns:
        The Matplotlib ``Figure`` object.
    """
    m = np.asarray(moneyness, dtype=float).ravel()
    t = np.asarray(tau_axis, dtype=float).ravel()
    z = np.asarray(iv_surface, dtype=float)
    if z.shape != (m.size, t.size):
        raise ValueError("iv_surface must have shape (len(moneyness), len(tau_axis))")

    plot_cfg = (cfg or {}).get("plot_surface", {})
    elev = float(plot_cfg.get("elev", 22))
    azim = float(plot_cfg.get("azim", -125))
    cmap = str(plot_cfg.get("cmap", "turbo"))
    zlim = plot_cfg.get("zlim")

    mm, tt = np.meshgrid(m, t, indexing="ij")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(mm, tt, z, cmap=cmap, linewidth=0.0, antialiased=True)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Maturity (tau)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("SABR Implied Volatility Surface")
    ax.view_init(elev=elev, azim=azim)
    if isinstance(zlim, (list, tuple)) and len(zlim) == 2:
        ax.set_zlim(float(zlim[0]), float(zlim[1]))
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="IV")
    fig.tight_layout()
    return fig


def plot_sabr_smiles(
    moneyness: np.ndarray,
    tau_axis: np.ndarray,
    iv_surface: np.ndarray,
    *,
    smile_taus: np.ndarray | None = None,
) -> Any:
    """Plot multiple SABR smiles from a surface as IV vs moneyness lines.

    Args:
        moneyness: One-dimensional moneyness axis.
        tau_axis: One-dimensional maturity axis.
        iv_surface: Surface array with shape ``(len(moneyness), len(tau_axis))``.
        smile_taus: Optional maturities to visualize. If ``None``, plots all columns.
            For non-grid maturities, nearest ``tau_axis`` values are used.

    Returns:
        The Matplotlib ``Figure`` object.
    """
    m = np.asarray(moneyness, dtype=float).ravel()
    t = np.asarray(tau_axis, dtype=float).ravel()
    z = np.asarray(iv_surface, dtype=float)
    if z.shape != (m.size, t.size):
        raise ValueError("iv_surface must have shape (len(moneyness), len(tau_axis))")

    if smile_taus is None:
        col_indices = list(range(t.size))
    else:
        req = np.asarray(smile_taus, dtype=float).ravel()
        col_indices = [int(np.argmin(np.abs(t - x))) for x in req]
        # Preserve order while dropping duplicates from nearest-neighbor mapping.
        seen: set[int] = set()
        col_indices = [i for i in col_indices if not (i in seen or seen.add(i))]

    fig, ax = plt.subplots(figsize=(9, 5))
    for j in col_indices:
        ax.plot(m, z[:, j], marker="o", linewidth=1.7, label=f"tau={t[j]:.2f}")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Volatility")
    ax.set_title("SABR Smiles by Maturity")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig
