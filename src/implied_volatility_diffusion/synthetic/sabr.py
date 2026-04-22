"""SABR + LHS recipes (batched and sequential)."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from implied_volatility_diffusion.core.grid import grid_axes
from implied_volatility_diffusion.core.lhs import (
    lhs_params_from_config,
    lhs_params_multi_batch_from_config,
)
from implied_volatility_diffusion.models.sabr import SABR_PARAM_ORDER, SabrModel, sabr_step
from implied_volatility_diffusion.synthetic.guards import (
    GuardSettings,
    guarded_build_surfaces,
)
from implied_volatility_diffusion.synthetic.surface import build_surfaces

_ALPHA_COL = SABR_PARAM_ORDER.index("alpha")
_RHO_COL = SABR_PARAM_ORDER.index("rho")
_NU_COL = SABR_PARAM_ORDER.index("nu")


def lhs_sabr_params(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """LHS sample over ``cfg['sabr_ranges']``; rows ``(alpha, rho, nu)``."""
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


def implied_vol_surface_for_sabr_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    spot: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implied-vol grid for one SABR parameter vector ``(alpha, rho, nu)``."""
    model = SabrModel.from_config(cfg)
    sb = build_surfaces(
        model,
        cfg,
        np.asarray(params, dtype=float).reshape(1, -1),
        spot_override=spot,
    )
    return sb.moneyness, sb.tau, sb.iv[0]


def implied_vol_surfaces_sabr_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
    guard: GuardSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LHS SABR draws + one IV surface per draw, with arbitrage guard applied."""
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

    model = SabrModel.from_config(cfg)
    sb = guarded_build_surfaces(model, cfg, params, guard=guard)
    return sb.params, sb.moneyness, sb.tau, sb.iv


def implied_vol_surfaces_sabr_sequential_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
    n_steps: int | None = None,
    dt: float | None = None,
    guard: GuardSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LHS SABR draws and a sequence of IV surfaces per draw along a SABR path.

    For each LHS draw ``(alpha_0, rho, nu)`` this integrates the SABR SDE for
    ``n_steps`` Euler increments of size ``dt`` under the pricing measure
    (``dS = (r-q) S dt + alpha S^beta dW1``, ``d alpha = nu alpha dW2``,
    ``dW1 dW2 = rho dt``) and regenerates a SABR IV surface at every step
    using the path's ``(S_k, alpha_k)`` as the new spot / alpha.

    Each per-step surface is passed through the arbitrage guard configured by
    ``cfg['arbitrage_guard']`` (default ``"repair"``): calendar monotonicity
    is projected on first, then butterfly/bounds are checked and reported.

    Returns:
        ``(params, moneyness, tau, iv)`` where ``iv`` has shape
        ``(n_paths, n_steps, n_moneyness, n_tau)``.
    """
    seq_cfg = cfg.get("sequential_ivs") or {}
    n_st = int(n_steps if n_steps is not None else seq_cfg.get("n_steps", 8))
    if n_st < 1:
        raise ValueError("n_steps must be >= 1")
    dt_eff = float(dt if dt is not None else seq_cfg.get("dt", 1.0 / 252.0))
    if dt_eff <= 0.0:
        raise ValueError("dt must be positive")
    path_stride = int(seq_cfg.get("path_seed_stride", 100_000))

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

    m_axis, tau_axis = grid_axes(cfg)
    market = cfg["market"]
    s0 = float(market["spot"])
    r = float(market.get("r", market.get("rate", market.get("risk_free_rate", 0.0))))
    q = float(market.get("dividend_yield", 0.0))
    beta = float((cfg.get("sabr") or {}).get("beta", 0.5))

    lhs_seed = int(seed if seed is not None else lhs_cfg.get("seed", 0))
    n_paths = int(params.shape[0])
    iv = np.empty((n_paths, n_st, m_axis.size, tau_axis.size), dtype=float)

    model = SabrModel.from_config(cfg)
    guard_settings = guard if guard is not None else GuardSettings.from_config(cfg)

    for p in range(n_paths):
        alpha0, rho_p, nu_p = (float(x) for x in params[p])
        rng = np.random.default_rng(lhs_seed + (p + 1) * path_stride)
        s_cur, alpha_cur = s0, alpha0
        for k in range(n_st):
            row_k = np.array([[alpha_cur, rho_p, nu_p]], dtype=float)
            sb = guarded_build_surfaces(
                model,
                cfg,
                row_k,
                spot_override=s_cur,
                guard=guard_settings,
            )
            iv[p, k, :, :] = sb.iv[0]
            if k < n_st - 1:
                s_cur, alpha_cur = sabr_step(
                    s_cur,
                    alpha_cur,
                    dt_eff,
                    r,
                    q,
                    beta,
                    rho_p,
                    nu_p,
                    rng,
                )
    return params, m_axis, tau_axis, iv


# Backward-compatible aliases for legacy notebook imports.
def implied_vol_surface_for_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    spot: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for ``implied_vol_surface_for_sabr_params``."""
    return implied_vol_surface_for_sabr_params(params, cfg, spot=spot)


def implied_vol_surfaces_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
    guard: GuardSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alias for ``implied_vol_surfaces_sabr_lhs``."""
    return implied_vol_surfaces_sabr_lhs(
        cfg,
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        seed_stride=seed_stride,
        guard=guard,
    )


def implied_vol_surfaces_sequential_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
    n_steps: int | None = None,
    dt: float | None = None,
    guard: GuardSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alias for ``implied_vol_surfaces_sabr_sequential_lhs``."""
    return implied_vol_surfaces_sabr_sequential_lhs(
        cfg,
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        seed_stride=seed_stride,
        n_steps=n_steps,
        dt=dt,
        guard=guard,
    )


def plot_sabr_surface(
    moneyness: np.ndarray,
    tau: np.ndarray,
    iv: np.ndarray,
    *,
    cfg: Mapping[str, Any] | None = None,
):
    """Plot a SABR implied-volatility surface and return the figure."""
    m_axis = np.asarray(moneyness, dtype=float)
    tau_axis = np.asarray(tau, dtype=float)
    iv_surface = np.asarray(iv, dtype=float)
    if iv_surface.shape != (m_axis.size, tau_axis.size):
        raise ValueError("iv must have shape (len(moneyness), len(tau))")

    plot_cfg = (cfg or {}).get("plot_surface", {})
    cmap = str(plot_cfg.get("cmap", "viridis"))
    elev = float(plot_cfg.get("elev", 22.0))
    azim = float(plot_cfg.get("azim", -125.0))

    # 3D surfaces require at least a 2x2 grid. Fall back to a 2D slice plot for
    # degenerate grids (e.g. one maturity or one moneyness value).
    if m_axis.size < 2 or tau_axis.size < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        if tau_axis.size == 1:
            ax.plot(m_axis, iv_surface[:, 0], marker="o", lw=1.6, label=rf"$\tau={tau_axis[0]:.4f}$")
        else:
            ax.plot(tau_axis, iv_surface[0, :], marker="o", lw=1.6, label=rf"$m={m_axis[0]:.4f}$")
        ax.set_xlabel(r"Log-moneyness $k=\log(K/S)$" if tau_axis.size == 1 else "Time to maturity (years)")
        ax.set_ylabel(r"Implied volatility $\sigma_{\mathrm{imp}}$")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=9)
        ax.set_title("SABR Synthetic IV Slice")
        fig.tight_layout()
        return fig

    m_grid, tau_grid = np.meshgrid(m_axis, tau_axis, indexing="ij")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(m_grid, tau_grid, iv_surface, cmap=cmap, edgecolor="none")

    ax.set_xlabel("Log-moneyness k = log(K/S)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.view_init(elev=elev, azim=azim)
    title = "SABR Synthetic IV Surface"
    if cfg is not None:
        beta = (cfg.get("sabr") or {}).get("beta")
        if beta is not None:
            title = f"{title} (beta={float(beta):.2f})"
    ax.set_title(title)
    if "zlim" in plot_cfg:
        zlim = plot_cfg["zlim"]
        if isinstance(zlim, (list, tuple)) and len(zlim) == 2:
            ax.set_zlim(float(zlim[0]), float(zlim[1]))
    fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.1, label="Implied volatility")
    return fig


def plot_sabr_smiles(
    moneyness: np.ndarray,
    tau: np.ndarray,
    iv: np.ndarray,
    *,
    smile_taus: np.ndarray | None = None,
):
    """Plot SABR smile slices for selected maturities and return the figure."""
    m_axis = np.asarray(moneyness, dtype=float)
    tau_axis = np.asarray(tau, dtype=float)
    iv_surface = np.asarray(iv, dtype=float)
    if iv_surface.shape != (m_axis.size, tau_axis.size):
        raise ValueError("iv must have shape (len(moneyness), len(tau))")

    fig, ax = plt.subplots(figsize=(8, 5))
    if smile_taus is None:
        idx = np.linspace(0, tau_axis.size - 1, num=min(5, tau_axis.size), dtype=int)
        idx = np.unique(idx)
        for j in idx:
            ax.plot(m_axis, iv_surface[:, j], lw=1.6, label=rf"$\tau={tau_axis[j]:.4f}$")
    else:
        tau_targets = np.atleast_1d(smile_taus).astype(float)
        if tau_axis.size == 1:
            # Keep one trace per requested tau so callers still see their
            # requested set in legend, even when only one maturity exists.
            for t in tau_targets:
                ax.plot(
                    m_axis,
                    iv_surface[:, 0],
                    lw=1.6,
                    alpha=0.9,
                    label=rf"$\tau_{{req}}={t:.4f}$ (only grid $\tau={tau_axis[0]:.4f}$)",
                )
        else:
            for t in tau_targets:
                j = int(np.argmin(np.abs(tau_axis - t)))
                left = max(j - 1, 0)
                right = min(j + 1, tau_axis.size - 1)
                if left == right:
                    smile = iv_surface[:, j]
                else:
                    x0, x1 = tau_axis[left], tau_axis[right]
                    w = 0.0 if x1 == x0 else float(np.clip((t - x0) / (x1 - x0), 0.0, 1.0))
                    smile = (1.0 - w) * iv_surface[:, left] + w * iv_surface[:, right]
                ax.plot(m_axis, smile, lw=1.6, label=rf"$\tau_{{req}}={t:.4f}$")

    ax.set_xlabel(r"Log-moneyness $k=\log(K/S)$")
    ax.set_ylabel(r"Implied volatility $\sigma_{\mathrm{imp}}$")
    ax.set_title("SABR Smile Slices")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    "implied_vol_surface_for_params",
    "implied_vol_surface_for_sabr_params",
    "implied_vol_surfaces_lhs",
    "implied_vol_surfaces_sabr_lhs",
    "implied_vol_surfaces_sequential_lhs",
    "implied_vol_surfaces_sabr_sequential_lhs",
    "lhs_sabr_params",
    "lhs_sabr_params_multi_batch",
    "plot_sabr_smiles",
    "plot_sabr_surface",
]
