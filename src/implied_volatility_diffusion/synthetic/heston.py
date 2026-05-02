"""Heston + LHS high-level recipes.

This module replaces ``synthetic_ivs_generator/heston_iv_surface.py``. The
legacy module now re-exports these names.
"""

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from implied_volatility_diffusion.config import merge_config_files
from implied_volatility_diffusion.core.grid import grid_axes
from implied_volatility_diffusion.core.lhs import (
    lhs_params_from_config,
    lhs_params_multi_batch_from_config,
)
from implied_volatility_diffusion.models.heston import HESTON_PARAM_ORDER, HestonModel, milstein_step
from implied_volatility_diffusion.synthetic.goals import HESTON_GOAL_YAML, HestonIvGoal, coerce_heston_iv_goal
from implied_volatility_diffusion.synthetic.guards import (
    GuardSettings,
    guarded_build_surfaces,
)

HESTON_IV_SURFACE_YAML = "heston_iv_surface.yaml"
IV_SURFACE_GRID_YAML = "iv_surface_grid.yaml"

_SIGMA_COL = HESTON_PARAM_ORDER.index("sigma")
_THETA_COL = HESTON_PARAM_ORDER.index("theta")
_KAPPA_COL = HESTON_PARAM_ORDER.index("kappa")


def _clip_sigma_to_feller(params: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """Clip ``sigma_v`` per row so ``2 kappa theta >= sigma_v^2 + eps``."""
    out = np.array(params, dtype=float, copy=True)
    two_kt = 2.0 * out[:, _KAPPA_COL] * out[:, _THETA_COL]
    feller_strict = two_kt - float(eps)
    sigma_max = np.sqrt(np.where(feller_strict > 0.0, feller_strict, two_kt))
    out[:, _SIGMA_COL] = np.minimum(out[:, _SIGMA_COL], sigma_max)
    return out


def load_heston_iv_surface_config(config_dir: str | Path) -> dict[str, Any]:
    """Load base Heston + grid YAML."""
    d = Path(config_dir)
    return merge_config_files(d / HESTON_IV_SURFACE_YAML, d / IV_SURFACE_GRID_YAML)


def load_heston_iv_surface_goal_config(
    config_dir: str | Path,
    goal: HestonIvGoal | str,
) -> dict[str, Any]:
    """Load base Heston + grid YAML, then merge a goal overlay."""
    g = coerce_heston_iv_goal(goal)
    d = Path(config_dir)
    return merge_config_files(
        d / HESTON_IV_SURFACE_YAML,
        d / IV_SURFACE_GRID_YAML,
        d / HESTON_GOAL_YAML[g],
    )


def lhs_heston_params(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube sample of Heston parameters (Feller-clipped)."""
    params = lhs_params_from_config(
        cfg,
        param_order=HESTON_PARAM_ORDER,
        ranges_key="heston_ranges",
        n_samples=n_samples,
        seed=seed,
    )
    eps = float((cfg.get("lhs") or {}).get("feller_eps", 0.0))
    return _clip_sigma_to_feller(params, eps=eps)


def lhs_heston_params_multi_batch(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> np.ndarray:
    """Several independent Feller-clipped LHS batches over ``cfg['heston_ranges']``."""
    params = lhs_params_multi_batch_from_config(
        cfg,
        param_order=HESTON_PARAM_ORDER,
        ranges_key="heston_ranges",
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        seed_stride=seed_stride,
    )
    eps = float((cfg.get("lhs") or {}).get("feller_eps", 0.0))
    return _clip_sigma_to_feller(params, eps=eps)


def implied_vol_surface_for_heston_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    spot: float | None = None,
    inst_var: float | None = None,
    guard: GuardSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implied-vol grid for one Heston parameter vector (arbitrage-guarded)."""
    model = HestonModel.from_config(cfg)
    sb = guarded_build_surfaces(
        model,
        cfg,
        np.asarray(params, dtype=float).reshape(1, -1),
        spot_override=spot,
        inst_var_override=(None if inst_var is None else np.array([float(inst_var)], dtype=float)),
        guard=guard,
    )
    return sb.moneyness, sb.tau, sb.iv[0]


def implied_vol_surfaces_heston_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
    guard: GuardSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LHS Heston draws + one IV surface per draw, with arbitrage guard applied."""
    lhs_cfg = cfg.get("lhs", {})
    nb = int(n_batches if n_batches is not None else lhs_cfg.get("n_batches", 1))
    if nb <= 1:
        params = lhs_heston_params(cfg, n_samples=n_samples, seed=seed)
    else:
        params = lhs_heston_params_multi_batch(
            cfg,
            n_samples=n_samples,
            n_batches=nb,
            seed=seed,
            seed_stride=seed_stride,
        )

    model = HestonModel.from_config(cfg)
    sb = guarded_build_surfaces(model, cfg, params, guard=guard)
    return sb.params, sb.moneyness, sb.tau, sb.iv


def implied_vol_surfaces_heston_sequential_lhs(
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
    """LHS Heston draws and a sequence of IV surfaces per draw along a Milstein path.

    Each per-step surface is passed through the arbitrage guard configured by
    ``cfg['arbitrage_guard']`` (default ``"repair"``).
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
        params = lhs_heston_params(cfg, n_samples=n_samples, seed=seed)
    else:
        params = lhs_heston_params_multi_batch(
            cfg,
            n_samples=n_samples,
            n_batches=nb,
            seed=seed,
            seed_stride=seed_stride,
        )

    m, tau = grid_axes(cfg)
    market = cfg["market"]
    s0 = float(market["spot"])
    q = float(market.get("dividend_yield", 0.0))

    lhs_seed = int(seed if seed is not None else lhs_cfg.get("seed", 0))
    n_paths = int(params.shape[0])
    iv = np.empty((n_paths, n_st, m.size, tau.size), dtype=float)

    guard_settings = guard if guard is not None else GuardSettings.from_config(cfg)

    for p in range(n_paths):
        row = params[p]
        v0, rho, sigma_h, theta, kappa, r = (float(x) for x in row)
        rng = np.random.default_rng(lhs_seed + (p + 1) * path_stride)
        s_cur, v_cur = s0, v0
        for k in range(n_st):
            _, _, surf = implied_vol_surface_for_heston_params(
                row,
                cfg,
                spot=s_cur,
                inst_var=v_cur,
                guard=guard_settings,
            )
            iv[p, k, :, :] = surf
            if k < n_st - 1:
                s_cur, v_cur = milstein_step(
                    s_cur,
                    v_cur,
                    dt_eff,
                    r,
                    q,
                    kappa,
                    theta,
                    sigma_h,
                    rho,
                    rng,
                )
    return params, m, tau, iv
