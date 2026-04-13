"""Heston COS + LHS configuration helpers (wraps generic :mod:`iv_surface` utilities)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ivs_config import merge_config_files

from implied_volatility_diffusion.iv_surface import (
    grid_axes,
    implied_vol_surface_on_grid,
    implied_vol_surfaces_from_param_matrix,
    lhs_params_from_config,
    lhs_params_multi_batch_from_config,
)
from implied_volatility_diffusion.synthetic_ivs_generator.black_scholes import implied_volatility
from implied_volatility_diffusion.synthetic_ivs_generator.heston_cos import heston_call_cos

HESTON_PARAM_ORDER = ("v0", "rho", "sigma", "theta", "kappa", "r")

HESTON_IV_SURFACE_YAML = "heston_iv_surface.yaml"
IV_SURFACE_GRID_YAML = "iv_surface_grid.yaml"


def load_heston_iv_surface_config(config_dir: str | Path) -> dict[str, Any]:
    d = Path(config_dir)
    return merge_config_files(d / HESTON_IV_SURFACE_YAML, d / IV_SURFACE_GRID_YAML)


def _heston_cos_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if "heston_cos_pricer" in cfg:
        return cfg["heston_cos_pricer"]
    return cfg.get("cos", {})


def lhs_heston_params(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube sample of Heston parameters; shape ``(n, 6)`` in ``HESTON_PARAM_ORDER``."""
    return lhs_params_from_config(
        cfg,
        param_order=HESTON_PARAM_ORDER,
        ranges_key="heston_ranges",
        n_samples=n_samples,
        seed=seed,
    )


def lhs_heston_params_multi_batch(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> np.ndarray:
    """Several independent LHS batches over ``cfg['heston_ranges']``."""
    return lhs_params_multi_batch_from_config(
        cfg,
        param_order=HESTON_PARAM_ORDER,
        ranges_key="heston_ranges",
        n_samples=n_samples,
        n_batches=n_batches,
        seed=seed,
        seed_stride=seed_stride,
    )


def implied_vol_surface_for_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implied-vol grid for one Heston parameter vector (COS prices + BS inversion)."""
    market = cfg["market"]
    spot = float(market["spot"])
    q = float(market.get("dividend_yield", 0.0))
    m, tau = grid_axes(cfg)

    v0, rho, sigma_h, theta, kappa, r = (float(x) for x in params)
    cos_cfg = _heston_cos_section(cfg)
    n_terms_base = int(cos_cfg.get("n_terms", 1024))
    L = float(cos_cfg.get("truncation_L", 14.0))
    tau_ref = float(cos_cfg.get("short_tau_tau_ref", 0.25))
    n_terms_max = int(cos_cfg.get("n_terms_max", 4096))

    iv_cfg = cfg.get("implied_vol", {})
    iv_opts = {
        "sigma_lo": float(iv_cfg.get("sigma_lo", 1e-4)),
        "sigma_hi": float(iv_cfg.get("sigma_hi", 5.0)),
        "xtol": float(iv_cfg.get("brent_xtol", 1e-8)),
        "newton_refinement_steps": int(
            iv_cfg.get("newton_refinement_steps", iv_cfg.get("newton_max_iter", 3))
        ),
        "newton_tol": float(iv_cfg.get("newton_tol", 1e-10)),
        "jackel_iterations": int(iv_cfg.get("jackel_iterations", 0)),
        "vega_floor_scale": float(iv_cfg.get("vega_floor_scale", 1e-14)),
    }

    def model_call_price(strike: float, ttm: float) -> float:
        scale = max(1.0, math.sqrt(tau_ref / max(ttm, 1e-12)))
        n_terms = int(min(n_terms_max, round(n_terms_base * scale)))
        return float(
            heston_call_cos(
                spot,
                strike,
                ttm,
                r,
                kappa,
                theta,
                sigma_h,
                rho,
                v0,
                dividend_yield=q,
                n_terms=n_terms,
                truncation_L=L,
            )
        )

    iv = implied_vol_surface_on_grid(
        m,
        tau,
        spot=spot,
        rate=r,
        dividend_yield=q,
        model_call_price=model_call_price,
        to_implied_vol=implied_volatility,
        implied_vol_options=iv_opts,
    )
    return m, tau, iv


def implied_vol_surfaces_lhs(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Latin-hypercube Heston draws and one implied-vol surface per draw."""
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

    def build_surface(row: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
        _, _, surf = implied_vol_surface_for_params(row, c)
        return surf

    return implied_vol_surfaces_from_param_matrix(params, cfg, build_surface=build_surface)
