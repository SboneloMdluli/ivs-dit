"""Load config, Latin Hypercube Heston draws, and implied-volatility surfaces."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.stats import qmc

from implied_volatility_diffusion.black_scholes import implied_volatility
from implied_volatility_diffusion.heston_cos import heston_call_cos


PARAM_ORDER = ("v0", "rho", "sigma", "theta", "kappa", "r")


def _grid_axis(spec: Mapping[str, Any]) -> np.ndarray:
    """Build 1D grid from start_point, step, end_point (end always included)."""
    start = float(spec["start_point"])
    step = float(spec["step"])
    end = float(spec["end_point"])
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    if end + 1e-15 < start:
        raise ValueError("end_point must be >= start_point")
    pts: list[float] = [start]
    tol = 1e-12 * max(1.0, abs(end))
    while True:
        nxt = pts[-1] + step
        if nxt > end + tol:
            break
        pts.append(float(nxt))
    if abs(pts[-1] - end) > tol:
        pts.append(end)
    base = np.asarray(pts, dtype=float)
    prepend = spec.get("prepend")
    if prepend:
        extra = np.asarray([float(x) for x in prepend], dtype=float)
        base = np.unique(np.sort(np.concatenate([extra, base])))
    return base


def grid_axes(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Moneyness and maturity axes from cfg['grid']."""
    grid = cfg["grid"]
    return _grid_axis(grid["moneyness"]), _grid_axis(grid["tau"])


def _lhs_unit_to_params(
    u: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    log_uniform_names: frozenset[str],
) -> np.ndarray:
    """Map LHS draws u in [0,1]^d to physical parameters (linear or log-uniform per dim)."""
    out = np.empty_like(u, dtype=float)
    for i, name in enumerate(PARAM_ORDER):
        lo, hi = float(lows[i]), float(highs[i])
        col = u[:, i]
        if name in log_uniform_names:
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError(
                    f"log_uniform for '{name}' requires strictly positive heston_ranges bounds"
                )
            out[:, i] = np.exp(np.log(lo) + col * (np.log(hi) - np.log(lo)))
        else:
            out[:, i] = lo + col * (hi - lo)
    return out


def lhs_heston_params(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube sample of Heston parameters; shape (n, 6) in PARAM_ORDER."""
    ranges = cfg["heston_ranges"]
    lhs_cfg = cfg.get("lhs", {})
    n = int(n_samples if n_samples is not None else lhs_cfg["n_samples"])
    rng_seed = int(seed if seed is not None else lhs_cfg["seed"])

    lows = np.array([float(ranges[name][0]) for name in PARAM_ORDER], dtype=float)
    highs = np.array([float(ranges[name][1]) for name in PARAM_ORDER], dtype=float)
    if np.any(highs <= lows):
        raise ValueError("heston_ranges require upper > lower for all parameters")

    log_list = lhs_cfg.get("log_uniform") or []
    log_uniform = frozenset(str(x) for x in log_list)
    unknown = log_uniform - frozenset(PARAM_ORDER)
    if unknown:
        raise ValueError(f"log_uniform contains unknown parameters: {sorted(unknown)}")

    d = len(PARAM_ORDER)
    sampler = qmc.LatinHypercube(d=d, seed=rng_seed)
    u = sampler.random(n=n)
    return _lhs_unit_to_params(u, lows, highs, log_uniform)


def lhs_heston_params_multi_batch(
    cfg: Mapping[str, Any],
    *,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> np.ndarray:
    """Stack several independent LHS designs with different RNG seeds.

    Total draws = n_batches * n_samples (each batch fills the hypercube anew).
    """
    lhs_cfg = cfg.get("lhs", {})
    nb = int(n_batches if n_batches is not None else lhs_cfg.get("n_batches", 1))
    stride = int(seed_stride if seed_stride is not None else lhs_cfg.get("seed_stride", 10_000))
    if nb < 1:
        raise ValueError("n_batches must be >= 1")
    base = int(seed if seed is not None else lhs_cfg["seed"])
    chunks = [
        lhs_heston_params(cfg, n_samples=n_samples, seed=base + b * stride) for b in range(nb)
    ]
    return np.vstack(chunks)


def implied_vol_surface_for_params(
    params: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the implied-vol grid for one Heston parameter vector.

    Returns:
        m: Shape (M,), moneyness ``K/S0``.
        tau: Shape (T,), maturities in years.
        iv: Shape (M, T), Black–Scholes implied volatility.
    """
    market = cfg["market"]
    spot = float(market["spot"])
    q = float(market.get("dividend_yield", 0.0))
    m, tau = grid_axes(cfg)

    cos_cfg = cfg.get("cos", {})
    n_terms_base = int(cos_cfg.get("n_terms", 1024))
    L = float(cos_cfg.get("truncation_L", 14.0))
    tau_ref = float(cos_cfg.get("short_tau_tau_ref", 0.25))
    n_terms_max = int(cos_cfg.get("n_terms_max", 4096))

    iv_cfg = cfg.get("implied_vol", {})
    sigma_lo = float(iv_cfg.get("sigma_lo", 1e-4))
    sigma_hi = float(iv_cfg.get("sigma_hi", 5.0))
    xtol = float(iv_cfg.get("brent_xtol", 1e-8))
    newton_refinement_steps = int(
        iv_cfg.get("newton_refinement_steps", iv_cfg.get("newton_max_iter", 3))
    )
    newton_tol = float(iv_cfg.get("newton_tol", 1e-10))
    jackel_iterations = int(iv_cfg.get("jackel_iterations", 0))
    vega_floor_scale = float(iv_cfg.get("vega_floor_scale", 1e-14))

    v0, rho, sigma_h, theta, kappa, r = (float(x) for x in params)

    iv = np.empty((m.size, tau.size), dtype=float)
    for i, mi in enumerate(m):
        strike = float(mi * spot)
        for j, tj in enumerate(tau):
            if tj <= 0:
                iv[i, j] = np.nan
                continue
            tjf = float(tj)
            scale = max(1.0, math.sqrt(tau_ref / max(tjf, 1e-12)))
            n_terms = int(min(n_terms_max, round(n_terms_base * scale)))
            price = heston_call_cos(
                spot,
                strike,
                tjf,
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
            iv[i, j] = implied_volatility(
                float(price),
                spot,
                strike,
                tjf,
                r,
                dividend_yield=q,
                sigma_lo=sigma_lo,
                sigma_hi=sigma_hi,
                xtol=xtol,
                newton_refinement_steps=newton_refinement_steps,
                newton_tol=newton_tol,
                jackel_iterations=jackel_iterations,
                vega_floor_scale=vega_floor_scale,
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
    """Latin-hypercube Heston draws and one implied-vol surface per draw.

    If ``lhs.n_batches`` > 1 (or ``n_batches`` is passed), uses independent Latin
    hypercubes with different seeds for more diverse parameter sets.

    Returns:
        params: Shape (N, 6), rows follow ``PARAM_ORDER``.
        m: Shape (M,), moneyness grid.
        tau: Shape (T,), maturity grid.
        iv: Shape (N, M, T), implied volatility.
    """
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
    iv = np.empty((params.shape[0], m.size, tau.size), dtype=float)
    for n, row in enumerate(params):
        _, _, surf = implied_vol_surface_for_params(row, cfg)
        iv[n, :, :] = surf
    return params, m, tau, iv
