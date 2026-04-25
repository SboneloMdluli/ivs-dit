"""Latin Hypercube sampling over config-defined parameter boxes.

LHS generation stays on NumPy (via ``scipy.stats.qmc``) and returns ``np.ndarray``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc


def _lhs_unit_to_params(
    u: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    log_uniform_names: frozenset[str],
    param_order: Sequence[str],
) -> np.ndarray:
    """Map LHS unit-cube draws to physical parameters (linear or log-uniform per dim)."""
    out = np.empty_like(u, dtype=float)
    for i, name in enumerate(param_order):
        lo, hi = float(lows[i]), float(highs[i])
        col = u[:, i]
        if name in log_uniform_names:
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError(f"log_uniform for '{name}' requires strictly positive range bounds")
            out[:, i] = np.exp(np.log(lo) + col * (np.log(hi) - np.log(lo)))
        else:
            out[:, i] = lo + col * (hi - lo)
    return out


def lhs_params_from_config(
    cfg: Mapping[str, Any],
    *,
    param_order: Sequence[str],
    ranges_key: str = "param_ranges",
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Latin Hypercube sample over box ranges in ``cfg[ranges_key]``.

    Rows follow ``param_order``. Reads ``cfg['lhs']`` for ``n_samples``, ``seed``,
    and optional ``log_uniform`` (parameter names mapped with log-uniform margins).
    """
    ranges = cfg[ranges_key]
    lhs_cfg = cfg.get("lhs", {})
    n = int(n_samples if n_samples is not None else lhs_cfg["n_samples"])
    rng_seed = int(seed if seed is not None else lhs_cfg["seed"])

    lows = np.array([float(ranges[name][0]) for name in param_order], dtype=float)
    highs = np.array([float(ranges[name][1]) for name in param_order], dtype=float)
    if np.any(highs <= lows):
        raise ValueError(f"{ranges_key!r} requires upper > lower for all parameters")

    log_list = lhs_cfg.get("log_uniform") or []
    log_uniform = frozenset(str(x) for x in log_list)
    unknown = log_uniform - frozenset(param_order)
    if unknown:
        raise ValueError(f"log_uniform contains unknown parameters: {sorted(unknown)}")

    d = len(param_order)
    sampler = qmc.LatinHypercube(d=d, seed=rng_seed)
    u = sampler.random(n=n)
    return _lhs_unit_to_params(u, lows, highs, log_uniform, list(param_order))


def lhs_params_multi_batch_from_config(
    cfg: Mapping[str, Any],
    *,
    param_order: Sequence[str],
    ranges_key: str = "param_ranges",
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    seed_stride: int | None = None,
) -> np.ndarray:
    """Stack several independent LHS designs with different RNG seeds."""
    lhs_cfg = cfg.get("lhs", {})
    nb = int(n_batches if n_batches is not None else lhs_cfg.get("n_batches", 1))
    stride = int(seed_stride if seed_stride is not None else lhs_cfg.get("seed_stride", 10_000))
    if nb < 1:
        raise ValueError("n_batches must be >= 1")
    base = int(seed if seed is not None else lhs_cfg["seed"])
    chunks = [
        lhs_params_from_config(
            cfg,
            param_order=param_order,
            ranges_key=ranges_key,
            n_samples=n_samples,
            seed=base + b * stride,
        )
        for b in range(nb)
    ]
    return np.vstack(chunks)
