from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.stats import qmc


class ModelCallPricer(Protocol):
    """Discounted European call price from absolute strike and time to maturity."""

    def __call__(self, strike: float, tau: float) -> float:
        """Return model discounted call price."""
        ...


class ImpliedVolInverter(Protocol):
    """Map model call price to Black–Scholes implied volatility."""

    def __call__(
        self,
        market_price: float,
        spot: float,
        strike: float,
        tau: float,
        rate: float,
        *,
        dividend_yield: float = 0.0,
        **kwargs: Any,
    ) -> float:
        """Return Black–Scholes implied volatility."""
        ...


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
    """Moneyness and maturity axes from ``cfg['grid']``."""
    grid = cfg["grid"]
    return _grid_axis(grid["moneyness"]), _grid_axis(grid["tau"])


def _lhs_unit_to_params(
    u: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    log_uniform_names: frozenset[str],
    param_order: Sequence[str],
) -> np.ndarray:
    """Map LHS draws u in [0,1]^d to physical parameters (linear or log-uniform per dim)."""
    out = np.empty_like(u, dtype=float)
    for i, name in enumerate(param_order):
        lo, hi = float(lows[i]), float(highs[i])
        col = u[:, i]
        if name in log_uniform_names:
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError(
                    f"log_uniform for '{name}' requires strictly positive range bounds"
                )
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

    Rows follow ``param_order``. Uses ``cfg['lhs']`` for ``n_samples``, ``seed``,
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


def implied_vol_surface_on_grid(
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    model_call_price: ModelCallPricer,
    to_implied_vol: ImpliedVolInverter,
    implied_vol_options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Black–Scholes implied volatility

    Args:
        moneyness: Values ``K / S`` (strike over spot).
        tau: Year fractions; non-positive entries yield NaN in the surface.
        spot: Spot level used to convert moneyness to strike.
        rate: Risk-free rate passed to ``to_implied_vol``.
        dividend_yield: Continuous yield passed through.
        model_call_price: ``(strike, tau) ->`` model **discounted** call price in the
            same units expected by ``to_implied_vol`` (e.g. Black–Scholes convention).
        to_implied_vol: Typically a Black–Scholes inverter; extra options go in
            ``implied_vol_options`` and are forwarded as keyword arguments.
        implied_vol_options: Optional mapping (e.g. ``sigma_lo``, ``brent_xtol``).

    Returns:
        Array of shape ``(len(moneyness), len(tau))``.
    """
    iv_kw = dict(implied_vol_options or {})
    m = np.asarray(moneyness, dtype=float)
    t = np.asarray(tau, dtype=float)
    out = np.empty((m.size, t.size), dtype=float)
    for i, mi in enumerate(m):
        strike = float(mi * spot)
        for j, tj in enumerate(t):
            if tj <= 0:
                out[i, j] = np.nan
                continue
            tjf = float(tj)
            price = model_call_price(strike, tjf)
            out[i, j] = to_implied_vol(
                float(price),
                spot,
                strike,
                tjf,
                rate,
                dividend_yield=dividend_yield,
                **iv_kw,
            )
    return out


def implied_vol_surfaces_from_param_matrix(
    params: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    build_surface: Callable[[np.ndarray, Mapping[str, Any]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
 
    m, tau = grid_axes(cfg)
    iv = np.empty((params.shape[0], m.size, tau.size), dtype=float)
    for n, row in enumerate(params):
        iv[n, :, :] = build_surface(row, cfg)
    return params, m, tau, iv
