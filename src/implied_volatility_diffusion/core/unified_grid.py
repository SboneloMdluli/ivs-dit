"""Unified `(k, tau)` grid and resampling helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from implied_volatility_diffusion.core.grid import build_grid_axis

DEFAULT_LOG_MONEYNESS_SPEC: dict[str, float] = {
    "start_point": -0.5,
    "step": 0.025,
    "end_point": 0.5,
}
DEFAULT_TAU_SPEC: dict[str, float] = {
    "start_point": 0.05,
    "step": 0.05,
    "end_point": 2.0,
}

UNIFIED_IV_GRID_YAML = "unified_iv_grid.yaml"


@dataclass(frozen=True)
class UnifiedGrid:
    """Shared ``(log-moneyness, tau)`` axes."""

    log_moneyness: np.ndarray
    tau: np.ndarray

    def __post_init__(self) -> None:
        """Coerce axes to float arrays."""
        k = np.asarray(self.log_moneyness, dtype=float)
        t = np.asarray(self.tau, dtype=float)
        object.__setattr__(self, "log_moneyness", k)
        object.__setattr__(self, "tau", t)

    @property
    def k(self) -> np.ndarray:
        """Alias for :attr:`log_moneyness` (common shorthand)."""
        return self.log_moneyness

    @property
    def moneyness(self) -> np.ndarray:
        """``m = K / S = exp(k)`` - drop-in replacement for legacy moneyness axes."""
        return np.exp(self.log_moneyness)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_k, n_tau)`` - the shape of every surface built on this grid."""
        return (int(self.log_moneyness.size), int(self.tau.size))

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(K, T)`` meshgrids with ``indexing='ij'``."""
        return np.meshgrid(self.log_moneyness, self.tau, indexing="ij")

    @classmethod
    def default(cls) -> "UnifiedGrid":
        """The canonical grid (k in [-0.5, 0.5], tau in [0.05, 2.0])."""
        return cls(
            log_moneyness=build_grid_axis(DEFAULT_LOG_MONEYNESS_SPEC),
            tau=build_grid_axis(DEFAULT_TAU_SPEC),
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "UnifiedGrid":
        """Build from ``cfg['unified_grid']`` or legacy ``cfg['grid']``."""
        if "unified_grid" in cfg:
            spec = cfg["unified_grid"]
            k_spec = spec.get("log_moneyness") or spec.get("k")
            if k_spec is None:
                raise KeyError("cfg['unified_grid'] is missing 'log_moneyness'")
            tau_spec = spec["tau"]
            return cls(
                log_moneyness=build_grid_axis(k_spec),
                tau=build_grid_axis(tau_spec),
            )
        if "grid" in cfg:
            grid = cfg["grid"]
            tau_spec = grid["tau"]
            if "log_moneyness" in grid:
                k_axis = build_grid_axis(grid["log_moneyness"])
            else:
                m_axis = build_grid_axis(grid["moneyness"])
                if np.any(m_axis <= 0.0):
                    raise ValueError("cannot convert non-positive moneyness to log-moneyness")
                k_axis = np.log(m_axis)
            return cls(log_moneyness=k_axis, tau=build_grid_axis(tau_spec))
        raise KeyError("cfg must contain 'unified_grid' or legacy 'grid' block")

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedGrid":
        """Load a grid from a YAML file (e.g. ``config/unified_iv_grid.yaml``)."""
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_config(yaml.safe_load(raw))

    def as_legacy_grid_cfg(self) -> dict[str, Any]:
        """Return ``{'grid': {'moneyness': ..., 'tau': ...}}`` for legacy APIs."""

        def _spec(axis: np.ndarray) -> dict[str, Any]:
            axis = np.asarray(axis, dtype=float)
            span = float(axis[-1] - axis[0])
            return {
                "start_point": float(axis[0]),
                "step": 10.0 * (span if span > 0.0 else 1.0),
                "end_point": float(axis[-1]),
                "prepend": [float(x) for x in axis],
            }

        return {
            "grid": {
                "moneyness": _spec(np.exp(self.log_moneyness)),
                "tau": _spec(self.tau),
            }
        }

    def with_config(self, cfg: Mapping[str, Any]) -> dict[str, Any]:
        """Return a copy of ``cfg`` with ``grid`` replaced by this unified grid."""
        out: dict[str, Any] = dict(cfg)
        out["grid"] = self.as_legacy_grid_cfg()["grid"]
        return out


def _bilinear_resample(
    values: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_dst: np.ndarray,
    y_dst: np.ndarray,
) -> np.ndarray:
    """Linear interpolation from ``(x_src, y_src)`` to ``(x_dst, y_dst)``."""
    vals = np.asarray(values, dtype=float)
    xs = np.asarray(x_src, dtype=float)
    ys = np.asarray(y_src, dtype=float)
    xd = np.asarray(x_dst, dtype=float)
    yd = np.asarray(y_dst, dtype=float)

    x_clip = np.clip(xd, xs[0], xs[-1])
    y_clip = np.clip(yd, ys[0], ys[-1])
    ix = np.clip(np.searchsorted(xs, x_clip, side="right") - 1, 0, xs.size - 2)
    iy = np.clip(np.searchsorted(ys, y_clip, side="right") - 1, 0, ys.size - 2)

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    wx = np.where(x1 > x0, (x_clip - x0) / (x1 - x0), 0.0)
    wy = np.where(y1 > y0, (y_clip - y0) / (y1 - y0), 0.0)

    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    WX, WY = np.meshgrid(wx, wy, indexing="ij")
    z00 = vals[IX, IY]
    z10 = vals[IX + 1, IY]
    z01 = vals[IX, IY + 1]
    z11 = vals[IX + 1, IY + 1]
    out = (1.0 - WX) * (1.0 - WY) * z00 + WX * (1.0 - WY) * z10 + (1.0 - WX) * WY * z01 + WX * WY * z11

    # Tolerance avoids boundary NaNs from floating-point grid drift.
    tol_x = 1e-9 * max(1.0, float(xs[-1] - xs[0]))
    tol_y = 1e-9 * max(1.0, float(ys[-1] - ys[0]))
    in_x = (xd >= xs[0] - tol_x) & (xd <= xs[-1] + tol_x)
    in_y = (yd >= ys[0] - tol_y) & (yd <= ys[-1] + tol_y)
    mask = np.outer(in_x, in_y)
    out = np.where(mask, out, np.nan)
    return out


def resample_to_unified_grid(
    iv: np.ndarray,
    *,
    k_src: np.ndarray | None = None,
    m_src: np.ndarray | None = None,
    tau_src: np.ndarray,
    grid: UnifiedGrid | None = None,
) -> np.ndarray:
    """Resample one surface onto the unified ``(k, tau)`` grid."""
    if (k_src is None) == (m_src is None):
        raise ValueError("provide exactly one of k_src or m_src")
    if k_src is None:
        m_arr = np.asarray(m_src, dtype=float)
        if np.any(m_arr <= 0.0):
            raise ValueError("m_src must be strictly positive to take log")
        k_src = np.log(m_arr)
    g = grid if grid is not None else UnifiedGrid.default()
    return _bilinear_resample(
        values=np.asarray(iv, dtype=float),
        x_src=np.asarray(k_src, dtype=float),
        y_src=np.asarray(tau_src, dtype=float),
        x_dst=g.log_moneyness,
        y_dst=g.tau,
    )


def resample_batch_to_unified_grid(
    iv_batch: np.ndarray,
    *,
    k_src: np.ndarray | None = None,
    m_src: np.ndarray | None = None,
    tau_src: np.ndarray,
    grid: UnifiedGrid | None = None,
) -> np.ndarray:
    """Resample a batch of surfaces (leading sample/batch dims preserved).

    Supports any leading dims, e.g. ``(N, I, J)`` for LHS batches or
    ``(N_paths, n_steps, I, J)`` for sequential paths.
    """
    arr = np.asarray(iv_batch, dtype=float)
    if arr.ndim < 2:
        raise ValueError("iv_batch must have the two trailing grid axes")
    g = grid if grid is not None else UnifiedGrid.default()
    lead_shape = arr.shape[:-2]
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    out = np.empty((flat.shape[0], g.log_moneyness.size, g.tau.size), dtype=float)
    for n in range(flat.shape[0]):
        out[n] = resample_to_unified_grid(
            flat[n],
            k_src=k_src,
            m_src=m_src,
            tau_src=tau_src,
            grid=g,
        )
    return out.reshape(*lead_shape, g.log_moneyness.size, g.tau.size)


__all__ = [
    "DEFAULT_LOG_MONEYNESS_SPEC",
    "DEFAULT_TAU_SPEC",
    "UNIFIED_IV_GRID_YAML",
    "UnifiedGrid",
    "resample_batch_to_unified_grid",
    "resample_to_unified_grid",
]
