"""Datatypes for joint return / IV-surface scenario batches."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointHistoricalState:
    """Historical joint path used by filtered historical simulation (FHS)."""

    log_returns: np.ndarray
    iv_surfaces: np.ndarray

    def __post_init__(self) -> None:
        r = np.asarray(self.log_returns, dtype=float).reshape(-1)
        iv = np.asarray(self.iv_surfaces, dtype=float)
        if iv.ndim != 3:
            raise ValueError(f"iv_surfaces must be (H, M, T); got shape {iv.shape}")
        if r.size != iv.shape[0]:
            raise ValueError(f"log_returns length {r.size} must match iv history {iv.shape[0]}")


@dataclass(frozen=True)
class PenaltyWeightingResult:
    """Arbitrage penalties and VolGAN-style weights for a scenario batch."""

    penalties: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.penalties, dtype=float).reshape(-1)
        w = np.asarray(self.weights, dtype=float).reshape(-1)
        if p.shape != w.shape:
            raise ValueError(f"penalties shape {p.shape} must match weights {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not np.isclose(w.sum(), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(f"weights must sum to 1; got {w.sum()}")


@dataclass(frozen=True)
class JointScenarioBatch:
    """N joint scenarios: log return and IV surface per path, with penalties and weights."""

    log_returns: np.ndarray
    iv_surfaces: np.ndarray
    penalties: np.ndarray
    weights: np.ndarray

    @property
    def n_scenarios(self) -> int:
        return int(self.iv_surfaces.shape[0])

    def __post_init__(self) -> None:
        r = np.asarray(self.log_returns, dtype=float).reshape(-1)
        iv = np.asarray(self.iv_surfaces, dtype=float)
        p = np.asarray(self.penalties, dtype=float).reshape(-1)
        w = np.asarray(self.weights, dtype=float).reshape(-1)
        n = r.size
        if iv.shape[0] != n:
            raise ValueError(f"iv_surfaces leading dim {iv.shape[0]} must match n={n}")
        if p.shape != (n,) or w.shape != (n,):
            raise ValueError("penalties and weights must have shape (n_scenarios,)")
