"""Scenario generators for joint asset return and IV-surface dynamics."""

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from implied_volatility_diffusion.scenarios.types import JointHistoricalState


@runtime_checkable
class JointScenarioGenerator(Protocol):
    """Produce ``n`` joint scenarios of log return and IV surface."""

    def generate(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(log_returns, iv_surfaces)`` with shapes ``(n,)`` and ``(n, M, T)``."""
        ...


@dataclass(frozen=True)
class FilteredHistoricalSettings:
    """FHS controls: EWMA volatility filter and optional IV level anchoring."""

    ewma_lambda: float = 0.94
    min_vol: float = 1e-8
    anchor_to_base_iv: bool = False
    base_iv: np.ndarray | None = None


class FilteredHistoricalSimulation:
    """Filtered Historical Simulation on joint (log return, IV surface) paths.

    Standardized historical return innovations are resampled and scaled by the
    current EWMA conditional volatility. IV surfaces are bootstrapped from
    history; when ``anchor_to_base_iv`` is set, surfaces are shifted so their
    mean level matches ``base_iv``.
    """

    def __init__(
        self,
        history: JointHistoricalState,
        *,
        settings: FilteredHistoricalSettings | None = None,
    ) -> None:
        self.history = history
        self.settings = settings or FilteredHistoricalSettings()
        h = history
        r = np.asarray(h.log_returns, dtype=float).reshape(-1)
        self._iv = np.asarray(h.iv_surfaces, dtype=float)
        cfg = self.settings
        self._scales = self._ewma_scales(r, cfg.ewma_lambda, cfg.min_vol)
        self._standardized = r / self._scales
        if self.settings.anchor_to_base_iv:
            if self.settings.base_iv is None:
                raise ValueError("base_iv is required when anchor_to_base_iv=True")
            base = np.asarray(self.settings.base_iv, dtype=float)
            if base.shape != self._iv.shape[1:]:
                raise ValueError(f"base_iv shape {base.shape} must match surface {self._iv.shape[1:]}")

    @staticmethod
    def _ewma_scales(log_returns: np.ndarray, lam: float, min_vol: float) -> np.ndarray:
        r = np.asarray(log_returns, dtype=float).reshape(-1)
        var = np.empty(r.size, dtype=float)
        var[0] = max(r[0] ** 2, min_vol**2)
        for t in range(1, r.size):
            var[t] = lam * var[t - 1] + (1.0 - lam) * r[t] ** 2
        return np.maximum(np.sqrt(var), min_vol)

    def current_return_scale(self) -> float:
        """EWMA volatility at the end of the historical window."""
        return float(self._scales[-1])

    def generate(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            raise ValueError("n must be positive")
        gen = rng or np.random.default_rng()
        h = len(self._standardized)
        idx = gen.integers(0, h, size=n)
        scale_now = self.current_return_scale()
        log_returns = scale_now * self._standardized[idx]
        iv = self._iv[idx].copy()
        if self.settings.anchor_to_base_iv:
            base = np.asarray(self.settings.base_iv, dtype=float)
            shift = base - np.mean(self._iv, axis=0, keepdims=True)
            iv = iv + shift
        return log_returns.astype(float), iv.astype(float)


class CallableJointScenarioGenerator:
    """Wrap any callable ``(n, rng) -> (log_returns, iv_surfaces)`` (e.g. VolGAN / diffusion)."""

    def __init__(
        self,
        fn: Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]],
        *,
        seed: int | None = None,
    ) -> None:
        self._fn = fn
        self._seed = seed

    def generate(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        gen = rng or np.random.default_rng(self._seed)
        log_returns, iv = self._fn(n, gen)
        log_returns = np.asarray(log_returns, dtype=float).reshape(-1)
        iv_arr = np.asarray(iv, dtype=float)
        if log_returns.size != n:
            raise ValueError(f"expected {n} log returns; got {log_returns.size}")
        if iv_arr.shape[0] != n:
            raise ValueError(f"expected {n} surfaces; got leading dim {iv_arr.shape[0]}")
        return log_returns, iv_arr
