"""Lightweight dataclasses passed between layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketState:
    """Spot / rate / dividend-yield triple at a valuation date."""

    spot: float
    rate: float
    dividend_yield: float = 0.0


@dataclass(frozen=True)
class SurfaceBatch:
    """A batch of IV surfaces together with the axes they live on.

    Attributes:
        params: Parameter matrix of shape ``(B, n_params)``.
        moneyness: 1D moneyness axis.
        tau: 1D maturity axis (years).
        iv: Implied-vol tensor with shape ``(B, n_moneyness, n_tau)``.
    """

    params: np.ndarray
    moneyness: np.ndarray
    tau: np.ndarray
    iv: np.ndarray

    def numpy(self) -> tuple:
        """Return all fields as NumPy arrays."""
        return (self.params, self.moneyness, self.tau, self.iv)
