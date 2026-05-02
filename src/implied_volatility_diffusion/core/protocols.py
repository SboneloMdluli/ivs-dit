"""Protocols that decouple pricing models from surface assembly."""

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelCallPricer(Protocol):
    """Scalar (strike, tau) -> discounted call price. Kept for legacy hooks."""

    def __call__(self, strike: float, tau: float) -> float: ...


@runtime_checkable
class ImpliedVolInverter(Protocol):
    """Map model call price -> Black-Scholes implied volatility (scalar)."""

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
    ) -> float: ...


@runtime_checkable
class VolModel(Protocol):
    """Stochastic / local-vol model that can price a batch of European calls."""

    #: Order of the parameter columns the model expects in ``params``.
    param_order: tuple[str, ...]

    def price_calls(
        self,
        params: np.ndarray,
        *,
        spot: np.ndarray | float,
        moneyness: np.ndarray,
        tau: np.ndarray,
        rate: np.ndarray | float,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        """Discounted European call prices with shape ``(B, n_moneyness, n_tau)``."""
        ...

    def implied_vol_surface(
        self,
        params: np.ndarray,
        *,
        spot: np.ndarray | float,
        moneyness: np.ndarray,
        tau: np.ndarray,
        rate: np.ndarray | float,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        """Black-Scholes implied volatility with the same ``(B, M, T)`` shape."""
        ...


class RangesConfig(Protocol):
    """Config section describing parameter box ranges used by LHS sampling."""

    def __getitem__(self, key: str) -> Any: ...


# Small alias used by type hints below.
ArrayLike = np.ndarray
ConfigMapping = Mapping[str, Any]
