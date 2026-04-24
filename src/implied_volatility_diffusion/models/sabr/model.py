"""SABR :class:`VolModel` adapter backed by QuantLib."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import QuantLib as ql

from implied_volatility_diffusion.models.base import register_model
from implied_volatility_diffusion.pricing.black_scholes import bs_call_price

SABR_PARAM_ORDER: tuple[str, ...] = ("alpha", "rho", "nu")


@dataclass(frozen=True)
class SabrSettings:
    """SABR model constants read from config (beta is fixed per surface)."""

    beta: float = 0.5

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SabrSettings":
        section = cfg.get("sabr", {}) or {}
        return cls(beta=float(section.get("beta", 0.5)))


class SabrModel:
    """SABR model wired to the :class:`VolModel` protocol via QuantLib."""

    param_order: tuple[str, ...] = SABR_PARAM_ORDER

    def __init__(self, *, settings: SabrSettings | None = None) -> None:
        self.settings = settings or SabrSettings()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SabrModel":
        return cls(settings=SabrSettings.from_config(cfg))

    @staticmethod
    def _as_batch(params: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        if p.ndim == 1:
            return p.reshape(1, -1)
        if p.ndim == 2:
            return p
        raise ValueError(f"params must be 1D or 2D, got shape {tuple(p.shape)}")

    def implied_vol_surface(
        self,
        params: np.ndarray,
        *,
        spot: np.ndarray | float,
        moneyness: np.ndarray,
        tau: np.ndarray,
        rate: np.ndarray | float = 0.0,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        """Return ``(B, M, T)`` Black vols via QuantLib SABR."""
        params_b = self._as_batch(params)
        m = np.asarray(moneyness, dtype=float).reshape(-1)
        t = np.asarray(tau, dtype=float).reshape(-1)
        spot_f = float(np.asarray(spot, dtype=float).item())
        rate_f = float(np.asarray(rate, dtype=float).item())
        q = float(dividend_yield)
        fwd = spot_f * np.exp((rate_f - q) * t)
        strikes = m * spot_f
        out = np.empty((params_b.shape[0], m.size, t.size), dtype=float)

        pos_t_idx = np.flatnonzero(t > 0.0)
        nonpos_t_idx = np.flatnonzero(t <= 0.0)
        out[:, :, nonpos_t_idx] = np.nan

        for bi in range(params_b.shape[0]):
            alpha, rho, nu = (float(x) for x in params_b[bi])
            for ti in pos_t_idx:
                tau_i = float(t[ti])
                fwd_i = float(fwd[ti])
                for mi, strike_i in enumerate(strikes):
                    out[bi, mi, ti] = float(
                        ql.sabrVolatility(
                            float(strike_i),
                            fwd_i,
                            tau_i,
                            alpha,
                            self.settings.beta,
                            nu,
                            rho,
                        )
                    )
        return out

    def price_calls(
        self,
        params: np.ndarray,
        *,
        spot: np.ndarray | float,
        moneyness: np.ndarray,
        tau: np.ndarray,
        rate: np.ndarray | float = 0.0,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        """Black-Scholes call prices from QuantLib SABR IV. Used by arbitrage checks."""
        iv = self.implied_vol_surface(
            params,
            spot=spot,
            moneyness=moneyness,
            tau=tau,
            rate=rate,
            dividend_yield=dividend_yield,
        )
        m = np.asarray(moneyness, dtype=float).reshape(-1)
        t = np.asarray(tau, dtype=float).reshape(-1)
        spot_f = float(np.asarray(spot, dtype=float).item())
        strikes = m.reshape(1, -1, 1) * spot_f
        tau_b = t.reshape(1, 1, -1)
        return bs_call_price(spot_f, strikes, tau_b, rate, iv, dividend_yield)


def _factory(cfg: Mapping[str, Any]) -> SabrModel:
    return SabrModel.from_config(cfg)


register_model("sabr", _factory)
