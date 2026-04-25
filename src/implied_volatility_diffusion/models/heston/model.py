"""Heston :class:`VolModel` adapter (NumPy implementation)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from implied_volatility_diffusion.models.base import register_model
from implied_volatility_diffusion.models.heston.cos import heston_call_cos
from implied_volatility_diffusion.pricing.implied_vol import implied_vol_from_prices

HESTON_PARAM_ORDER: tuple[str, ...] = ("v0", "rho", "sigma", "theta", "kappa", "r")


@dataclass(frozen=True)
class HestonCosSettings:
    """COS-pricer hyperparameters pulled from the config."""

    n_terms_base: int = 1024
    truncation_L: float = 14.0
    short_tau_ref: float = 0.25
    n_terms_max: int = 4096

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "HestonCosSettings":
        section = cfg.get("heston_cos_pricer") or cfg.get("cos") or {}
        return cls(
            n_terms_base=int(section.get("n_terms", 1024)),
            truncation_L=float(section.get("truncation_L", 14.0)),
            short_tau_ref=float(section.get("short_tau_tau_ref", 0.25)),
            n_terms_max=int(section.get("n_terms_max", 4096)),
        )


@dataclass(frozen=True)
class ImpliedVolSettings:
    """Newton / Brent tuning for Black-Scholes IV inversion."""

    sigma_lo: float = 1e-4
    sigma_hi: float = 10.0
    tau_extrapolate_below: float = float("nan")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ImpliedVolSettings":
        section = cfg.get("implied_vol", {}) or {}
        return cls(
            sigma_lo=float(section.get("sigma_lo", 1e-4)),
            sigma_hi=float(section.get("sigma_hi", 10.0)),
            tau_extrapolate_below=float(section.get("tau_extrapolate_below", float("nan"))),
        )


class HestonModel:
    """Heston COS pricer packaged as a :class:`VolModel`.

    Parameters follow :data:`HESTON_PARAM_ORDER`; :meth:`price_calls` accepts
    either a ``(n_params,)`` vector or a ``(B, n_params)`` batch.
    """

    param_order: tuple[str, ...] = HESTON_PARAM_ORDER

    def __init__(
        self,
        *,
        cos: HestonCosSettings | None = None,
        iv: ImpliedVolSettings | None = None,
    ) -> None:
        self.cos = cos or HestonCosSettings()
        self.iv = iv or ImpliedVolSettings()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "HestonModel":
        return cls(cos=HestonCosSettings.from_config(cfg), iv=ImpliedVolSettings.from_config(cfg))

    @staticmethod
    def _as_batch(params: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        if p.ndim == 1:
            return p.reshape(1, -1)
        if p.ndim == 2:
            return p
        raise ValueError(f"params must be 1D or 2D, got shape {tuple(p.shape)}")

    def _n_terms_for(self, tau: float) -> int:
        tau_safe = max(float(tau), 1e-12)
        short_scale = math.sqrt(self.cos.short_tau_ref / tau_safe)
        long_scale = math.sqrt(tau_safe / self.cos.short_tau_ref)
        scale = max(1.0, short_scale, long_scale)
        return int(min(self.cos.n_terms_max, round(self.cos.n_terms_base * scale)))

    def price_calls(
        self,
        params: np.ndarray,
        *,
        spot: np.ndarray | float,
        moneyness: np.ndarray,
        tau: np.ndarray,
        rate: np.ndarray | float | None = None,
        dividend_yield: float = 0.0,
        inst_var: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(B, n_moneyness, n_tau)`` discounted call prices.

        ``rate`` is ignored (per-row ``r`` is taken from the parameter matrix),
        but kept in the signature to satisfy :class:`VolModel`.
        ``inst_var`` overrides the ``v0`` column per row (used by the
        sequential-IVS generator to plug in an instantaneous variance along a
        simulated path).
        """
        del rate
        params_b = self._as_batch(params)
        m = np.asarray(moneyness, dtype=float).reshape(-1)
        t = np.asarray(tau, dtype=float).reshape(-1)
        n_batch = int(params_b.shape[0])
        n_m = int(m.size)
        n_t = int(t.size)
        spot_f = float(np.asarray(spot, dtype=float).item())
        q = float(dividend_yield)

        iv_override = None
        if inst_var is not None:
            iv_override = np.asarray(inst_var, dtype=float).reshape(-1)
            if iv_override.size != n_batch:
                raise ValueError("inst_var must have one entry per row of params")

        out = np.empty((n_batch, n_m, n_t), dtype=float)
        strikes = m * spot_f

        for bi in range(n_batch):
            v0_b, rho_b, sig_b, theta_b, kappa_b, r_b = (float(x.item()) for x in params_b[bi])
            v_price = v0_b if iv_override is None else float(iv_override[bi].item())
            for ti in range(n_t):
                tau_val = float(t[ti])
                if tau_val <= 0.0:
                    out[bi, :, ti] = np.maximum(spot_f - strikes, 0.0)
                    continue
                n_terms = self._n_terms_for(tau_val)
                out[bi, :, ti] = np.array(
                    [
                        heston_call_cos(
                            spot_f,
                            float(kv),
                            tau_val,
                            r_b,
                            kappa_b,
                            theta_b,
                            sig_b,
                            rho_b,
                            v_price,
                            q,
                            n_terms,
                            self.cos.truncation_L,
                        )
                        for kv in strikes
                    ],
                    dtype=float,
                )
        return out

    def implied_vol_surface(
        self,
        params: np.ndarray,
        *,
        spot: np.ndarray | float,
        moneyness: np.ndarray,
        tau: np.ndarray,
        rate: np.ndarray | float | None = None,
        dividend_yield: float = 0.0,
        inst_var: np.ndarray | None = None,
    ) -> np.ndarray:
        """Heston call prices inverted to Black-Scholes IV, shape ``(B, M, T)``."""
        params_b = self._as_batch(params)
        m = np.asarray(moneyness, dtype=float).reshape(-1)
        t = np.asarray(tau, dtype=float).reshape(-1)
        spot_f = float(np.asarray(spot, dtype=float).item())

        prices = self.price_calls(
            params_b,
            spot=spot_f,
            moneyness=m,
            tau=t,
            rate=rate,
            dividend_yield=dividend_yield,
            inst_var=inst_var,
        )

        iv_surfaces = np.empty_like(prices)
        strikes = m * spot_f  # (M,)
        for bi in range(int(params_b.shape[0])):
            rate_bi = float(params_b[bi, HESTON_PARAM_ORDER.index("r")])
            # Price-to-IV inversion per row (prices_bi shape (M, T)).
            iv_bi = implied_vol_from_prices(
                prices[bi],
                spot=spot_f,
                strike=strikes.reshape(-1, 1),
                tau=t.reshape(1, -1),
                rate=rate_bi,
                dividend_yield=dividend_yield,
                sigma_lo=self.iv.sigma_lo,
                sigma_hi=self.iv.sigma_hi,
            )
            iv_surfaces[bi] = iv_bi

        thr = float(self.iv.tau_extrapolate_below)
        if math.isfinite(thr) and thr > 0.0 and t.size > 0:
            j_ref = int(np.searchsorted(t, thr, side="left"))
            if 0 <= j_ref < t.size:
                mask = t + 1e-12 < thr
                if bool(np.any(mask)):
                    ref_col = iv_surfaces[..., j_ref : j_ref + 1]
                    iv_surfaces[..., mask] = ref_col
        return iv_surfaces


def _factory(cfg: Mapping[str, Any]) -> HestonModel:
    return HestonModel.from_config(cfg)


register_model("heston", _factory)
