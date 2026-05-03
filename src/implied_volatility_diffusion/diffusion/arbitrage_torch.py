"""Differentiable no-arbitrage penalties on IV surface."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via :func:`torch.special.erf`."""
    return 0.5 * (1.0 + torch.special.erf(x / math.sqrt(2.0)))


def torch_bs_call(
    spot: torch.Tensor | float,
    strike: torch.Tensor,
    tau: torch.Tensor,
    rate: torch.Tensor | float,
    sigma: torch.Tensor,
    dividend_yield: torch.Tensor | float = 0.0,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Vectorized differentiable Black-Scholes-Merton call price."""
    sig = torch.clamp(sigma, min=eps)
    t = torch.clamp(tau, min=eps)
    sqrt_t = torch.sqrt(t)
    log_sk = torch.log(spot / strike)
    d1 = (log_sk + (rate - dividend_yield + 0.5 * sig * sig) * t) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    return spot * torch.exp(-dividend_yield * t) * _normal_cdf(d1) - strike * torch.exp(-rate * t) * _normal_cdf(d2)


@dataclass(frozen=True)
class ArbitrageWeights:
    """Per-family multipliers; set to 0 to disable a family."""

    calendar: float = 1.0
    butterfly: float = 1.0
    call: float = 1.0


class ArbitragePenalty(nn.Module):
    """Differentiable arbitrage penalties on **unnormalized IV** surfaces.

    Args:
        log_moneyness: ``(n_k,)`` log-moneyness axis of the IV grid.
        tau: ``(n_tau,)`` maturity axis.
        spot: Reference spot used to recover strikes ``K = spot * exp(k)``.
        rate: Risk-free rate.
        dividend_yield: Continuous dividend yield.
        weights: Per-family penalty multipliers.
    """

    def __init__(
        self,
        *,
        log_moneyness: np.ndarray | torch.Tensor,
        tau: np.ndarray | torch.Tensor,
        spot: float = 1.0,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        weights: ArbitrageWeights | None = None,
    ) -> None:
        super().__init__()
        k = torch.as_tensor(np.asarray(log_moneyness), dtype=torch.float32).reshape(-1)
        t = torch.as_tensor(np.asarray(tau), dtype=torch.float32).reshape(-1)
        self.register_buffer("log_moneyness", k)
        self.register_buffer("tau", t)
        self.register_buffer("strikes", float(spot) * torch.exp(k))
        self.spot = float(spot)
        self.rate = float(rate)
        self.dividend_yield = float(dividend_yield)
        self.weights = weights or ArbitrageWeights()

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.log_moneyness.numel()), int(self.tau.numel()))

    @staticmethod
    def _ensure_iv_grid(iv: torch.Tensor) -> torch.Tensor:
        """Squeeze a single-channel surface to ``(B, n_k, n_tau)`` if needed."""
        if iv.dim() == 4:
            if iv.shape[1] != 1:
                raise ValueError(f"arbitrage penalty supports 1-channel surfaces; got C={iv.shape[1]}")
            return iv.squeeze(1)
        if iv.dim() != 3:
            raise ValueError(f"expected (B, n_k, n_tau) or (B, 1, n_k, n_tau); got {tuple(iv.shape)}")
        return iv

    def call_prices(self, iv: torch.Tensor) -> torch.Tensor:
        """Differentiable BS call prices on the ``(B, n_k, n_tau)`` IV grid."""
        iv = self._ensure_iv_grid(iv)
        b = iv.shape[0]
        k = self.strikes.view(1, -1, 1).expand(b, -1, self.tau.numel())
        t = self.tau.view(1, 1, -1).expand(b, self.strikes.numel(), -1)
        return torch_bs_call(
            spot=self.spot,
            strike=k,
            tau=t,
            rate=self.rate,
            sigma=iv,
            dividend_yield=self.dividend_yield,
        )

    def calendar_penalty(self, iv: torch.Tensor) -> torch.Tensor:
        """Per-sample mean of ``relu(-d(sigma^2 tau)/dtau)`` over the grid."""
        iv = self._ensure_iv_grid(iv)
        if iv.shape[-1] < 2:
            return iv.new_zeros(iv.shape[0])
        w = (iv * iv) * self.tau.view(1, 1, -1)
        diff = w[..., 1:] - w[..., :-1]
        return torch.relu(-diff).mean(dim=(-1, -2))

    def butterfly_penalty(self, c: torch.Tensor) -> torch.Tensor:
        """Per-sample mean of ``relu(-d^2 C/dK^2)`` (call convexity in strike)."""
        if c.shape[-2] < 3:
            return c.new_zeros(c.shape[0])
        k = self.strikes
        dk_l = (k[1:-1] - k[:-2]).view(1, -1, 1)
        dk_r = (k[2:] - k[1:-1]).view(1, -1, 1)
        denom = dk_l * dk_r * (dk_l + dk_r) / 2.0
        d2 = (dk_l * c[..., 2:, :] - (dk_l + dk_r) * c[..., 1:-1, :] + dk_r * c[..., :-2, :]) / denom
        return torch.relu(-d2).mean(dim=(-1, -2))

    def call_penalty(self, c: torch.Tensor) -> torch.Tensor:
        """Per-sample mean of ``relu(dC/dK)`` (calls must decrease in strike)."""
        if c.shape[-2] < 2:
            return c.new_zeros(c.shape[0])
        dk = (self.strikes[1:] - self.strikes[:-1]).view(1, -1, 1)
        slope = (c[..., 1:, :] - c[..., :-1, :]) / dk
        return torch.relu(slope).mean(dim=(-1, -2))

    def forward(self, iv: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute every enabled penalty as a per-sample ``(B,)`` tensor.

        ``iv`` must be an **unnormalized** IV surface batch shaped
        ``(B, n_k, n_tau)`` or ``(B, 1, n_k, n_tau)``.
        """
        iv = self._ensure_iv_grid(iv)
        if iv.shape[-2:] != self.grid_shape:
            raise ValueError(f"trailing IV shape {tuple(iv.shape[-2:])} must match grid {self.grid_shape}")
        c = (
            self.call_prices(iv)
            if any(getattr(self.weights, name) > 0 for name in ("butterfly", "call"))
            else iv.new_zeros(iv.shape)
        )

        # Return a dictionary with the penalties
        return {
            "calendar": self.calendar_penalty(iv) * self.weights.calendar,
            "butterfly": self.butterfly_penalty(c) * self.weights.butterfly,
            "call": self.call_penalty(c) * self.weights.call,
        }

    def total(self, iv: torch.Tensor) -> torch.Tensor:
        """Sum of all enabled penalties as a per-sample ``(B,)`` tensor."""
        parts = self.forward(iv)
        return torch.stack(list(parts.values()), dim=0).sum(dim=0)


__all__ = ["ArbitragePenalty", "ArbitrageWeights", "torch_bs_call"]
