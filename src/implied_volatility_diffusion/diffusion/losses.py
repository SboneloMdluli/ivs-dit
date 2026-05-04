"""Diffusion training loss with optional smoothness and time-annealed arbitrage penalties."""

from dataclasses import dataclass, field
from typing import Callable, Literal, cast

import torch
import torch.nn as nn

from implied_volatility_diffusion.diffusion.arbitrage_torch import ArbitragePenalty
from implied_volatility_diffusion.diffusion.model import DiffusionModel
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler

ArbitrageSchedule = Literal["alpha_bar", "sqrt_alpha_bar", "linear", "snr", "constant"]
SmoothnessSpace = Literal["iv", "z"]


def _arbitrage_weights(scheduler: VPNoiseScheduler, t: torch.Tensor, schedule: ArbitrageSchedule) -> torch.Tensor:
    """Per-sample weights ~1 near t≈0, ~0 near T (monotone in t).

    Methods: alpha_bar; sqrt(alpha_bar); SNR clipped to [0,1]; linear 1−t/(T−1); constant 1.
    """
    alpha_bar = scheduler.alpha_bar_at(t)
    if schedule == "alpha_bar":
        return alpha_bar
    if schedule == "sqrt_alpha_bar":
        return torch.sqrt(alpha_bar)
    if schedule == "snr":
        snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-8)
        return torch.clamp(snr, max=1.0)
    if schedule == "linear":
        t_max = max(scheduler.timesteps - 1, 1)
        return torch.clamp(1.0 - t.float() / t_max, min=0.0)
    if schedule == "constant":
        return torch.ones_like(alpha_bar)


def _dirichlet_energy_index_mean_per_sample(iv: torch.Tensor) -> torch.Tensor:
    """Per batch row: mean squared forward differences along m and τ with unit spacing, summed."""
    device, dtype = iv.device, iv.dtype
    out = torch.zeros(iv.shape[0], device=device, dtype=dtype)
    h, w = iv.shape[-2], iv.shape[-1]
    if h >= 2:
        dg = iv[..., 1:, :] - iv[..., :-1, :]
        out = out + (dg * dg).flatten(1).mean(dim=1)
    if w >= 2:
        dg = iv[..., :, 1:] - iv[..., :, :-1]
        out = out + (dg * dg).flatten(1).mean(dim=1)
    return out


def _iv_dirichlet_smoothness_per_sample(iv: torch.Tensor) -> torch.Tensor:
    """Per-batch Dirichlet energy on the predicted surface using unit grid spacing (index mesh).

    Time-annealing (e.g. SNR or linear in ``t``) is applied outside via :func:`_arbitrage_weights`.
    """
    return _dirichlet_energy_index_mean_per_sample(iv)


@dataclass
class DiffusionLossConfig:
    """Hyperparameters for :class:`DiffusionLoss`."""

    arbitrage_lambda: float = 0.5
    # schedules as in Zhou et al. (arXiv:2511.07571)
    arbitrage_schedule: ArbitrageSchedule = "snr"
    component_names: tuple[str, ...] = field(default_factory=lambda: ("calendar", "butterfly", "call"))

    predicted_z0_clip: tuple[float, float] | None = (-4.0, 4.0)
    # Min-SNR weighting (Hang et al., arXiv:2303.09556); γ=5 default
    min_snr_gamma: float | None = 5.0

    smoothness_lambda: float = 1e-10
    # Same schedule vocabulary as arbitrage (`snr`, `linear`, ...); uses `_arbitrage_weights`.
    smoothness_schedule: ArbitrageSchedule = "snr"
    smoothness_space: SmoothnessSpace = "iv"


class DiffusionLoss(nn.Module):
    """Epsilon MSE plus optional Dirichlet smoothness and arbitrage penalties."""

    def __init__(
        self,
        arbitrage_penalty: ArbitragePenalty | Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
        *,
        config: DiffusionLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.arbitrage_penalty = arbitrage_penalty
        self.config = config or DiffusionLossConfig()

    def sample_timesteps(
        self, batch_size: int, scheduler: VPNoiseScheduler, *, device: torch.device | None = None
    ) -> torch.Tensor:
        """Uniform integer t in ``[0, timesteps)``."""
        return torch.randint(0, scheduler.timesteps, (batch_size,), device=device, dtype=torch.long)

    def forward(
        self,
        model: DiffusionModel,
        iv0: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training loss on **unnormalized** IV batch.

        Dict includes ``loss``, ``loss_eps``, and when enabled ``loss_smooth`` / ``loss_arb``
        plus per-family ``arb_*`` keys.
        """
        scheduler = model.scheduler
        device = iv0.device
        if t is None:
            t = self.sample_timesteps(iv0.shape[0], scheduler, device=device)
        if t.dim() == 0:
            t = t.expand(iv0.shape[0])

        z_t, z0, eps = model.add_noise(iv0, t, noise=noise)

        pred = model.predict_noise(z_t, t, cond)

        target = eps if model.prediction_type == "epsilon" else z0
        per_sample_mse = ((pred - target) ** 2).flatten(1).mean(dim=1)

        if self.config.min_snr_gamma is not None and self.config.min_snr_gamma > 0:
            alpha_bar = scheduler.alpha_bar_at(t)
            snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-8)
            min_snr = torch.clamp(snr, max=float(self.config.min_snr_gamma))

            if model.prediction_type == "epsilon":
                w_loss = min_snr / torch.clamp(snr, min=1e-8)
            else:
                w_loss = min_snr
            loss_eps = (w_loss * per_sample_mse).mean()
        else:
            loss_eps = per_sample_mse.mean()

        out: dict[str, torch.Tensor] = {"loss_eps": loss_eps}
        loss_total = loss_eps

        arb_on = self.arbitrage_penalty is not None and self.config.arbitrage_lambda > 0.0
        smooth_on = self.config.smoothness_lambda > 0.0

        if arb_on or smooth_on:
            x0_z = self._recover_x0_z(model, z_t, pred, t)
            clip = self.config.predicted_z0_clip
            if clip is not None:
                x0_z = torch.clamp(x0_z, clip[0], clip[1])
            need_iv = arb_on or (smooth_on and self.config.smoothness_space == "iv")
            if need_iv:
                iv_pred = torch.nan_to_num(model.denormalize(x0_z), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                iv_pred = None

            if smooth_on:
                w_s = _arbitrage_weights(scheduler, t, self.config.smoothness_schedule)
                surf = x0_z if self.config.smoothness_space == "z" else cast(torch.Tensor, iv_pred)
                smooth_per = _iv_dirichlet_smoothness_per_sample(surf)
                loss_smooth = (w_s * smooth_per).mean()
                out["loss_smooth"] = loss_smooth
                out["smooth_weight_mean"] = w_s.mean().detach()
                loss_total = loss_total + self.config.smoothness_lambda * loss_smooth

            if arb_on:
                parts = self.arbitrage_penalty(cast(torch.Tensor, iv_pred))
                w = _arbitrage_weights(scheduler, t, self.config.arbitrage_schedule)
                arb_per_sample = torch.zeros(iv0.shape[0], device=device)
                for name, value in parts.items():
                    weighted = w * value
                    out[f"arb_{name}"] = weighted.mean()
                    arb_per_sample = arb_per_sample + weighted
                loss_arb = arb_per_sample.mean()
                out["loss_arb"] = loss_arb
                out["arb_weight_mean"] = w.mean().detach()
                loss_total = loss_total + self.config.arbitrage_lambda * loss_arb

        out["loss"] = loss_total
        return out

    @staticmethod
    def _recover_x0_z(
        model: DiffusionModel,
        z_t: torch.Tensor,
        pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x0 in z-space from ``pred`` (avoids an extra backbone pass)."""
        if model.prediction_type == "x0":
            return pred
        alpha_bar = model.scheduler.alpha_bar_at(t).view(t.shape[0], *([1] * (z_t.dim() - 1)))
        sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-8))
        sqrt_one_minus_ab = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
        return (z_t - sqrt_one_minus_ab * pred) / sqrt_ab


_volgan_iv_dirichlet_smoothness_per_sample = _iv_dirichlet_smoothness_per_sample

__all__ = [
    "ArbitrageSchedule",
    "DiffusionLoss",
    "DiffusionLossConfig",
    "SmoothnessSpace",
]
