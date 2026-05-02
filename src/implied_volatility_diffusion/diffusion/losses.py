"""Diffusion training loss with time-annealed arbitrage penalties."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn

from implied_volatility_diffusion.diffusion.arbitrage_torch import ArbitragePenalty
from implied_volatility_diffusion.diffusion.model import DiffusionModel
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler

ArbitrageSchedule = Literal["alpha_bar", "sqrt_alpha_bar", "linear", "snr", "constant"]


def _arbitrage_weights(scheduler: VPNoiseScheduler, t: torch.Tensor, schedule: ArbitrageSchedule) -> torch.Tensor:
    """Per-sample arbitrage weight, monotonically decreasing in ``t``.

    Schedules (all map ``t -> 1`` near 0 and ``-> 0`` near T):

    * ``alpha_bar``      ``alpha_bar_t``
    * ``sqrt_alpha_bar`` ``sqrt(alpha_bar_t)``
    * ``snr``            ``alpha_bar_t / (1 - alpha_bar_t + eps)`` then clipped to [0, 1]
    * ``linear``         ``1 - t / (T - 1)``
    * ``constant``       ``1.0`` (no annealing)
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
        T_max = max(scheduler.timesteps - 1, 1)
        return torch.clamp(1.0 - t.float() / T_max, min=0.0)
    if schedule == "constant":
        return torch.ones_like(alpha_bar)
    raise ValueError(f"unknown arbitrage schedule: {schedule}")


@dataclass
class DiffusionLossConfig:
    """Knobs for :class:`DiffusionLoss`."""

    arbitrage_lambda: float = 0.1
    arbitrage_schedule: ArbitrageSchedule = "alpha_bar"
    component_names: tuple[str, ...] = field(default_factory=lambda: ("calendar", "butterfly", "call"))

    predicted_z0_clip: tuple[float, float] = (-4.0, 4.0)
    # reference : https://arxiv.org/pdf/2303.09556.pdf
    # Training via Min-SNR Weighting Strategy")
    # gamma = 5 is recommended by the paper
    min_snr_gamma: float = 5.0


class DiffusionLoss(nn.Module):
    """Compose ``epsilon`` MSE with a time-annealed arbitrage penalty."""

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
        """Uniform integer ``t`` in ``[0, timesteps)``."""
        return torch.randint(0, scheduler.timesteps, (batch_size,), device=device, dtype=torch.long)

    def __call__(
        self,
        model: DiffusionModel,
        iv0: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train-step loss on an **unnormalized** IV batch.

        Returns a dict with ``loss``, ``loss_eps``, ``loss_arb`` (when an
        arbitrage penalty is attached), and per-family arbitrage components.
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

        if self.arbitrage_penalty is not None and self.config.arbitrage_lambda > 0.0:
            x0_z = self._recover_x0_z(model, z_t, pred, t)
            clip = self.config.predicted_z0_clip
            if clip is not None:
                x0_z = torch.clamp(x0_z, clip[0], clip[1])
            iv_pred = torch.nan_to_num(model.denormalize(x0_z), nan=0.0, posinf=0.0, neginf=0.0)

            parts = self.arbitrage_penalty(iv_pred)
            w = _arbitrage_weights(scheduler, t, self.config.arbitrage_schedule)
            arb_per_sample = torch.zeros(iv0.shape[0], device=device)
            for name, value in parts.items():
                weighted = value * w
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
        """Recover x0 in z-space from a cached backbone output.

        Reuses the already-computed ``pred`` instead of calling
        :meth:`DiffusionModel.predict_x0_z`, which would invoke the backbone a
        second time and double the per-step cost.
        """
        if model.prediction_type == "x0":
            return pred
        alpha_bar = model.scheduler.alpha_bar_at(t).view(t.shape[0], *([1] * (z_t.dim() - 1)))
        sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-8))
        sqrt_one_minus_ab = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
        return (z_t - sqrt_one_minus_ab * pred) / sqrt_ab


__all__ = [
    "ArbitrageSchedule",
    "DiffusionLoss",
    "DiffusionLossConfig",
]
