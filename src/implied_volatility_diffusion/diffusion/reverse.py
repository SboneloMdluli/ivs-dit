"""Reverse diffusion sampling that returns **unnormalized IV surfaces**."""

from typing import Callable

import torch
import torch.nn as nn

from implied_volatility_diffusion.diffusion.model import DiffusionModel


def _broadcast(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape ``(B,)`` -> ``(B, 1, 1, ...)`` to broadcast against ``ref``."""
    return value.view(value.shape[0], *([1] * (ref.dim() - 1)))


class ReverseDiffusion(nn.Module):
    """Decoupled DDPM/DDIM sampler operating on a :class:`DiffusionModel`.

    All public sampling methods return **unnormalized IV** surfaces; intermediate
    states ``z_t`` and the predicted clean ``x0`` are exposed in z-space via
    the optional callbacks for diagnostics.
    """

    def __init__(self, model: DiffusionModel) -> None:
        super().__init__()
        if not isinstance(model, DiffusionModel):
            raise TypeError(
                f"ReverseDiffusion expects a DiffusionModel that owns the normalizer; got {type(model).__name__}"
            )
        self.model = model

    @property
    def scheduler(self):  # noqa: D401 - simple alias
        """Forward through to the model's scheduler."""
        return self.model.scheduler

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.model.grid_shape

    def _eps_and_x0_z(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        clip_z: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone and return ``(eps, x0_z)`` for either parameterization."""
        pred = self.model.predict_noise(z_t, t, cond)
        alpha_bar = _broadcast(self.scheduler.alpha_bar_at(t), z_t)
        if self.model.prediction_type == "epsilon":
            eps = pred
            x0_z = (z_t - torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0)) * eps) / torch.sqrt(
                torch.clamp(alpha_bar, min=1e-8)
            )
        else:
            x0_z = pred
            eps = (z_t - torch.sqrt(torch.clamp(alpha_bar, min=1e-8)) * x0_z) / torch.sqrt(
                torch.clamp(1.0 - alpha_bar, min=1e-8)
            )
        if clip_z is not None:
            x0_z = torch.clamp(x0_z, clip_z[0], clip_z[1])
            eps = (z_t - torch.sqrt(torch.clamp(alpha_bar, min=1e-8)) * x0_z) / torch.sqrt(
                torch.clamp(1.0 - alpha_bar, min=1e-8)
            )
        return eps, x0_z

    @torch.no_grad()
    def p_sample(
        self,
        z_t: torch.Tensor,
        t_index: int,
        *,
        cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        clip_z: tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One DDPM ancestral step ``z_t -> z_{t-1}``. Returns ``(z_prev, x0_z)``."""
        b = z_t.shape[0]
        device = z_t.device
        t = torch.full((b,), int(t_index), device=device, dtype=torch.long)
        eps, x0_z = self._eps_and_x0_z(z_t, t, cond, clip_z)

        beta = self.scheduler.extract("betas", t, z_t.shape)
        alpha = self.scheduler.extract("alphas", t, z_t.shape)
        alpha_bar = _broadcast(self.scheduler.alpha_bar_at(t), z_t)
        post_var = self.scheduler.extract("posterior_variance", t, z_t.shape)

        mean = (1.0 / torch.sqrt(alpha)) * (z_t - beta / torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8)) * eps)
        if t_index == 0:
            return mean, x0_z
        noise = torch.randn(z_t.shape, device=device, dtype=z_t.dtype, generator=generator)
        return mean + torch.sqrt(post_var) * noise, x0_z

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...] | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
        cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        clip_z: tuple[float, float] | None = None,
        callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Run the full DDPM ancestral chain. Returns **unnormalized IV** of shape ``shape``.

        If ``shape`` is omitted, a default of ``(batch_size, 1, *grid_shape)``
        is used (with ``batch_size=1`` if also omitted).
        """
        shape = self._resolve_shape(shape, batch_size)
        device = device if device is not None else next(self.model.parameters()).device
        z_t = torch.randn(shape, device=device, generator=generator)
        for t_index in reversed(range(self.scheduler.timesteps)):
            z_t, x0_z = self.p_sample(
                z_t,
                t_index,
                cond=cond,
                generator=generator,
                clip_z=clip_z,
            )
            if callback is not None:
                callback(t_index, z_t, x0_z)
        return self.model.denormalize(z_t)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...] | None = None,
        *,
        batch_size: int | None = None,
        num_steps: int = 50,
        eta: float = 0.0,
        device: torch.device | None = None,
        cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        clip_z: tuple[float, float] | None = None,
        callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """DDIM (Song et al. 2020) sampler with adjustable stochasticity ``eta``.

        ``eta=0`` is fully deterministic; ``eta=1`` matches DDPM ancestral noise.
        Returns **unnormalized IV**.
        """
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        shape = self._resolve_shape(shape, batch_size)
        device = device if device is not None else next(self.model.parameters()).device

        T = self.scheduler.timesteps
        step_indices = torch.linspace(0, T - 1, num_steps + 1, device=device).round().long()
        step_indices = torch.unique(step_indices, sorted=True).flip(0)

        z_t = torch.randn(shape, device=device, generator=generator)
        for i in range(step_indices.shape[0] - 1):
            t_idx = int(step_indices[i].item())
            t_prev_idx = int(step_indices[i + 1].item())
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_idx, device=device, dtype=torch.long)
            eps, x0_z = self._eps_and_x0_z(z_t, t, cond, clip_z)

            alpha_bar_t = _broadcast(self.scheduler.alpha_bar_at(t), z_t)
            alpha_bar_prev = _broadcast(self.scheduler.alpha_bar_at(t_prev), z_t)

            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-8), min=0.0)
                * torch.clamp(1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-8), min=0.0)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps
            noise = torch.randn(z_t.shape, device=device, dtype=z_t.dtype, generator=generator) if eta > 0 else 0.0
            z_t = torch.sqrt(alpha_bar_prev) * x0_z + dir_xt + sigma * noise
            if callback is not None:
                callback(t_idx, z_t, x0_z)
        return self.model.denormalize(z_t)

    def _resolve_shape(self, shape: tuple[int, ...] | None, batch_size: int | None) -> tuple[int, ...]:
        if shape is not None:
            if shape[-2:] != self.grid_shape:
                raise ValueError(f"sample shape trailing axes {shape[-2:]} must match grid {self.grid_shape}")
            return shape
        b = int(batch_size) if batch_size is not None else 1
        return (b, self.model.in_channels, *self.grid_shape)


__all__ = ["ReverseDiffusion"]
