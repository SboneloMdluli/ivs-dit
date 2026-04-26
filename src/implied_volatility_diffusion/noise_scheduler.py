"""Closed-form noising scheduler for the forward diffusion process.

This scheduler follows a variance-preserving (VP/OU) process with constant
beta. Its marginals are available in closed form:

    x_t = exp(-0.5 * beta * t) * x_0 + sqrt(1 - exp(-beta * t)) * eps,
    eps ~ N(0, I).

As t -> inf, x_t converges to N(0, I), so variance remains bounded.

reference: https://huggingface.co/blog/annotated-diffusion
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class VPNoiseScheduler:
    """Variance-preserving forward noising process with closed-form marginals."""

    beta: float = 1.0

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        """Return (alpha_t, sigma_t) for time t >= 0."""
        if t < 0.0:
            raise ValueError("t must be non-negative")
        alpha = float(np.exp(-0.5 * self.beta * t))
        sigma = float(np.sqrt(max(0.0, 1.0 - np.exp(-self.beta * t))))
        return alpha, sigma

    def add_noise(self, x0: np.ndarray, t: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample x_t conditioned on x_0."""
        alpha, sigma = self.alpha_sigma(t)
        z = (rng or np.random.default_rng()).standard_normal(size=np.shape(x0))
        return alpha * np.asarray(x0, dtype=float) + sigma * z
    
    
class VPNoiseScheduler(nn.Module):
    """Variance Preserving (VP) diffusion scheduler with:
    - linear beta schedule
    - cosine schedule (DDPM)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
    ):
        super().__init__()

        self.timesteps = timesteps
        self.beta_schedule = beta_schedule

        betas = self._build_betas(beta_schedule, timesteps, beta_start, beta_end, cosine_s)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bar", torch.cumprod(self.alphas, dim=0))

    def _build_betas(self, schedule, T, beta_start, beta_end, cosine_s):
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T)

        elif schedule == "cosine":
            steps = T + 1
            t = torch.linspace(0, T, steps)

            alphas_cumprod = torch.cos(((t / T) + cosine_s) / (1 + cosine_s) * torch.pi / 2) ** 2
            alpha_bar =  alphas_cumprod / alphas_cumprod[0]

            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return torch.clamp(betas, 1e-8, 0.999)

        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x0)

        if t.dim() == 0:
            t = t.expand(x0.shape[0])

        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))

        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        return self.q_sample(x0, t, noise), noise
