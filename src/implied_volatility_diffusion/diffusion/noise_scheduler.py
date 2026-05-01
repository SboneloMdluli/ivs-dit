"""Variance-preserving (VP) forward diffusion scheduler."""

import numpy as np
import torch
import torch.nn as nn


class VPNoiseScheduler(nn.Module):
    """VP scheduler with discrete DDPM-style timesteps plus standalone helpers.

    Supports:
    - discrete noising with ``q_sample`` for model training
    - standalone forward noising from continuous time ``t`` via ``add_noise``
    - convenience accessors (:meth:`extract`, :meth:`alpha_bar_at`) used by the
      reverse-process sampler and arbitrage-weighted loss
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
        beta: float | None = None,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.beta = float(beta) if beta is not None else None

        betas = self._build_betas(beta_schedule, self.timesteps, beta_start, beta_end, cosine_s)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=alpha_bar.dtype), alpha_bar[:-1]])
        # Posterior variance for DDPM ancestral sampler (Ho et al. 2020 eq. 7).
        posterior_var = betas * (1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar, min=1e-20)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_variance", posterior_var)

    @staticmethod
    def _build_betas(
        schedule: str,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        cosine_s: float,
    ) -> torch.Tensor:
        if timesteps <= 0:
            raise ValueError("timesteps must be > 0")
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        if schedule == "cosine":
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps, dtype=torch.float32)

            alphas_cumprod = torch.cos(((t / timesteps) + cosine_s) / (1 + cosine_s) * torch.pi / 2) ** 2
            alpha_bar = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

            return torch.clamp(betas, 1e-8, 0.999)
        raise ValueError(f"Unknown schedule: {schedule}")

    def alpha_sigma(self, t: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return closed-form VP ``(alpha_t, sigma_t)`` for time ``t >= 0``."""
        if self.beta is None:
            raise ValueError("alpha_sigma requires constant beta; pass beta=... to VPNoiseScheduler")
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        if torch.any(t_tensor < 0):
            raise ValueError("t must be non-negative")

        alpha = torch.exp(-0.5 * self.beta * t_tensor)
        sigma = torch.sqrt(torch.clamp(1.0 - torch.exp(-self.beta * t_tensor), min=0.0))
        return alpha, sigma

    def add_noise(
        self,
        x0: np.ndarray | torch.Tensor,
        t: float,
        *,
        rng: np.random.Generator | None = None,
        generator: torch.Generator | None = None,
        noise: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Standalone closed-form noising with a NumPy-or-Torch API.

        - NumPy input -> NumPy output
        - Torch input -> Torch output
        """
        alpha, sigma = self.alpha_sigma(float(t))
        if isinstance(x0, torch.Tensor):
            alpha_t = alpha.to(device=x0.device, dtype=x0.dtype)
            sigma_t = sigma.to(device=x0.device, dtype=x0.dtype)
            if noise is None:
                eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
            else:
                if not isinstance(noise, torch.Tensor):
                    raise TypeError("noise must be torch.Tensor when x0 is torch.Tensor")
                eps = noise.to(device=x0.device, dtype=x0.dtype)
            return alpha_t * x0 + sigma_t * eps

        x0_np = np.asarray(x0, dtype=float)
        if noise is None:
            eps_np = (rng or np.random.default_rng()).standard_normal(size=np.shape(x0_np))
        else:
            eps_np = np.asarray(noise, dtype=float)
            if eps_np.shape != x0_np.shape:
                raise ValueError(f"noise shape {eps_np.shape} != x0 shape {x0_np.shape}")
        return float(alpha.item()) * x0_np + float(sigma.item()) * eps_np

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discrete forward process: ``x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps``."""
        if noise is None:
            noise = torch.randn_like(x0)
        if isinstance(t, int):
            t = torch.full((x0.shape[0],), t, dtype=torch.long, device=x0.device)
        elif t.dim() == 0:
            t = t.expand(x0.shape[0])
        t = t.to(dtype=torch.long, device=x0.device)

        # Extract alpha_bar for the given timesteps
        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))

        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def forward_process(
        self,
        x0: torch.Tensor,
        t_values: torch.Tensor | list[int] | list[float],
        *,
        generator: torch.Generator | None = None,
        mode: str = "discrete",
    ) -> torch.Tensor:
        """Return stacked noised samples for a sequence of times.

        ``mode='discrete'`` uses DDPM-style integer timesteps with ``q_sample``.
        ``mode='continuous'`` uses closed-form VP noising with ``add_noise``.
        """
        t_tensor = torch.as_tensor(t_values, device=x0.device).flatten()
        outputs = []
        if mode == "discrete":
            t_tensor = t_tensor.to(dtype=torch.long)
            for t in t_tensor:
                noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
                outputs.append(self.q_sample(x0, t, noise=noise))
            return torch.stack(outputs, dim=0)
        if mode == "continuous":
            t_tensor = t_tensor.to(dtype=torch.float32)
            for t in t_tensor:
                outputs.append(self.add_noise(x0, float(t.item()), generator=generator))
            return torch.stack(outputs, dim=0)
        raise ValueError(f"Unknown mode: {mode}. Expected 'discrete' or 'continuous'.")

    def extract(self, buffer_name: str, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        """Index a registered buffer by ``t`` and reshape to broadcast against ``x_shape``."""
        buf = getattr(self, buffer_name)
        out = buf.gather(0, t.to(dtype=torch.long, device=buf.device))
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def alpha_bar_at(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``alpha_bar_t`` for a (B,)-shaped integer index tensor."""
        idx = t.to(dtype=torch.long, device=self.alpha_bar.device)
        return self.alpha_bar.gather(0, idx)

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        return self.q_sample(x0, t, noise), noise
