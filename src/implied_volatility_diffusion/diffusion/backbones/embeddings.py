"""Time/position embeddings shared by U-Net and ViT denoisers."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding dim must be a positive even integer")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # Match the annotated diffusion / DDPM reference:
        #   freq_i = exp(-log(10000) * i / (half - 1)),  i = 0..half-1.
        denom = max(half - 1, 1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / denom)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        # [sin(t * freq), cos(t * freq)]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    """Standard sinusoidal -> MLP embedding head used by both backbones."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def out_dim(self) -> int:
        # The hidden_dim is captured in the last Linear layer's out_features.
        return self.mlp[-1].out_features

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


__all__ = ["SinusoidalTimeEmbedding", "TimeEmbeddingMLP"]
