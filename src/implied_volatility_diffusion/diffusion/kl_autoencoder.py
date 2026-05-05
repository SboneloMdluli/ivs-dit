"""KL-regularized autoencoder.
Encoder: ``(B, C, H, W)`` → Gaussian latent ``z``.
Decoder: ``z`` → ``(B, C, H, W)``.
Loss: ``MSE + β * KL(q(z|x) ‖ N(0,I))``.

Reference: Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022 (arXiv:2112.10752).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from implied_volatility_diffusion.diffusion.latent_blocks import (
    DownBlock,
    UpBlock,
    crop_tensor,
    groupnorm,
    pad_tensor,
)
from implied_volatility_diffusion.diffusion.latent_grid import (
    halving_spatial_factor,
    latent_padded_hw,
    latent_spatial_hw,
)


class _DiagonalGaussian:
    """Diagonal Gaussian posterior q(z|x); holds μ and log σ²."""

    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean = mean
        self.logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.std)

    def kl_loss(self) -> torch.Tensor:
        """Mean KL(q ‖ N(0,I)): −½(1 + log σ² − μ² − σ²)."""
        return (-0.5 * (1.0 + self.logvar - self.mean.pow(2) - self.logvar.exp())).mean()


@dataclass
class KLAutoencoderOutput:
    """Return bundle from :meth:`KLAutoencoder.forward` when ``return_output=True``."""

    reconstruction: torch.Tensor
    z: torch.Tensor
    posterior_mean: torch.Tensor
    rec_loss: torch.Tensor
    kl_loss: torch.Tensor


class KLAutoencoder(nn.Module):
    """KL-regularized conv autoencoder for IV surface grids."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
        num_downsample: int = 2,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        _ = halving_spatial_factor(num_downsample)
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.num_downsample = int(num_downsample)
        self.double_z = bool(double_z)
        enc_out = 2 * self.latent_channels if self.double_z else self.latent_channels

        # Encoder
        self.enc_stem = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, padding=1)
        self.enc_down = nn.ModuleList(DownBlock(self.base_channels) for _ in range(self.num_downsample))
        self.enc_norm = groupnorm(self.base_channels)
        self.enc_out = nn.Conv2d(self.base_channels, enc_out, kernel_size=3, padding=1)

        # Decoder
        self.dec_in = nn.Conv2d(self.latent_channels, self.base_channels, kernel_size=3, padding=1)
        self.dec_up = nn.ModuleList(UpBlock(self.base_channels) for _ in range(self.num_downsample))
        self.dec_norm = groupnorm(self.base_channels)
        self.dec_out = nn.Conv2d(self.base_channels, self.in_channels, kernel_size=3, padding=1)

    def padded_hw(self, h: int, w: int) -> tuple[int, int]:
        return latent_padded_hw(h, w, self.num_downsample)

    def latent_shape(self, h: int, w: int) -> tuple[int, int]:
        return latent_spatial_hw(h, w, self.num_downsample)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        """Return ``(z, mu, pads)``; pass ``pads`` and ``orig_hw`` to :meth:`decode`."""
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W); got {tuple(x.shape)}")
        f = halving_spatial_factor(self.num_downsample)
        x_pad, pads = pad_tensor(x, multiple_h=f, multiple_w=f)
        h = self.enc_stem(x_pad)
        for blk in self.enc_down:
            h = blk(h)
        h = F.silu(self.enc_norm(h))
        h = self.enc_out(h)
        if self.double_z:
            mu, logvar = torch.chunk(h, 2, dim=1)
            posterior = _DiagonalGaussian(mu, logvar)
            return posterior.sample(), mu, pads
        return h, h, pads

    def decode(
        self,
        z: torch.Tensor,
        *,
        orig_hw: tuple[int, int],
        pads: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Decode latent ``z`` → IV surface ``(B, in_channels, H, W)``."""
        h = self.dec_in(z)
        for blk in self.dec_up:
            h = blk(h)
        h = F.silu(self.dec_norm(h))
        h = self.dec_out(h)
        return crop_tensor(h, pads, target_h=orig_hw[0], target_w=orig_hw[1])

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_output: bool = False,
    ) -> torch.Tensor | KLAutoencoderOutput:
        """Encode → sample z → decode. Set ``return_output=True`` for losses."""
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        f = halving_spatial_factor(self.num_downsample)
        x_pad, pads = pad_tensor(x, multiple_h=f, multiple_w=f)
        h = self.enc_stem(x_pad)
        for blk in self.enc_down:
            h = blk(h)
        h = F.silu(self.enc_norm(h))
        h = self.enc_out(h)

        if self.double_z:
            mu, logvar = torch.chunk(h, 2, dim=1)
            posterior = _DiagonalGaussian(mu, logvar)
            z = posterior.sample()
            kl = posterior.kl_loss()
        else:
            mu = z = h
            kl = torch.zeros((), device=x.device, dtype=x.dtype)

        recon = self.decode(z, orig_hw=(orig_h, orig_w), pads=pads)

        if return_output:
            return KLAutoencoderOutput(
                reconstruction=recon,
                z=z,
                posterior_mean=mu,
                rec_loss=F.mse_loss(recon, x),
                kl_loss=kl,
            )
        return recon


__all__ = ["KLAutoencoder", "KLAutoencoderOutput"]
