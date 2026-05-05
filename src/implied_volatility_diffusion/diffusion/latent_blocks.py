"""Shared conv building blocks for the LDM encoder–decoder (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from implied_volatility_diffusion.diffusion.latent_grid import symmetric_pad_widths


def pad_tensor(
    x: torch.Tensor,
    *,
    multiple_h: int,
    multiple_w: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h, w = x.shape[-2:]
    pads = symmetric_pad_widths(h, w, multiple_h=multiple_h, multiple_w=multiple_w)
    pad_top, pad_bottom, pad_left, pad_right = pads
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return padded, pads


def crop_tensor(
    x: torch.Tensor,
    pads: tuple[int, int, int, int],
    *,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    pad_top, _pad_bottom, pad_left, _pad_right = pads
    return x[..., pad_top : pad_top + target_h, pad_left : pad_left + target_w]


def groupnorm(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=max(1, g), num_channels=ch)


class DownBlock(nn.Module):
    """Norm → conv → stride-2 conv (halve spatial size)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            groupnorm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """×2 nearest upsample → two convs."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            groupnorm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.net(x)


__all__ = [
    "DownBlock",
    "UpBlock",
    "crop_tensor",
    "groupnorm",
    "pad_tensor",
]
