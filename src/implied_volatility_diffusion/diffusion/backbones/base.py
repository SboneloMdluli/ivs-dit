"""Abstract backbone API for the reverse-diffusion denoiser.

The reverse process and loss only ever talk to a backbone through this
interface, which means the U-Net here can be replaced with a ViT (or any
other denoiser) without touching the sampler or training loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

BackboneFactory = Callable[[Mapping[str, Any]], "DenoisingBackbone"]


class DenoisingBackbone(nn.Module, ABC):
    """Abstract denoiser ``f_theta(x_t, t) -> prediction``.

    Subclasses must declare their I/O contract via :attr:`in_channels` and
    :attr:`out_channels` and implement :meth:`forward`. The prediction target
    is whatever ``prediction_type`` the trainer expects (``epsilon`` by
    default, but ``x0`` and ``v`` are also valid choices).
    """

    in_channels: int = 1
    out_channels: int = 1
    prediction_type: str = "epsilon"

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the diffusion target from a noisy input.

        Args:
            x: ``(B, C_in, H, W)`` noisy surface.
            t: ``(B,)`` long tensor of discrete timesteps.
            cond: Optional conditioning tensor. Concrete backbones decide its
                shape: surface-conditioned denoisers (e.g. the U-Net used for
                next-day forecasting) expect ``(B, C_cond, H, W)`` matching
                ``x``'s grid, while scalar/embedding backbones may accept
                ``(B, D)``. Backbones may ignore this argument entirely.

        Returns:
            Tensor of shape ``(B, C_out, H, W)`` matching the input grid.
        """


_REGISTRY: dict[str, BackboneFactory] = {}


def register_backbone(name: str, factory: BackboneFactory) -> None:
    """Register a denoiser backbone factory under ``name``."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("backbone name must be non-empty")
    _REGISTRY[key] = factory


def get_backbone_factory(name: str) -> BackboneFactory:
    """Look up a registered backbone factory."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown backbone {name!r}; registered: {available}")
    return _REGISTRY[key]


def build_backbone(name: str, cfg: Mapping[str, Any] | None = None) -> "DenoisingBackbone":
    """Instantiate a backbone by registered name with ``cfg`` kwargs."""
    return get_backbone_factory(name)(cfg or {})


def iter_backbone_names() -> list[str]:
    """Return all registered backbone names (sorted)."""
    return sorted(_REGISTRY)


__all__ = [
    "BackboneFactory",
    "DenoisingBackbone",
    "build_backbone",
    "get_backbone_factory",
    "iter_backbone_names",
    "register_backbone",
]
