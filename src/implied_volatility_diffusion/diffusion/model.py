"""API works in **unnormalized IV surfaces**:
and ``ReverseDiffusion`` returns sampled IV surfaces.
"""

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from implied_volatility_diffusion.core.normalization import SurfaceNormalizer
from implied_volatility_diffusion.diffusion.backbones.base import (
    DenoisingBackbone,
    build_backbone,
)
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler

_DEFAULT_IV_FLOOR = 1e-8


def _broadcast(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape ``(B,)`` -> ``(B, 1, 1, ...)`` to broadcast against ``ref``."""
    return value.view(value.shape[0], *([1] * (ref.dim() - 1)))


class DiffusionModel(nn.Module):
    """The model takes / returns **unnormalized IV surfaces** in its public API.
    Denoising happens in z-space (``z = (log(iv) - mean) / std``),

    Args:
        backbone: Any :class:`DenoisingBackbone` (e.g. :class: UNet/ ViT).
        scheduler: VP noise schedule.
        mean: per-cell mean of ``log(IV)``.
        std:  per-cell std of ``log(IV)``.
        iv_floor: Clamp applied before ``log`` for numerical safety.
        prediction_type: ``"epsilon"`` (default) or ``"x0"``. Inherits from
            ``backbone.prediction_type`` if not given.
    """

    def __init__(
        self,
        backbone: DenoisingBackbone | nn.Module,
        scheduler: VPNoiseScheduler,
        *,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        iv_floor: float = _DEFAULT_IV_FLOOR,
        prediction_type: str | None = None,
    ) -> None:
        super().__init__()
        mean_t = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        std_t = torch.as_tensor(np.asarray(std), dtype=torch.float32)

        self.backbone = backbone
        self.scheduler = scheduler
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)
        self.iv_floor = float(iv_floor)
        self.prediction_type = prediction_type

    @classmethod
    def from_surface_normalizer(
        cls,
        backbone: DenoisingBackbone | nn.Module,
        scheduler: VPNoiseScheduler,
        normalizer: "SurfaceNormalizer",
        **kwargs: Any,
    ) -> "DiffusionModel":
        """Build from a fitted :class:`SurfaceNormalizer`"""
        return cls(
            backbone,
            scheduler,
            mean=normalizer.mean,
            std=normalizer.std,
            iv_floor=getattr(normalizer, "iv_floor", _DEFAULT_IV_FLOOR),
            **kwargs,
        )

    @classmethod
    def with_unit_stats(
        cls,
        backbone: DenoisingBackbone | nn.Module,
        scheduler: VPNoiseScheduler,
        grid_shape: tuple[int, int],
        **kwargs: Any,
    ) -> "DiffusionModel":
        """Build a passthrough model (mean=0, std=1) for testing or pre-normalized data."""
        return cls(
            backbone,
            scheduler,
            mean=np.zeros(grid_shape, dtype=np.float32),
            std=np.ones(grid_shape, dtype=np.float32),
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        scheduler: VPNoiseScheduler,
        *,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
    ) -> "DiffusionModel":
        """Build with the backbone selected by ``cfg['backbone']`` (registry name).

        ``cfg`` should look like::

            {"backbone": "unet", "backbone_kwargs": {...}, "iv_floor": 1e-8}
        """
        name = str(cfg.get("backbone", "unet"))
        backbone_cfg = cfg.get("backbone_kwargs") or {}
        backbone = build_backbone(name, backbone_cfg)
        return cls(
            backbone,
            scheduler,
            mean=mean,
            std=std,
            iv_floor=float(cfg.get("iv_floor", _DEFAULT_IV_FLOOR)),
            prediction_type=cfg.get("prediction_type"),
        )

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.mean.shape[0]), int(self.mean.shape[1])

    @property
    def in_channels(self) -> int:
        return int(getattr(self.backbone, "in_channels", 1))

    @property
    def out_channels(self) -> int:
        return int(getattr(self.backbone, "out_channels", 1))

    def _check_grid(self, x: torch.Tensor) -> None:
        if x.shape[-2:] != self.grid_shape:
            raise ValueError(f"trailing shape {tuple(x.shape[-2:])} must match grid {self.grid_shape}")

    def normalize(self, iv: torch.Tensor) -> torch.Tensor:
        """Map unnormalized IV -> z = ``(log(clamp(iv, iv_floor)) - mean) / std``."""
        self._check_grid(iv)
        log_iv = torch.log(torch.clamp(iv, min=self.iv_floor))
        return (log_iv - self.mean) / self.std

    def denormalize(self, z: torch.Tensor, *, return_log_iv: bool = False) -> torch.Tensor:
        """Z to unnormalized IVS"""
        self._check_grid(z)
        log_iv = z * self.std + self.mean
        return log_iv if return_log_iv else torch.exp(log_iv)

    def add_noise(
        self,
        iv0: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward diffusion from a clean IV surface.

        Returns ``(z_t, z0, noise)`` where ``z_t`` is the z-space noisy state
        used by the backbone, ``z0`` is the normalized clean surface, and
        ``noise`` is the realized Gaussian noise (so the loss can compare).
        """
        z0 = self.normalize(iv0)
        if noise is None:
            noise = torch.randn_like(z0)
        z_t = self.scheduler.q_sample(z0, t, noise=noise)
        return z_t, z0, noise

    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backbone forward pass in z-space."""
        return self.backbone(z_t, t, cond) if cond is not None else self.backbone(z_t, t)

    def predict_x0_z(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        clip: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Predicted clean **z-space** surface from a noisy ``z_t``."""
        pred = self.predict_noise(z_t, t, cond)
        alpha_bar = _broadcast(self.scheduler.alpha_bar_at(t), z_t)
        if self.prediction_type == "epsilon":
            sqrt_one_minus_ab = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
            sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-8))
            x0 = (z_t - sqrt_one_minus_ab * pred) / sqrt_ab
        else:
            x0 = pred
        if clip is not None:
            x0 = torch.clamp(x0, clip[0], clip[1])
        return x0

    def predict_iv(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        clip_z: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Predicted clean **IV** surface from a noisy ``z_t`` (denormalized)."""
        return self.denormalize(self.predict_x0_z(z_t, t, cond, clip=clip_z))

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backbone passthrough in z-space (used by the sampler).

        For the IV-space training entry point, see :class:`DiffusionLoss`.
        For the IV-space sampling entry point, see :class:`ReverseDiffusion`.
        """
        return self.predict_noise(z_t, t, cond)


__all__ = ["DiffusionModel"]
