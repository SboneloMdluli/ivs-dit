"""Reverse-diffusion training and sampling for IV surfaces.

Public API in this package operates on unnormalized IV surfaces:
"""

from implied_volatility_diffusion.diffusion.arbitrage_torch import (
    ArbitragePenalty,
    ArbitrageWeights,
    torch_bs_call,
)
from implied_volatility_diffusion.diffusion.backbones import (
    DenoisingBackbone,
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    UNet,
    build_backbone,
    iter_backbone_names,
    register_backbone,
)
from implied_volatility_diffusion.diffusion.losses import (
    ArbitrageSchedule,
    DiffusionLoss,
    DiffusionLossConfig,
)
from implied_volatility_diffusion.diffusion.model import DiffusionModel
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler
from implied_volatility_diffusion.diffusion.reverse import ReverseDiffusion
from implied_volatility_diffusion.diffusion.validation import (
    SurfaceGenerationDiagnostics,
    arbitrage_violation_score,
    plot_performance_metrics,
    plot_surface_comparison,
    regenerate_until_arb_free,
)

__all__ = [
    "ArbitragePenalty",
    "ArbitrageSchedule",
    "ArbitrageWeights",
    "DenoisingBackbone",
    "DiffusionLoss",
    "DiffusionLossConfig",
    "DiffusionModel",
    "ReverseDiffusion",
    "VPNoiseScheduler",
    "SinusoidalTimeEmbedding",
    "TimeEmbeddingMLP",
    "UNet",
    "build_backbone",
    "iter_backbone_names",
    "register_backbone",
    "SurfaceGenerationDiagnostics",
    "arbitrage_violation_score",
    "plot_performance_metrics",
    "plot_surface_comparison",
    "regenerate_until_arb_free",
    "torch_bs_call",
]
