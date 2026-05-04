"""Reverse-diffusion training and sampling for IV surfaces."""

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
from implied_volatility_diffusion.diffusion.kl_autoencoder import (
    KLAutoencoder,
    KLAutoencoderOutput,
)
from implied_volatility_diffusion.diffusion.latent_blocks import (
    DownBlock,
    UpBlock,
    crop_tensor,
    groupnorm,
    pad_tensor,
)
from implied_volatility_diffusion.diffusion.latent_grid import (
    crop_surface,
    halving_spatial_factor,
    latent_padded_hw,
    latent_spatial_hw,
    pad_surface,
    symmetric_pad_widths,
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
    "DownBlock",
    "KLAutoencoder",
    "KLAutoencoderOutput",
    "ReverseDiffusion",
    "SinusoidalTimeEmbedding",
    "SurfaceGenerationDiagnostics",
    "TimeEmbeddingMLP",
    "UNet",
    "UpBlock",
    "VPNoiseScheduler",
    "arbitrage_violation_score",
    "build_backbone",
    "crop_surface",
    "crop_tensor",
    "groupnorm",
    "halving_spatial_factor",
    "iter_backbone_names",
    "latent_padded_hw",
    "latent_spatial_hw",
    "pad_surface",
    "pad_tensor",
    "plot_performance_metrics",
    "plot_surface_comparison",
    "regenerate_until_arb_free",
    "register_backbone",
    "symmetric_pad_widths",
    "torch_bs_call",
]
