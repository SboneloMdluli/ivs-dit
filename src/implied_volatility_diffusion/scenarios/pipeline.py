"""End-to-end scenario generation with penalty evaluation and VolGAN weighting."""

import numpy as np
import torch

from implied_volatility_diffusion.diffusion.arbitrage_torch import ArbitragePenalty
from implied_volatility_diffusion.scenarios.generators import JointScenarioGenerator
from implied_volatility_diffusion.scenarios.penalty import SurfaceArbitragePenalty
from implied_volatility_diffusion.scenarios.types import JointScenarioBatch, PenaltyWeightingResult
from implied_volatility_diffusion.scenarios.weighting import (
    volgan_exponential_weights,
    volgan_exponential_weights_torch,
)


def penalize_iv_surfaces(
    iv_surfaces: np.ndarray,
    penalty: SurfaceArbitragePenalty,
) -> np.ndarray:
    """Compute Φ(σ^i) for each scenario IV surface."""
    return penalty.batch(iv_surfaces)


def weight_scenarios_from_penalties(
    penalties: np.ndarray,
    beta: float,
) -> PenaltyWeightingResult:
    """Assign VolGAN weights w_i ∝ exp(-β Φ_i) from precomputed penalties."""
    phi = np.asarray(penalties, dtype=float).reshape(-1)
    weights = volgan_exponential_weights(phi, beta)
    return PenaltyWeightingResult(penalties=phi, weights=weights)


def penalize_and_weight_iv_surfaces(
    iv_surfaces: np.ndarray,
    penalty: SurfaceArbitragePenalty,
    beta: float,
) -> PenaltyWeightingResult:
    """Evaluate Φ then apply exponential VolGAN reweighting."""
    phi = penalize_iv_surfaces(iv_surfaces, penalty)
    return weight_scenarios_from_penalties(phi, beta)


def generate_weighted_joint_scenarios(
    generator: JointScenarioGenerator,
    penalty: SurfaceArbitragePenalty,
    n: int,
    beta: float,
    *,
    rng: np.random.Generator | None = None,
) -> JointScenarioBatch:
    """Generate N joint scenarios and return penalties with VolGAN weights."""
    log_returns, iv_surfaces = generator.generate(n, rng=rng)
    pw = penalize_and_weight_iv_surfaces(iv_surfaces, penalty, beta)
    return JointScenarioBatch(
        log_returns=log_returns,
        iv_surfaces=iv_surfaces,
        penalties=pw.penalties,
        weights=pw.weights,
    )


def penalize_and_weight_iv_surfaces_torch(
    iv: torch.Tensor,
    penalty: ArbitragePenalty,
    beta: float,
) -> PenaltyWeightingResult:
    """Torch path for generative models: Φ from :class:`ArbitragePenalty`, same weights."""
    phi_t = penalty.total(iv)
    phi = phi_t.detach().cpu().numpy().reshape(-1)
    w = volgan_exponential_weights_torch(phi_t, beta).detach().cpu().numpy().reshape(-1)
    return PenaltyWeightingResult(penalties=phi, weights=w)
