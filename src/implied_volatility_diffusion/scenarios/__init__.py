"""Joint return / IV scenario generation with arbitrage penalties and VolGAN weights."""

from implied_volatility_diffusion.scenarios.generators import (
    CallableJointScenarioGenerator,
    FilteredHistoricalSettings,
    FilteredHistoricalSimulation,
    JointScenarioGenerator,
)
from implied_volatility_diffusion.scenarios.penalty import (
    PenaltyMatrices,
    SurfaceArbitragePenalty,
    SurfaceArbitrageWeights,
    smoothness_penalty,
    smoothness_penalty_moneyness,
    smoothness_penalty_tau,
)
from implied_volatility_diffusion.scenarios.pipeline import (
    generate_weighted_joint_scenarios,
    penalize_and_weight_iv_surfaces,
    penalize_and_weight_iv_surfaces_torch,
    penalize_iv_surfaces,
    weight_scenarios_from_penalties,
)
from implied_volatility_diffusion.scenarios.types import (
    JointHistoricalState,
    JointScenarioBatch,
    PenaltyWeightingResult,
)
from implied_volatility_diffusion.scenarios.weighting import (
    adaptive_beta,
    effective_sample_size,
    fraction_arbitrage_free,
    relative_entropy,
    volgan_exponential_weights,
    volgan_exponential_weights_torch,
)

__all__ = [
    "CallableJointScenarioGenerator",
    "FilteredHistoricalSettings",
    "FilteredHistoricalSimulation",
    "JointHistoricalState",
    "JointScenarioBatch",
    "JointScenarioGenerator",
    "PenaltyMatrices",
    "PenaltyWeightingResult",
    "SurfaceArbitragePenalty",
    "SurfaceArbitrageWeights",
    "adaptive_beta",
    "effective_sample_size",
    "fraction_arbitrage_free",
    "generate_weighted_joint_scenarios",
    "penalize_and_weight_iv_surfaces",
    "penalize_and_weight_iv_surfaces_torch",
    "penalize_iv_surfaces",
    "relative_entropy",
    "smoothness_penalty",
    "smoothness_penalty_moneyness",
    "smoothness_penalty_tau",
    "volgan_exponential_weights",
    "volgan_exponential_weights_torch",
    "weight_scenarios_from_penalties",
]
