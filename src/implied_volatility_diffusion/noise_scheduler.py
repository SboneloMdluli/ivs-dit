"""Closed-form noising scheduler for the forward diffusion process.

This scheduler follows a variance-preserving (VP/OU) process with constant
beta. Its marginals are available in closed form:

    x_t = exp(-0.5 * beta * t) * x_0 + sqrt(1 - exp(-beta * t)) * eps,
    eps ~ N(0, I).

As t -> inf, x_t converges to N(0, I), so variance remains bounded.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VPNoiseScheduler:
    """Variance-preserving forward noising process with closed-form marginals."""

    beta: float = 1.0

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        """Return (alpha_t, sigma_t) for time t >= 0."""
        if t < 0.0:
            raise ValueError("t must be non-negative")
        alpha = float(np.exp(-0.5 * self.beta * t))
        sigma = float(np.sqrt(max(0.0, 1.0 - np.exp(-self.beta * t))))
        return alpha, sigma

    def add_noise(self, x0: np.ndarray, t: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample x_t conditioned on x_0."""
        alpha, sigma = self.alpha_sigma(t)
        z = (rng or np.random.default_rng()).standard_normal(size=np.shape(x0))
        return alpha * np.asarray(x0, dtype=float) + sigma * z
