"""Unit tests for the Heston Milstein discretization and Feller helpers."""

import math

import numpy as np

from implied_volatility_diffusion.models.heston.simulation import (
    feller_index,
    is_feller_satisfied,
    milstein_step,
)


def test_feller_helpers() -> None:
    assert is_feller_satisfied(2.0, 0.04, 0.2)
    assert not is_feller_satisfied(0.5, 0.04, 0.5)
    assert math.isclose(feller_index(2.0, 0.04, 0.2), 2.0 * 2.0 * 0.04 - 0.04)


def _simulate_milstein_path(n_steps: int, *, sigma_v: float, kappa: float, theta: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    s, v = 100.0, 0.04
    s_path = [s]
    v_path = [v]
    for _ in range(n_steps):
        s, v = milstein_step(
            s,
            v,
            dt=1.0 / 252.0,
            r=0.02,
            q=0.0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=-0.7,
            rng=rng,
        )
        s_path.append(s)
        v_path.append(v)
    return np.asarray(s_path), np.asarray(v_path)


def test_milstein_keeps_variance_nonnegative_and_spot_positive() -> None:
    s_path, v_path = _simulate_milstein_path(500, sigma_v=1.0, kappa=1.0, theta=0.04, seed=11)
    assert np.all(v_path >= 0.0)
    assert np.all(s_path > 0.0)
    assert np.all(np.isfinite(s_path))
    assert np.all(np.isfinite(v_path))


def test_milstein_drift_only_returns_to_theta() -> None:
    """With sigma_v -> 0 the variance Milstein step is deterministic and reverts to theta."""
    rng = np.random.default_rng(0)
    v = 0.20
    for _ in range(20_000):
        _, v = milstein_step(
            s=100.0,
            v=v,
            dt=1.0 / 252.0,
            r=0.0,
            q=0.0,
            kappa=5.0,
            theta=0.04,
            sigma_v=1e-12,
            rho=0.0,
            rng=rng,
        )
    assert abs(v - 0.04) < 1e-3
