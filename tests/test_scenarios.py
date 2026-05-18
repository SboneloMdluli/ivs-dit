"""Tests for joint scenario generation and VolGAN penalty weighting."""

import numpy as np
import pytest
import torch

from implied_volatility_diffusion.diffusion.arbitrage_torch import ArbitragePenalty
from implied_volatility_diffusion.scenarios import (
    CallableJointScenarioGenerator,
    FilteredHistoricalSimulation,
    JointHistoricalState,
    SurfaceArbitragePenalty,
    generate_weighted_joint_scenarios,
    penalize_and_weight_iv_surfaces_torch,
    volgan_exponential_weights,
)


def _flat_iv(m: np.ndarray, tau: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    return np.full((m.size, tau.size), sigma, dtype=float)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    m = np.linspace(0.7, 1.3, 11)
    tau = np.linspace(0.1, 1.0, 6)
    return m, tau


def test_flat_surface_has_zero_penalty() -> None:
    m, tau = _grid()
    penalty = SurfaceArbitragePenalty(moneyness=m, tau=tau, spot=100.0, rate=0.02)
    phi = penalty(_flat_iv(m, tau))
    assert phi == pytest.approx(0.0, abs=1e-12)


def test_volgan_weights_favor_low_penalty() -> None:
    phi = np.array([0.0, 0.5, 2.0])
    w = volgan_exponential_weights(phi, beta=2.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    assert w[0] == pytest.approx(np.exp(-2.0 * 0.0) / np.sum(np.exp(-2.0 * phi)))


def test_volgan_uniform_when_beta_zero() -> None:
    phi = np.array([0.0, 1.0, 5.0])
    w = volgan_exponential_weights(phi, beta=0.0)
    assert np.allclose(w, 1.0 / 3.0)


def test_fhs_resamples_history() -> None:
    m, tau = _grid()
    h = 20
    rng = np.random.default_rng(1)
    log_r_hist = rng.normal(0.0, 0.01, h)
    iv_hist = np.stack([_flat_iv(m, tau, 0.20 + 0.05 * rng.random()) for _ in range(h)])
    fhs = FilteredHistoricalSimulation(JointHistoricalState(log_r_hist, iv_hist))
    log_r, iv = fhs.generate(5, rng=np.random.default_rng(2))
    assert log_r.shape == (5,)
    assert iv.shape == (5, m.size, tau.size)


def test_generate_weighted_joint_scenarios() -> None:
    m, tau = _grid()
    base = _flat_iv(m, tau)

    def _fn(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        del rng
        return np.zeros(n), np.broadcast_to(base, (n, *base.shape)).copy()

    gen = CallableJointScenarioGenerator(_fn)
    penalty = SurfaceArbitragePenalty(moneyness=m, tau=tau, spot=100.0, rate=0.0)
    batch = generate_weighted_joint_scenarios(gen, penalty, n=4, beta=1.0, rng=np.random.default_rng(0))
    assert batch.n_scenarios == 4
    assert np.all(batch.penalties == pytest.approx(0.0, abs=1e-10))
    assert np.allclose(batch.weights, 0.25)


def test_callable_generator_wrapper() -> None:
    m, tau = _grid()
    base = _flat_iv(m, tau)

    def _fn(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        del rng
        return np.zeros(n), np.broadcast_to(base, (n, *base.shape)).copy()

    gen = CallableJointScenarioGenerator(_fn)
    log_r, iv = gen.generate(3, rng=np.random.default_rng(0))
    assert log_r.shape == (3,)
    assert iv.shape == (3, m.size, tau.size)


def test_torch_penalty_weighting_matches_numpy_on_flat() -> None:
    m, tau = _grid()
    iv = _flat_iv(m, tau)
    batch_iv = torch.as_tensor(iv, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    log_k = np.log(m)
    torch_pen = ArbitragePenalty(log_moneyness=log_k, tau=tau, spot=100.0, rate=0.0)
    np_pen = SurfaceArbitragePenalty(moneyness=m, tau=tau, spot=100.0, rate=0.0)
    phi_np = np_pen.batch(np.stack([iv, iv, iv]))
    result = penalize_and_weight_iv_surfaces_torch(batch_iv, torch_pen, beta=1.5)
    assert np.allclose(result.penalties, phi_np, atol=1e-6)
    assert result.weights.sum() == pytest.approx(1.0)
