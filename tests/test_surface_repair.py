"""Tests for IV surface smoothness and no-arbitrage repair."""

from pathlib import Path

import numpy as np
import yaml

from implied_volatility_diffusion import (
    ScenarioRepairResult,
    SurfaceRepairSettings,
    TargetedRepairSettings,
    check_iv_surface_arbitrage,
    repair_and_reweight_scenarios,
    repair_calendar_monotone,
    repair_iv_surface,
    repair_iv_surface_targeted,
    volgan_generative_repair_settings,
)
from implied_volatility_diffusion.core.surface_repair import repair_butterfly_convex
from implied_volatility_diffusion.models.sabr import SabrModel
from implied_volatility_diffusion.scenarios.penalty import (
    SurfaceArbitragePenalty,
    smoothness_penalty,
    smoothness_penalty_moneyness,
)
from implied_volatility_diffusion.scenarios.weighting import (
    adaptive_beta,
    effective_sample_size,
    fraction_arbitrage_free,
    relative_entropy,
    volgan_exponential_weights,
)
from implied_volatility_diffusion.synthetic.sabr import implied_vol_surfaces_sabr_lhs
from implied_volatility_diffusion.synthetic.surface import build_surfaces


def _flat_bs_surface(moneyness: np.ndarray, tau: np.ndarray, sigma: float) -> np.ndarray:
    return np.full((moneyness.size, tau.size), float(sigma), dtype=float)


def test_calendar_repair_restores_monotone_total_variance() -> None:
    m = np.linspace(0.5, 2.5, 21)
    tau = np.array([0.25, 0.5, 1.0])
    iv = np.empty((m.size, tau.size), dtype=float)
    iv[:, 0] = 0.40
    iv[:, 1] = 0.20
    iv[:, 2] = 0.20

    rep = repair_calendar_monotone(iv, tau)
    w = (rep**2) * tau[None, :]
    assert np.all(np.diff(w, axis=1) >= -1e-10)

    report = check_iv_surface_arbitrage(rep, m, tau, spot=100.0, rate=0.0)
    assert report.calendar_ok


def test_butterfly_repair_preserves_sabr_smile_shape() -> None:
    """Bound-saturated wings must not collapse to iv_floor after call projection."""
    cfg = yaml.safe_load(Path("config/sabr_iv_surface.yaml").read_text())
    model = SabrModel.from_config(cfg)
    batch = build_surfaces(model, cfg, np.array([[0.2, -0.85, 0.4]]), spot_override=100.0)
    m, tau, iv = batch.moneyness, batch.tau, batch.iv[0]
    spot, rate = float(cfg["market"]["spot"]), float(cfg["market"]["r"])

    before = check_iv_surface_arbitrage(iv, m, tau, spot=spot, rate=rate, tol=1e-6)
    assert before.arbitrage_free

    rep = repair_butterfly_convex(iv, m, tau, spot=spot, rate=rate, repair_wings=True)
    left_short = (m[:, None] < 1.0) & (tau[None, :] < 0.5)
    assert np.all(rep[left_short] > 0.01)
    assert np.all(rep[m > 1.0, :] > 0.01)
    left_rmse = float(np.sqrt(np.mean((rep[left_short] - iv[left_short]) ** 2)))
    right_rmse = float(np.sqrt(np.mean((rep[m > 1.0, :] - iv[m > 1.0, :]) ** 2)))
    assert left_rmse < 1e-4
    assert right_rmse < 1e-4


def test_sabr_guard_does_not_destroy_clean_surface() -> None:
    cfg = yaml.safe_load(Path("config/sabr_iv_surface.yaml").read_text())
    cfg["lhs"]["n_samples"] = 2
    _, m, tau, iv_batch = implied_vol_surfaces_sabr_lhs(cfg)
    spot, rate = float(cfg["market"]["spot"]), float(cfg["market"]["r"])

    left_short = (m[:, None] < 1.0) & (tau[None, :] < 0.5)
    for iv in iv_batch:
        assert np.nanmin(iv[left_short]) > 0.01
        report = check_iv_surface_arbitrage(iv, m, tau, spot=spot, rate=rate, tol=1e-6)
        assert report.arbitrage_free


def test_butterfly_repair_on_concave_smile() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.2, 1.0, 8)
    iv = np.empty((m.size, tau.size), dtype=float)
    for j in range(tau.size):
        iv[:, j] = 0.40 - 0.30 * np.exp(-((m - 1.0) ** 2) / 0.01)

    before = check_iv_surface_arbitrage(iv, m, tau, spot=100.0, rate=0.0)
    assert not before.butterfly_ok

    rep = repair_butterfly_convex(iv, m, tau, spot=100.0, rate=0.0)
    after = check_iv_surface_arbitrage(rep, m, tau, spot=100.0, rate=0.0, tol=1e-6)
    assert after.butterfly_ok
    assert after.bounds_ok


def test_volgan_generative_settings_only_repair_when_violated() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.2, 1.0, 8)
    iv = _flat_bs_surface(m, tau, 0.25)
    settings = volgan_generative_repair_settings(tol=1e-6)
    assert settings.only_if_violated is True
    repaired = repair_iv_surface(iv, m, tau, spot=100.0, rate=0.0, settings=settings)
    assert np.allclose(repaired, iv)


def test_full_repair_pipeline_on_perturbed_flat_surface() -> None:
    m = np.linspace(0.5, 2.5, 31)
    tau = np.linspace(0.05, 2.0, 15)
    rng = np.random.default_rng(42)
    iv = _flat_bs_surface(m, tau, 0.25)
    noise = rng.normal(0.0, 0.02, size=iv.shape)
    iv = np.clip(iv + noise, 0.05, 1.5)
    iv[:, 3] *= 0.85  # inject calendar stress

    settings = SurfaceRepairSettings(
        smooth_sigma_log_moneyness=0.5,
        smooth_sigma_tau=0.3,
        max_iterations=8,
        tol=1e-6,
    )
    repaired = repair_iv_surface(iv, m, tau, spot=100.0, rate=0.02, settings=settings)
    report = check_iv_surface_arbitrage(repaired, m, tau, spot=100.0, rate=0.02, tol=1e-6)
    assert report.arbitrage_free


def test_penalty_matrices_zero_for_arbitrage_free_surface() -> None:
    m = np.linspace(0.7, 1.3, 11)
    tau = np.linspace(0.1, 1.0, 5)
    iv = _flat_bs_surface(m, tau, 0.25)
    penalty = SurfaceArbitragePenalty(moneyness=m, tau=tau, spot=100.0, rate=0.0)
    pm = penalty.penalty_matrices(iv)
    assert pm.arbitrage_free
    assert pm.phi == 0.0
    assert pm.P1_calendar.shape == (m.size, tau.size)
    assert pm.P2_call.shape == (m.size, tau.size)
    assert pm.P3_butterfly.shape == (m.size, tau.size)


def test_penalty_matrices_detect_calendar_violation() -> None:
    m = np.linspace(0.7, 1.3, 11)
    tau = np.array([0.25, 0.5, 1.0])
    iv = np.full((m.size, tau.size), 0.25)
    iv[:, 1] = 0.10  # artificially low middle maturity -> calendar violation
    penalty = SurfaceArbitragePenalty(moneyness=m, tau=tau, spot=100.0, rate=0.0)
    pm = penalty.penalty_matrices(iv)
    assert pm.phi_calendar > 0.0
    assert not pm.arbitrage_free


def test_penalty_matrices_detect_butterfly_violation() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.2, 1.0, 5)
    iv = np.full((m.size, tau.size), 0.25)
    for j in range(tau.size):
        iv[:, j] = 0.40 - 0.30 * np.exp(-((m - 1.0) ** 2) / 0.01)

    penalty = SurfaceArbitragePenalty(moneyness=m, tau=tau, spot=100.0, rate=0.0)
    pm = penalty.penalty_matrices(iv)
    assert pm.phi_butterfly > 0.0


def test_smoothness_penalty_zero_for_flat_surface() -> None:
    m = np.linspace(0.7, 1.3, 11)
    tau = np.linspace(0.1, 1.0, 5)
    iv = _flat_bs_surface(m, tau, 0.25)
    lm, lt = smoothness_penalty(iv, m, tau)
    assert lm == 0.0
    assert lt == 0.0


def test_smoothness_penalty_positive_for_rough_surface() -> None:
    m = np.linspace(0.7, 1.3, 11)
    tau = np.linspace(0.1, 1.0, 5)
    rng = np.random.default_rng(123)
    iv = _flat_bs_surface(m, tau, 0.25) + rng.normal(0.0, 0.05, size=(m.size, tau.size))
    iv = np.clip(iv, 0.05, 1.0)
    lm, lt = smoothness_penalty(iv, m, tau)
    assert lm > 0.0
    assert lt > 0.0


def test_smoothness_moneyness_increases_with_roughness() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.1, 1.0, 5)
    log_iv_smooth = np.log(np.full((m.size, tau.size), 0.25))
    rng = np.random.default_rng(42)
    log_iv_rough = log_iv_smooth + rng.normal(0.0, 0.1, size=log_iv_smooth.shape)
    lm_smooth = smoothness_penalty_moneyness(log_iv_smooth, m)
    lm_rough = smoothness_penalty_moneyness(log_iv_rough, m)
    assert lm_rough > lm_smooth


def test_volgan_weights_uniform_for_zero_penalties() -> None:
    phi = np.zeros(100)
    w = volgan_exponential_weights(phi, beta=100.0)
    assert np.allclose(w, 1.0 / 100)


def test_volgan_weights_concentrate_on_low_penalty() -> None:
    phi = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    w = volgan_exponential_weights(phi, beta=100.0)
    assert w[0] > w[3] > w[4]
    assert np.sum(w[:3]) > 0.999


def test_adaptive_beta_computation() -> None:
    w_uniform = np.full(100, 1.0 / 100)
    beta = adaptive_beta(w_uniform, scale=500.0)
    assert beta == 500.0 / (1.0 / 100)
    assert beta == 50000.0


def test_relative_entropy_zero_for_uniform() -> None:
    w = np.full(50, 1.0 / 50)
    re = relative_entropy(w)
    assert abs(re) < 1e-10


def test_relative_entropy_positive_for_concentrated() -> None:
    phi = np.array([0.0, 0.0, 0.5, 1.0, 2.0])
    w = volgan_exponential_weights(phi, beta=10.0)
    re = relative_entropy(w)
    assert re > 0.0


def test_effective_sample_size_equals_n_for_uniform() -> None:
    n = 100
    w = np.full(n, 1.0 / n)
    ess = effective_sample_size(w)
    assert abs(ess - n) < 1e-10


def test_effective_sample_size_decreases_with_concentration() -> None:
    phi_mild = np.array([0.0, 0.0, 0.01, 0.02, 0.03])
    phi_severe = np.array([0.0, 0.0, 1.0, 5.0, 10.0])
    w_mild = volgan_exponential_weights(phi_mild, beta=10.0)
    w_severe = volgan_exponential_weights(phi_severe, beta=10.0)
    assert effective_sample_size(w_mild) > effective_sample_size(w_severe)


def test_fraction_arbitrage_free() -> None:
    phi = np.array([0.0, 0.0, 0.0, 0.1, 0.5])
    assert fraction_arbitrage_free(phi) == 0.6


def test_targeted_repair_preserves_clean_surface() -> None:
    m = np.linspace(0.7, 1.3, 21)
    tau = np.linspace(0.2, 1.0, 8)
    iv = _flat_bs_surface(m, tau, 0.25)
    repaired = repair_iv_surface_targeted(iv, m, tau, spot=100.0, rate=0.0)
    assert np.allclose(repaired, iv, atol=1e-6)


def test_targeted_repair_fixes_violations() -> None:
    m = np.linspace(0.5, 2.5, 31)
    tau = np.linspace(0.05, 2.0, 15)
    rng = np.random.default_rng(42)
    iv = _flat_bs_surface(m, tau, 0.25)
    noise = rng.normal(0.0, 0.02, size=iv.shape)
    iv = np.clip(iv + noise, 0.05, 1.5)
    iv[:, 3] *= 0.85

    settings = TargetedRepairSettings(
        smooth_sigma_log_moneyness=0.5,
        smooth_sigma_tau=0.3,
        max_iterations=8,
        tol=1e-6,
    )
    repaired = repair_iv_surface_targeted(iv, m, tau, spot=100.0, rate=0.02, settings=settings)
    report = check_iv_surface_arbitrage(repaired, m, tau, spot=100.0, rate=0.02, tol=1e-6)
    assert report.arbitrage_free


def test_repair_and_reweight_scenarios() -> None:
    m = np.linspace(0.7, 1.3, 15)
    tau = np.linspace(0.1, 1.0, 6)
    rng = np.random.default_rng(99)
    n_scenarios = 10
    iv_batch = np.stack(
        [_flat_bs_surface(m, tau, 0.25) + rng.normal(0.0, 0.01, (m.size, tau.size)) for _ in range(n_scenarios)]
    )
    iv_batch = np.clip(iv_batch, 0.05, 1.5)

    result = repair_and_reweight_scenarios(
        iv_batch,
        m,
        tau,
        spot=100.0,
        rate=0.0,
        repair_settings=SurfaceRepairSettings(max_iterations=4, tol=1e-6),
        beta=100.0,
    )
    assert isinstance(result, ScenarioRepairResult)
    assert result.n_scenarios == n_scenarios
    assert result.iv_surfaces.shape == iv_batch.shape
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.fraction_clean_after >= result.fraction_clean_before
