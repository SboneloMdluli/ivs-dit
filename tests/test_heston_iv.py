"""Tests for Heston COS, Black–Scholes implied vol, and config/LHS helpers."""

import math
from pathlib import Path

import numpy as np
import pytest

from implied_volatility_diffusion.iv_surface import grid_axes
from implied_volatility_diffusion.models.heston.heston_cos import _heston_cf, heston_call_cos
from implied_volatility_diffusion.pricing.black_scholes import bs_call_price_scalar as call_price
from implied_volatility_diffusion.pricing.implied_vol import implied_volatility
from implied_volatility_diffusion.synthetic.heston import (
    implied_vol_surface_for_heston_params as implied_vol_surface_for_params,
    implied_vol_surfaces_heston_lhs as implied_vol_surfaces_lhs,
    implied_vol_surfaces_heston_sequential_lhs as implied_vol_surfaces_sequential_lhs,
    lhs_heston_params,
    lhs_heston_params_multi_batch,
    load_heston_iv_surface_config,
)
from implied_volatility_diffusion.config import merge_config


def test_heston_cf_at_zero() -> None:
    u = np.array([0.0])
    phi = _heston_cf(u, 1.0, 100.0, 0.03, 0.0, 2.0, 0.04, 0.3, -0.7, 0.04)
    assert abs(complex(phi[0]) - 1.0) < 1e-10


def test_heston_cf_forward_measure() -> None:
    S, T, r, q = 100.0, 0.5, 0.03, 0.0
    u = np.array([-1j], dtype=complex)
    phi = _heston_cf(u, T, S, r, q, 2.0, 0.04, 0.3, -0.7, 0.04)
    fwd = S * math.exp((r - q) * T)
    assert abs(complex(phi[0]) - fwd) < 1e-6


def test_implied_vol_clamps_slightly_sub_intrinsic() -> None:
    """COS-style noise can price just below intrinsic; inversion should not raise."""
    S, K, T, r, _ = 100.0, 100.0, 0.5, 0.03, 0.25
    intr = max(S - K * math.exp(-r * T), 0.0)
    iv = implied_volatility(intr - 1e-6, S, K, T, r)
    assert math.isfinite(iv) and iv > 0.0


def test_bs_implied_vol_roundtrip() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.25
    price = call_price(S, K, T, r, sigma)
    iv = implied_volatility(price, S, K, T, r)
    assert abs(iv - sigma) < 1e-6


def test_heston_degenerates_to_bs() -> None:
    """High mean reversion and low vol-of-vol ≈ constant variance (Black–Scholes)."""
    S, K, T, r = 100.0, 100.0, 0.5, 0.03
    price = heston_call_cos(S, K, T, r, 50.0, 0.04, 0.01, 0.0, 0.04)
    iv = implied_volatility(price, S, K, T, r)
    assert abs(iv - math.sqrt(0.04)) < 1e-3


def test_grid_tau_endpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    m, tau = grid_axes(cfg)
    grid_cfg = cfg["grid"]
    assert abs(float(tau[0]) - float(grid_cfg["tau"]["start_point"])) < 1e-9
    assert abs(float(tau[-1]) - float(grid_cfg["tau"]["end_point"])) < 1e-9
    assert abs(float(m[0]) - float(grid_cfg["moneyness"]["start_point"])) < 1e-9
    assert abs(float(m[-1]) - float(grid_cfg["moneyness"]["end_point"])) < 1e-9


def test_tau_extrapolate_below_aligns_short_maturity_column() -> None:
    """Short-tau columns are replaced from the first maturity at/above ``tau_extrapolate_below``."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    params = np.array([0.04, -0.4, 0.35, 0.04, 2.5, 0.02], dtype=float)
    _, tau, iv = implied_vol_surface_for_params(params, cfg)
    thr = float(cfg["implied_vol"]["tau_extrapolate_below"])
    j_ref = int(np.searchsorted(np.asarray(tau, dtype=float), thr, side="left"))
    assert j_ref < len(tau) and float(tau[j_ref]) + 1e-12 >= thr
    for j, tj in enumerate(tau):
        if float(tj) + 1e-12 < thr:
            np.testing.assert_allclose(iv[:, j], iv[:, j_ref], rtol=1e-12, atol=1e-12)


def test_load_config_and_lhs() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    params = lhs_heston_params(cfg, n_samples=5, seed=0)
    assert params.shape == (5, 6)


def test_lhs_multi_batch_and_log_uniform() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    p1 = lhs_heston_params_multi_batch(cfg, n_samples=8, n_batches=2, seed=1, seed_stride=999)
    assert p1.shape == (16, 6)
    assert np.all(p1[:, 0] > 0) and np.all(p1[:, 3] > 0)  # v0, theta > 0


def test_sequential_ivs_shape_and_state_override() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    cfg = merge_config(
        cfg,
        {
            "lhs": {"n_samples": 2, "n_batches": 1, "seed": 91},
            "sequential_ivs": {"n_steps": 3, "dt": 0.01},
        },
    )
    params, m, tau, iv = implied_vol_surfaces_sequential_lhs(cfg, n_samples=2, seed=91)
    assert params.shape == (2, 6)
    assert iv.shape == (2, 3, len(m), len(tau))
    assert np.all(np.isfinite(iv)) and np.all(iv > 0)

    row = lhs_heston_params(cfg, n_samples=1, seed=5)[0]
    _, _, iv_a = implied_vol_surface_for_params(row, cfg)
    _, _, iv_b = implied_vol_surface_for_params(row, cfg, spot=100.0, inst_var=float(row[0]))
    assert np.allclose(iv_a, iv_b)


def test_lhs_satisfies_feller() -> None:
    """LHS draws clip ``sigma_v`` so ``2*kappa*theta >= sigma_v^2`` for every row."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    p = lhs_heston_params_multi_batch(cfg, n_samples=64, n_batches=2, seed=7)
    sigma, theta, kappa = p[:, 2], p[:, 3], p[:, 4]
    assert np.all(2.0 * kappa * theta + 1e-15 >= sigma * sigma)


@pytest.mark.slow
def test_lhs_surfaces_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    _, m, tau, iv = implied_vol_surfaces_lhs(cfg, n_samples=1, n_batches=1, seed=0)
    assert iv.shape == (1, len(m), len(tau))
    assert np.all(np.isfinite(iv))
    assert np.all(iv > 0)
