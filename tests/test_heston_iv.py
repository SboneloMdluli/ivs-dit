"""Tests for Heston COS, Black–Scholes implied vol, and config/LHS helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from implied_volatility_diffusion.iv_surface import grid_axes
from implied_volatility_diffusion.synthetic_ivs_generator.black_scholes import (
    call_price,
    implied_volatility,
)
from implied_volatility_diffusion.synthetic_ivs_generator.heston_cos import (
    _heston_cf,
    heston_call_cos,
)
from implied_volatility_diffusion.synthetic_ivs_generator.heston_iv_surface import (
    implied_vol_surfaces_lhs,
    lhs_heston_params,
    lhs_heston_params_multi_batch,
    load_heston_iv_surface_config,
)


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


def test_grid_tau_prepend() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    _, tau = grid_axes(cfg)
    assert tau[0] < 0.2
    assert abs(float(tau[-1]) - 1.0) < 1e-9


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


@pytest.mark.slow
def test_lhs_surfaces_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    _, m, tau, iv = implied_vol_surfaces_lhs(cfg, n_samples=1, n_batches=1, seed=0)
    assert iv.shape == (1, len(m), len(tau))
    assert np.all(np.isfinite(iv))
    assert np.all(iv > 0)
