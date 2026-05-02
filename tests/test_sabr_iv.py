"""Tests for SABR Hagan IV, calibration, and surface builders."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from implied_volatility_diffusion.iv_surface import grid_axes
from implied_volatility_diffusion.models.sabr.calibration import (
    calibrate_params_for_expiries,
    calibrate_sabr_to_implied_vols,
    implied_vol_surface_from_calibrated_slices,
)
from implied_volatility_diffusion.models.sabr.hagan import sabr_hagan_lognormal_iv
from implied_volatility_diffusion.synthetic.sabr import (
    implied_vol_surface_for_sabr_params as implied_vol_surface_for_params,
)
from implied_volatility_diffusion.synthetic.sabr import implied_vol_surfaces_sabr_lhs as implied_vol_surfaces_lhs
from implied_volatility_diffusion.synthetic.sabr import (
    lhs_sabr_params,
)


def _load_sabr_cfg() -> dict:
    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "sabr_iv_surface.yaml").open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def test_hagan_atm_positive() -> None:
    F, K, T = 100.0, 100.0, 0.5
    iv = sabr_hagan_lognormal_iv(F, K, T, alpha=0.25, beta=0.5, rho=-0.2, nu=0.4)
    assert np.isfinite(iv) and iv > 0.0


def test_hagan_wing_differs_from_atm() -> None:
    F, T = 100.0, 0.5
    iv_atm = sabr_hagan_lognormal_iv(F, F, T, 0.25, 0.5, -0.3, 0.5)
    iv_put = sabr_hagan_lognormal_iv(F, 85.0, T, 0.25, 0.5, -0.3, 0.5)
    assert abs(iv_put - iv_atm) > 1e-4


def test_calibrate_recovers_synthetic_surface() -> None:
    F, T, beta = 100.0, 0.75, 0.5
    alpha_t, rho_t, nu_t = 0.22, -0.35, 0.45
    strikes = np.linspace(80.0, 120.0, 12)
    mkt = np.array(
        [
            sabr_hagan_lognormal_iv(F, float(k), T, alpha_t, beta, rho_t, nu_t)
            for k in strikes
        ],
        dtype=float,
    )
    alpha, rho, nu, res = calibrate_sabr_to_implied_vols(
        F, T, strikes, mkt, beta=beta, initial_guess=(0.2, 0.0, 0.4)
    )
    assert res.success
    assert abs(alpha - alpha_t) < 0.05
    assert abs(rho - rho_t) < 0.15
    assert abs(nu - nu_t) < 0.15


def test_grid_axes_and_surface_shape() -> None:
    cfg = _load_sabr_cfg()
    m, tau = grid_axes(cfg)
    params = np.array([0.2, -0.2, 0.35], dtype=float)
    m2, tau2, iv = implied_vol_surface_for_params(params, cfg)
    assert np.array_equal(m, m2) and np.array_equal(tau, tau2)
    assert iv.shape == (m.size, tau.size)
    assert np.all(np.isfinite(iv))


def test_lhs_params_shape() -> None:
    cfg = _load_sabr_cfg()
    p = lhs_sabr_params(cfg, n_samples=6, seed=0)
    assert p.shape == (6, 3)


@pytest.mark.slow
def test_lhs_surfaces_smoke() -> None:
    cfg = _load_sabr_cfg()
    _, m, tau, iv = implied_vol_surfaces_lhs(cfg, n_samples=2, n_batches=1, seed=1)
    assert iv.shape == (2, len(m), len(tau))
    assert np.all(np.isfinite(iv))


def test_calibrate_params_for_expiries_and_grid_map() -> None:
    spot, r, q = 100.0, 0.03, 0.0
    beta = 0.5
    taus = np.array([0.3, 0.9], dtype=float)
    rows = []
    strikes_sets = []
    iv_sets = []
    for T in taus:
        F = spot * np.exp((r - q) * float(T))
        ks = np.linspace(F * 0.85, F * 1.15, 10)
        alpha_t, rho_t, nu_t = 0.2 + 0.02 * float(T), -0.25, 0.4
        ivs = np.array(
            [sabr_hagan_lognormal_iv(F, float(k), float(T), alpha_t, beta, rho_t, nu_t) for k in ks],
            dtype=float,
        )
        strikes_sets.append(ks)
        iv_sets.append(ivs)
        rows.append([alpha_t, rho_t, nu_t])
    params_fit, _ = calibrate_params_for_expiries(
        spot, r, q, taus, strikes_sets, iv_sets, beta=beta
    )
    assert params_fit.shape == (2, 3)
    m = np.array([0.9, 1.0, 1.1], dtype=float)
    tau_grid = np.array([0.3, 0.6, 0.9], dtype=float)
    surf = implied_vol_surface_from_calibrated_slices(
        spot,
        r,
        q,
        m,
        tau_grid,
        taus,
        params_fit,
        beta=beta,
    )
    assert surf.shape == (3, 3)
    assert np.all(np.isfinite(surf))
