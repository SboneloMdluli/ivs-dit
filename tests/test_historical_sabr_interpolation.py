"""Tests for historic SABR surface build (kernel-comparable grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import (
    build_kernel_surface,
)
from implied_volatility_diffusion.data.historical_sabr_interpolation import (
    build_historical_sabr_surface,
    compare_kernel_sabr_surfaces,
    rmse_masked,
)
from implied_volatility_diffusion.synthetic_ivs_generator.sabr import sabr_lognormal_iv
from implied_volatility_diffusion.synthetic_ivs_generator.sabr_iv_surface import (
    forward_from_spot,
)


def _synthetic_day_two_expiries() -> pd.DataFrame:
    spot = 100.0
    r, q, beta = 0.0, 0.0, 0.5
    alpha_t, rho_t, nu_t = 0.22, -0.25, 0.45
    rows: list[dict[str, float | object]] = []
    quote = pd.Timestamp("2024-01-02")
    for tau, exp in [(0.3, pd.Timestamp("2024-04-01")), (0.6, pd.Timestamp("2024-07-01"))]:
        fwd = forward_from_spot(spot, tau, r, q)
        for k in np.linspace(-0.12, 0.12, 8):
            strike = float(spot * np.exp(k))
            iv = float(sabr_lognormal_iv(fwd, strike, tau, alpha_t, beta, rho_t, nu_t))
            rows.append(
                {
                    "quote_date": quote,
                    "expire_date": exp,
                    "underlying_last": spot,
                    "strike": strike,
                    "k": float(k),
                    "tau": float(tau),
                    "iv": iv,
                    "vega": 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_build_historical_sabr_surface_shape_and_finite() -> None:
    day = _synthetic_day_two_expiries()
    k_grid = np.linspace(-0.15, 0.15, 11)
    tau_grid = np.array([0.3, 0.6], dtype=float)
    out = build_historical_sabr_surface(day, k_grid, tau_grid, r=0.0, q=0.0, beta=0.5)
    assert out.surface.shape == (len(k_grid), len(tau_grid))
    assert np.all(np.isfinite(out.surface))
    assert out.calibrated_params.shape == (2, 3)


def test_compare_kernel_sabr_identical_zero_rmse() -> None:
    z = np.ones((3, 4), dtype=float)
    stats = compare_kernel_sabr_surfaces(z, z.copy())
    assert stats["rmse"] == 0.0
    assert stats["n_overlap"] == 12


def test_rmse_masked_ignores_nan() -> None:
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    b = np.array([[1.1, 2.0], [2.9, 4.0]])
    rmse, n = rmse_masked(a, b)
    assert n == 3
    assert rmse > 0.0


def test_kernel_vs_sabr_same_grid_small_rmse() -> None:
    """Synthetic SABR quotes: kernel smooths noise-free data; SABR should match structure."""
    day = _synthetic_day_two_expiries()
    k_grid = np.linspace(-0.12, 0.12, 9)
    tau_grid = np.array([0.3, 0.6], dtype=float)
    sabr = build_historical_sabr_surface(day, k_grid, tau_grid, r=0.0, q=0.0, beta=0.5)
    kern, _ = build_kernel_surface(
        day,
        x_grid=k_grid,
        tau_grid=tau_grid,
        h1=0.02,
        h2=0.15,
        clip_upper=1.0,
    )
    stats = compare_kernel_sabr_surfaces(kern, sabr.surface)
    assert stats["n_overlap"] > 0
    assert float(stats["rmse"]) < 0.08
