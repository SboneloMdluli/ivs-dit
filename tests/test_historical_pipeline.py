"""Tests for the full historic IV surface pipeline (smoothing -> interp/extrap -> unified grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from implied_volatility_diffusion.core.normalization import SurfaceNormalizer
from implied_volatility_diffusion.core.unified_grid import UnifiedGrid
from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import (
    build_historical_surface_pipeline,
    fill_surface_interp_extrap,
    save_historic_pipeline_report_pdf,
)


def _synthetic_day(seed: int = 0, n_strikes: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quote = pd.Timestamp("2024-05-15")
    rows: list[dict[str, float | object]] = []
    taus = [0.08, 0.25, 0.5, 1.0, 1.5]
    for tau in taus:
        for k in np.linspace(-0.3, 0.3, n_strikes):
            iv = 0.22 - 0.3 * k + 0.05 * k * k + 0.02 * np.log(1.0 + tau) + 0.005 * rng.standard_normal()
            rows.append(
                {
                    "quote_date": quote,
                    "expire_date": quote + pd.Timedelta(days=int(365 * tau)),
                    "underlying_last": 100.0,
                    "strike": float(100.0 * np.exp(k)),
                    "k": float(k),
                    "tau": float(tau),
                    "moneyness": float(np.exp(k)),
                    "iv": float(max(iv, 1e-4)),
                    "vega": 0.05,
                    "smooth_weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_fill_surface_interp_extrap_fills_every_cell() -> None:
    x = np.linspace(-0.5, 0.5, 9)
    tau = np.linspace(0.05, 2.0, 7)
    XI, TI = np.meshgrid(x, tau, indexing="ij")
    Z = 0.2 + 0.1 * XI + 0.05 * TI
    Z_hole = Z.copy()
    Z_hole[2:5, 2:5] = np.nan
    Z_hole[0, :] = np.nan
    Z_hole[-1, -2:] = np.nan

    filled = fill_surface_interp_extrap(Z_hole, x, tau)
    assert filled.shape == Z.shape
    assert np.all(np.isfinite(filled))
    interior = np.s_[2:5, 2:5]
    np.testing.assert_allclose(filled[interior], Z[interior], atol=5e-2)


def test_pipeline_produces_unified_grid_surface() -> None:
    day = _synthetic_day()
    grid = UnifiedGrid.default()

    stages = build_historical_surface_pipeline(
        day,
        unified_grid=grid,
        x_grid_smooth=np.linspace(-0.5, 0.5, 31),
        tau_grid_smooth=np.linspace(0.05, 2.0, 25),
        h1=0.02,
        h2=0.15,
    )

    assert stages.iv_unified.shape == grid.shape
    assert np.all(np.isfinite(stages.iv_unified))
    assert stages.iv_normalized is None
    assert stages.smoothed.shape == (31, 25)
    assert stages.filled.shape == (31, 25)
    assert np.all(np.isfinite(stages.filled))


def test_pipeline_normalizes_when_normalizer_provided() -> None:
    day = _synthetic_day()
    grid = UnifiedGrid.default()

    stack = np.stack(
        [
            np.full(grid.shape, 0.20),
            np.full(grid.shape, 0.25),
            np.full(grid.shape, 0.30),
        ],
        axis=0,
    )
    normalizer = SurfaceNormalizer(grid_shape=grid.shape).fit(stack)

    stages = build_historical_surface_pipeline(
        day,
        unified_grid=grid,
        x_grid_smooth=np.linspace(-0.5, 0.5, 21),
        tau_grid_smooth=np.linspace(0.05, 2.0, 21),
        h1=0.03,
        h2=0.2,
        normalizer=normalizer,
    )

    assert stages.iv_normalized is not None
    assert stages.iv_normalized.shape == grid.shape
    assert np.all(np.isfinite(stages.iv_normalized))


def test_pipeline_rejects_mismatched_normalizer_shape() -> None:
    day = _synthetic_day()
    grid = UnifiedGrid.default()
    bad = SurfaceNormalizer(grid_shape=(5, 5))
    bad.fit(np.random.default_rng(0).uniform(0.1, 0.3, size=(3, 5, 5)))

    with pytest.raises(ValueError, match="normalizer grid_shape"):
        build_historical_surface_pipeline(
            day,
            unified_grid=grid,
            x_grid_smooth=np.linspace(-0.5, 0.5, 21),
            tau_grid_smooth=np.linspace(0.05, 2.0, 21),
            normalizer=bad,
        )


def test_save_historic_pipeline_report_pdf_writes_file(tmp_path) -> None:
    day = _synthetic_day()
    grid = UnifiedGrid.default()
    stages = build_historical_surface_pipeline(
        day,
        unified_grid=grid,
        x_grid_smooth=np.linspace(-0.5, 0.5, 21),
        tau_grid_smooth=np.linspace(0.05, 2.0, 21),
        h1=0.03,
        h2=0.2,
    )
    out = tmp_path / "report.pdf"
    written = save_historic_pipeline_report_pdf(stages, out)
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0
