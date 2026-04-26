"""Tests for the unified ``(k, tau)`` grid and its resampling helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from implied_volatility_diffusion import UnifiedGrid, resample_batch_to_unified_grid, resample_to_unified_grid
from implied_volatility_diffusion.core.grid import grid_axes

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_grid_bounds_and_shape() -> None:
    grid = UnifiedGrid.default()
    assert grid.log_moneyness[0] == pytest.approx(-0.5)
    assert grid.log_moneyness[-1] == pytest.approx(0.5)
    assert grid.tau[0] == pytest.approx(0.05)
    assert grid.tau[-1] == pytest.approx(2.0)
    assert grid.shape == (41, 40)
    assert np.allclose(grid.moneyness, np.exp(grid.log_moneyness))


def test_default_grid_matches_yaml() -> None:
    grid_default = UnifiedGrid.default()
    grid_yaml = UnifiedGrid.load(REPO_ROOT / "config" / "unified_iv_grid.yaml")
    assert np.allclose(grid_default.log_moneyness, grid_yaml.log_moneyness)
    assert np.allclose(grid_default.tau, grid_yaml.tau)


def test_with_config_drives_grid_axes() -> None:
    grid = UnifiedGrid.default()
    cfg = grid.with_config({"market": {"spot": 100.0}})
    m_axis, tau_axis = grid_axes(cfg)
    assert np.allclose(m_axis, grid.moneyness)
    assert np.allclose(tau_axis, grid.tau)


def test_from_config_legacy_moneyness_roundtrip() -> None:
    legacy_cfg = {
        "grid": {
            "moneyness": {"start_point": 0.6, "step": 0.1, "end_point": 1.4},
            "tau": {"start_point": 0.1, "step": 0.1, "end_point": 0.5},
        }
    }
    grid = UnifiedGrid.from_config(legacy_cfg)
    assert np.allclose(grid.moneyness, np.arange(0.6, 1.4 + 1e-9, 0.1))
    assert np.allclose(grid.tau, np.arange(0.1, 0.5 + 1e-9, 0.1))


def test_resample_preserves_linear_surface() -> None:
    grid = UnifiedGrid.default()
    k_src = np.linspace(-0.6, 0.6, 25)
    tau_src = np.linspace(0.05, 2.0, 20)
    K, T = np.meshgrid(k_src, tau_src, indexing="ij")
    linear = 0.2 + 0.3 * K + 0.1 * T

    out = resample_to_unified_grid(linear, k_src=k_src, tau_src=tau_src, grid=grid)
    K2, T2 = grid.meshgrid()
    expected = 0.2 + 0.3 * K2 + 0.1 * T2
    assert np.allclose(out, expected, atol=1e-12)


def test_resample_masks_out_of_range() -> None:
    grid = UnifiedGrid.default()
    k_src = np.linspace(-0.1, 0.1, 5)
    tau_src = np.linspace(0.5, 1.0, 5)
    surface = np.ones((k_src.size, tau_src.size))
    out = resample_to_unified_grid(surface, k_src=k_src, tau_src=tau_src, grid=grid)
    # pixels outside the source rectangle must be NaN
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[20, 15])  # k=0 in-range, tau in-range


def test_resample_batch_preserves_leading_dims() -> None:
    grid = UnifiedGrid.default()
    k_src = np.linspace(-0.5, 0.5, 11)
    tau_src = np.linspace(0.05, 2.0, 10)
    batch = np.random.default_rng(0).normal(size=(3, 4, k_src.size, tau_src.size))
    out = resample_batch_to_unified_grid(batch, k_src=k_src, tau_src=tau_src, grid=grid)
    assert out.shape == (3, 4, *grid.shape)


def test_moneyness_source_supported() -> None:
    grid = UnifiedGrid.default()
    m_src = np.linspace(0.7, 1.3, 13)
    tau_src = np.linspace(0.1, 1.5, 12)
    surface = np.full((m_src.size, tau_src.size), 0.2)
    out = resample_to_unified_grid(surface, m_src=m_src, tau_src=tau_src, grid=grid)
    inside = (grid.log_moneyness >= np.log(0.7)) & (grid.log_moneyness <= np.log(1.3))
    inside_tau = (grid.tau >= 0.1) & (grid.tau <= 1.5)
    mask = np.outer(inside, inside_tau)
    assert np.all(np.isclose(out[mask], 0.2))
