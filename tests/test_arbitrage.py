"""Tests for the IV-surface no-arbitrage diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from implied_volatility_diffusion import (
    check_iv_surface_arbitrage,
    check_iv_surfaces_arbitrage,
    grid_axes,
    implied_vol_surface_for_params,
    implied_vol_surfaces_lhs,
    load_heston_iv_surface_config,
)


def _flat_bs_surface(moneyness: np.ndarray, tau: np.ndarray, sigma: float) -> np.ndarray:
    """Constant Black-Scholes IV surface: trivially arbitrage-free."""
    return np.full((moneyness.size, tau.size), float(sigma), dtype=float)


def test_flat_bs_surface_is_arbitrage_free() -> None:
    m = np.linspace(0.5, 2.5, 21)
    tau = np.linspace(0.05, 2.0, 12)
    iv = _flat_bs_surface(m, tau, 0.25)
    rep = check_iv_surface_arbitrage(iv, m, tau, spot=100.0, rate=0.02, dividend_yield=0.0)
    assert rep.arbitrage_free
    assert rep.butterfly_ok
    assert rep.calendar_ok
    assert rep.bounds_ok
    assert rep.n_butterfly_violations == 0
    assert rep.n_calendar_violations == 0
    assert rep.n_bound_violations == 0


def test_calendar_violation_detected_when_total_variance_decreases() -> None:
    m = np.linspace(0.5, 2.5, 21)
    tau = np.array([0.25, 0.5, 1.0])
    iv = np.empty((m.size, tau.size), dtype=float)
    iv[:, 0] = 0.40
    iv[:, 1] = 0.20
    iv[:, 2] = 0.20
    rep = check_iv_surface_arbitrage(iv, m, tau, spot=100.0, rate=0.0)
    w = (iv ** 2) * tau[None, :]
    assert np.any(np.diff(w, axis=1) < 0.0)
    assert not rep.calendar_ok
    assert rep.worst_calendar < 0.0
    assert rep.n_calendar_violations > 0


def test_butterfly_violation_detected_for_concave_smile() -> None:
    m = np.linspace(0.7, 1.3, 13)
    tau = np.array([0.5])
    iv = (0.40 - 0.30 * np.exp(-((m - 1.0) ** 2) / 0.01))[:, None]
    rep = check_iv_surface_arbitrage(iv, m, tau, spot=100.0, rate=0.0)
    assert not rep.butterfly_ok
    assert rep.worst_butterfly < 0.0
    assert rep.n_butterfly_violations > 0


def test_batched_check_returns_one_report_per_surface() -> None:
    m = np.linspace(0.5, 2.5, 11)
    tau = np.linspace(0.1, 1.0, 6)
    surfaces = np.stack(
        [
            _flat_bs_surface(m, tau, 0.20),
            _flat_bs_surface(m, tau, 0.30),
            _flat_bs_surface(m, tau, 0.40),
        ],
        axis=0,
    )
    reports = check_iv_surfaces_arbitrage(surfaces, m, tau, spot=100.0, rate=0.01)
    assert len(reports) == surfaces.shape[0]
    assert all(r.arbitrage_free for r in reports)


def test_batched_check_handles_extra_leading_axes() -> None:
    m = np.linspace(0.5, 2.5, 9)
    tau = np.linspace(0.1, 1.0, 5)
    surfaces = np.broadcast_to(_flat_bs_surface(m, tau, 0.25), (2, 3, m.size, tau.size)).copy()
    reports = check_iv_surfaces_arbitrage(surfaces, m, tau, spot=100.0, rate=0.0)
    assert len(reports) == 2 * 3


@pytest.mark.slow
def test_heston_lhs_surfaces_are_arbitrage_free() -> None:
    """End-to-end: Heston-COS surfaces from the default config should pass both checks."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    params, m, tau, iv = implied_vol_surfaces_lhs(cfg, n_samples=4, n_batches=1, seed=2024)
    market = cfg["market"]
    spot = float(market["spot"])
    q = float(market.get("dividend_yield", 0.0))
    r = float(params[0, -1])
    reports = check_iv_surfaces_arbitrage(iv, m, tau, spot=spot, rate=r, dividend_yield=q, tol=1e-6)
    assert all(r.calendar_ok for r in reports)
    assert all(r.butterfly_ok for r in reports)
    assert all(r.bounds_ok for r in reports)


@pytest.mark.slow
def test_single_heston_surface_passes_arbitrage() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_heston_iv_surface_config(root / "config")
    m, tau = grid_axes(cfg)
    params = np.array([0.04, -0.5, 0.30, 0.04, 2.0, 0.02], dtype=float)
    _, _, iv = implied_vol_surface_for_params(params, cfg)
    rep = check_iv_surface_arbitrage(
        iv,
        m,
        tau,
        spot=float(cfg["market"]["spot"]),
        rate=float(params[-1]),
        dividend_yield=float(cfg["market"].get("dividend_yield", 0.0)),
        tol=1e-6,
    )
    assert rep.arbitrage_free, rep
