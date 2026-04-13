"""Tests for the ``ivs_config`` package."""

from __future__ import annotations

from pathlib import Path

import pytest

from implied_volatility_diffusion.synthetic_ivs_generator.heston_iv_surface import (
    load_heston_iv_surface_config,
)
from ivs_config import load_config, merge_config, merge_config_files


def test_merge_config_deep() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_heston_iv_surface_config(root / "config")
    merged = merge_config(
        base,
        {"lhs": {"n_samples": 7}, "market": {"spot": 200.0}},
    )
    assert merged["lhs"]["n_samples"] == 7
    assert merged["lhs"]["seed"] == base["lhs"]["seed"]
    assert merged["market"]["spot"] == 200.0
    assert merged["market"]["dividend_yield"] == base["market"]["dividend_yield"]
    assert base["lhs"]["n_samples"] != 7


def test_merge_config_files_order() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "config"
    merged = merge_config_files(cfg_dir / "heston_iv_surface.yaml", cfg_dir / "iv_surface_grid.yaml")
    assert "heston_ranges" in merged and "grid" in merged and "plot_surface" in merged
    assert merged["grid"]["moneyness"]["start_point"] == 0.6


def test_load_config_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(p)
