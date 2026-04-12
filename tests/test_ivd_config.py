"""Tests for the ``ivd_config`` package."""

from __future__ import annotations

from pathlib import Path

import pytest

from ivd_config import load_config, merge_config


def test_merge_config_deep() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "config" / "heston_iv_surface.yaml")
    merged = merge_config(
        base,
        {"lhs": {"n_samples": 7}, "market": {"spot": 200.0}},
    )
    assert merged["lhs"]["n_samples"] == 7
    assert merged["lhs"]["seed"] == base["lhs"]["seed"]
    assert merged["market"]["spot"] == 200.0
    assert merged["market"]["dividend_yield"] == base["market"]["dividend_yield"]
    assert base["lhs"]["n_samples"] != 7


def test_load_config_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(p)
