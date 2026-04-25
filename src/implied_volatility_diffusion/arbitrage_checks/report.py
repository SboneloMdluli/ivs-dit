"""Public dataclass describing a single surface's no-arbitrage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArbitrageReport:
    """Per-surface arbitrage diagnostics. See :mod:`arbitrage_checks.checks`."""

    butterfly_ok: bool
    calendar_ok: bool
    bounds_ok: bool
    arbitrage_free: bool
    worst_butterfly: float
    worst_monotonicity: float
    worst_calendar: float
    worst_bound: float
    n_butterfly_violations: int
    n_calendar_violations: int
    n_bound_violations: int
