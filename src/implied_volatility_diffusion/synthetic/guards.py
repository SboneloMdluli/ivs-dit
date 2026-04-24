"""Arbitrage guards for synthetic IV surface generation.

This module wraps :func:`build_surfaces` with a policy that enforces the
Roper / Gatheral no-arbitrage conditions already diagnosed by
:mod:`implied_volatility_diffusion.arbitrage`:

* **butterfly** (strike-wise convex, non-increasing undiscounted call),
* **calendar** (total variance ``w = sigma^2 * tau`` non-decreasing in tau),
* **bounds** (BS call prices inside the model-free interval).

Policies (selected per recipe or via ``cfg['arbitrage_guard']``):

* ``"none"``     - no check, legacy behaviour.
* ``"warn"``     - run checks, emit a :mod:`warnings` message on violations.
* ``"repair"``   - project each surface onto the calendar-monotone cone,
                    then re-check; emit a warning if residual violations remain.
* ``"filter"``   - drop surfaces that fail the checks (with optional repair
                    pass first); raises if the filtered batch is empty.
* ``"raise"``    - raise :class:`ArbitrageError` on any violation.

The calendar repair is a local per-moneyness cumulative-max projection of
``w`` along the tau axis. It is idempotent and a minimum-perturbation fix:
it never decreases any IV, and it preserves every already-monotone slice
exactly. Butterfly violations are diagnosed but not auto-repaired (that
requires a more elaborate SVI/SSVI projection out of scope here).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from implied_volatility_diffusion.arbitrage import (
    ArbitrageReport,
    check_iv_surface_arbitrage,
    check_iv_surfaces_arbitrage,
)
from implied_volatility_diffusion.core.protocols import VolModel
from implied_volatility_diffusion.core.types import SurfaceBatch
from implied_volatility_diffusion.synthetic.surface import build_surfaces

GuardPolicy = Literal["none", "warn", "repair", "filter", "raise"]
_ALLOWED_POLICIES: tuple[GuardPolicy, ...] = ("none", "warn", "repair", "filter", "raise")


class ArbitrageError(ValueError):
    """Raised when a surface batch violates no-arbitrage under the ``"raise"`` policy."""


@dataclass(frozen=True)
class GuardSettings:
    """Arbitrage-guard knobs read from config.

    Attributes:
        policy: One of ``"none" | "warn" | "repair" | "filter" | "raise"``.
        tol: Numerical tolerance used by the arbitrage checker.
        repair_before_filter: If ``True``, apply calendar repair before the
            filter policy re-checks surfaces (default ``True``).
    """

    policy: GuardPolicy = "repair"
    tol: float = 1e-8
    repair_before_filter: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GuardSettings":
        section = cfg.get("arbitrage_guard") or {}
        policy_raw = str(section.get("policy", "repair")).lower()
        if policy_raw not in _ALLOWED_POLICIES:
            raise ValueError(f"arbitrage_guard.policy must be one of {_ALLOWED_POLICIES}; got {policy_raw!r}")
        return cls(
            policy=policy_raw,  # type: ignore[arg-type]
            tol=float(section.get("tol", 1e-8)),
            repair_before_filter=bool(section.get("repair_before_filter", True)),
        )


# ---------------------------------------------------------------------------
# repair primitives
# ---------------------------------------------------------------------------


def repair_calendar_monotone(iv: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Project an IV surface onto the calendar-monotone cone.

    Enforces that the total implied variance ``w(m, tau_i) = iv^2 * tau_i``
    is non-decreasing in ``tau_i`` for every fixed moneyness. Accepts an
    array with a trailing ``(..., n_moneyness, n_tau)`` layout; the
    projection is applied along the last axis.

    Non-finite / non-positive IVs are left untouched in place (they are
    surfaced as-is and counted by the arbitrage checker).

    Args:
        iv: Implied-vol tensor, shape ``(..., n_moneyness, n_tau)``.
        tau: Strictly increasing year-fraction grid, shape ``(n_tau,)``.

    Returns:
        Corrected IV tensor with the same shape as ``iv``. Never mutates ``iv``.
    """
    iv_arr = np.asarray(iv, dtype=float)
    t = np.asarray(tau, dtype=float).reshape(-1)
    if iv_arr.shape[-1] != t.size:
        raise ValueError(f"iv last axis ({iv_arr.shape[-1]}) must match len(tau)={t.size}")
    if t.size < 2:
        return iv_arr.copy()

    w = (iv_arr**2) * t  # broadcast along last axis
    valid = np.isfinite(w) & (iv_arr > 0.0)
    w_fill = np.where(valid, w, -np.inf)
    w_mono = np.maximum.accumulate(w_fill, axis=-1)
    w_mono = np.where(np.isfinite(w_mono), w_mono, w)

    with np.errstate(invalid="ignore", divide="ignore"):
        iv_new = np.sqrt(np.maximum(w_mono, 0.0) / t)
    # Preserve positions that were not finite/positive to begin with.
    return np.where(valid, iv_new, iv_arr)


# ---------------------------------------------------------------------------
# guard orchestration
# ---------------------------------------------------------------------------


def _summarise(reports: Iterable[ArbitrageReport]) -> tuple[int, int, int, int]:
    rs = list(reports)
    n_total = len(rs)
    n_bfly = sum(int(not r.butterfly_ok) for r in rs)
    n_cal = sum(int(not r.calendar_ok) for r in rs)
    n_bnd = sum(int(not r.bounds_ok) for r in rs)
    return n_total, n_bfly, n_cal, n_bnd


def _raise_or_warn(
    reports: list[ArbitrageReport],
    *,
    policy: GuardPolicy,
    stage: str,
) -> None:
    n_total, n_bfly, n_cal, n_bnd = _summarise(reports)
    if n_bfly == 0 and n_cal == 0 and n_bnd == 0:
        return
    msg = (
        f"arbitrage guard ({stage}): {n_bfly}/{n_total} butterfly, "
        f"{n_cal}/{n_total} calendar, {n_bnd}/{n_total} bound violations"
    )
    if policy == "raise":
        raise ArbitrageError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def enforce_arbitrage(
    batch: SurfaceBatch,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    settings: GuardSettings,
) -> SurfaceBatch:
    """Apply the guard ``settings.policy`` to ``batch`` and return the result.

    The returned :class:`SurfaceBatch` may be shorter than the input in the
    ``"filter"`` policy (arbitrage-violating rows dropped).
    """
    if settings.policy == "none":
        return batch

    iv = batch.iv
    if iv.ndim < 3:
        raise ValueError(f"enforce_arbitrage expects iv with shape (..., M, T); got {iv.shape}")

    if settings.policy in ("repair", "filter") and settings.repair_before_filter:
        iv_rep = repair_calendar_monotone(iv, batch.tau)
        batch = replace(batch, iv=iv_rep)

    reports = check_iv_surfaces_arbitrage(
        batch.iv,
        batch.moneyness,
        batch.tau,
        spot=float(spot),
        rate=float(rate),
        dividend_yield=float(dividend_yield),
        tol=float(settings.tol),
    )

    if settings.policy == "raise":
        _raise_or_warn(reports, policy="raise", stage="post-check")
        return batch
    if settings.policy == "warn":
        _raise_or_warn(reports, policy="warn", stage="post-check")
        return batch
    if settings.policy == "repair":
        _raise_or_warn(reports, policy="warn", stage="post-repair")
        return batch

    # filter: drop rows that still violate anything.
    keep_mask = np.array([r.arbitrage_free for r in reports], dtype=bool)
    leading = batch.iv.shape[:-2]
    keep_mask = keep_mask.reshape(leading) if leading else keep_mask
    if not bool(np.any(keep_mask)):
        raise ArbitrageError("arbitrage guard (filter): zero surfaces passed all checks")
    if bool(np.all(keep_mask)):
        return batch

    # Only support filtering along the single leading axis (the LHS-sample axis).
    if len(leading) != 1:
        raise ValueError(
            f"filter policy only supports iv tensors with a single leading sample axis; got leading shape {leading}"
        )
    idx = np.flatnonzero(keep_mask)
    n_dropped = int(batch.iv.shape[0] - idx.size)
    if n_dropped:
        warnings.warn(
            f"arbitrage guard (filter): dropped {n_dropped} surfaces",
            RuntimeWarning,
            stacklevel=3,
        )
    return SurfaceBatch(
        params=batch.params[idx],
        moneyness=batch.moneyness,
        tau=batch.tau,
        iv=batch.iv[idx],
    )


def guarded_build_surfaces(
    model: VolModel,
    cfg: Mapping[str, Any],
    params: np.ndarray,
    *,
    spot_override: float | None = None,
    inst_var_override: np.ndarray | None = None,
    guard: GuardSettings | None = None,
) -> SurfaceBatch:
    """:func:`build_surfaces` with an arbitrage guard applied afterwards.

    If ``guard`` is ``None`` the guard is loaded from ``cfg['arbitrage_guard']``
    (defaults to ``policy="repair"``).
    """
    batch = build_surfaces(
        model,
        cfg,
        params,
        spot_override=spot_override,
        inst_var_override=inst_var_override,
    )
    settings = guard if guard is not None else GuardSettings.from_config(cfg)
    if settings.policy == "none":
        return batch
    market = cfg["market"]
    spot = float(spot_override if spot_override is not None else market["spot"])
    rate = float(market.get("r", market.get("rate", market.get("risk_free_rate", 0.0))))
    q = float(market.get("dividend_yield", 0.0))
    return enforce_arbitrage(
        batch,
        spot=spot,
        rate=rate,
        dividend_yield=q,
        settings=settings,
    )


def assert_arbitrage_free(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
) -> ArbitrageReport:
    """Strict guard usable from notebooks / tests: raise on any violation."""
    report = check_iv_surface_arbitrage(
        iv,
        moneyness,
        tau,
        spot=float(spot),
        rate=float(rate),
        dividend_yield=float(dividend_yield),
        tol=float(tol),
    )
    if not report.arbitrage_free:
        raise ArbitrageError(
            "surface is not arbitrage-free: "
            f"butterfly_ok={report.butterfly_ok}, calendar_ok={report.calendar_ok}, "
            f"bounds_ok={report.bounds_ok}, worst_butterfly={report.worst_butterfly:.3e}, "
            f"worst_calendar={report.worst_calendar:.3e}, worst_bound={report.worst_bound:.3e}"
        )
    return report


__all__ = [
    "ArbitrageError",
    "GuardPolicy",
    "GuardSettings",
    "assert_arbitrage_free",
    "enforce_arbitrage",
    "guarded_build_surfaces",
    "repair_calendar_monotone",
]
