"""SABR calibration helpers for historical data surfaces."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import (
    filter_day_for_surface,
)
from implied_volatility_diffusion.models.sabr.calibration import (
    calibrate_params_for_expiries,
    implied_vol_surface_from_calibrated_slices,
)


@dataclass(frozen=True)
class HistoricalSABRResult:
    """Output container for historical SABR surface calibration."""

    surface: np.ndarray
    day_sub: pd.DataFrame
    expiry_taus: np.ndarray
    calibrated_params: np.ndarray
    spot: float
    calibration_details: list[dict[str, object]]


def _spot_from_day(day_sub: pd.DataFrame, spot_col: str) -> float:
    if spot_col not in day_sub.columns:
        raise ValueError(f"Missing spot column {spot_col!r} for SABR forward.")
    s = float(np.nanmedian(day_sub[spot_col].to_numpy(dtype=float)))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"Invalid spot median from {spot_col!r}.")
    return s


def _expiry_group_key(df: pd.DataFrame) -> pd.Series:
    if "expire_date" in df.columns and df["expire_date"].notna().any():
        return df["expire_date"]
    return np.round(df["tau"].to_numpy(dtype=float), 8)


def _build_smiles_by_expiry(
    day_sub: pd.DataFrame,
    *,
    min_points: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return ``strikes_per_expiry``, ``ivs_per_expiry``, ``expiry_taus`` sorted by tau."""
    gkey = _expiry_group_key(day_sub)
    strikes_per: list[np.ndarray] = []
    ivs_per: list[np.ndarray] = []
    taus: list[float] = []

    for _, grp in day_sub.groupby(gkey, sort=False):
        if len(grp) < min_points:
            continue
        tau_med = float(np.nanmedian(grp["tau"].to_numpy(dtype=float)))
        if not np.isfinite(tau_med) or tau_med <= 0.0:
            continue
        sub = grp.sort_values("strike")
        ks = sub["strike"].to_numpy(dtype=float)
        iv = sub["iv"].to_numpy(dtype=float)
        uniq = pd.DataFrame({"strike": ks, "iv": iv}).groupby("strike", as_index=False).mean().sort_values("strike")
        ks_u = uniq["strike"].to_numpy(dtype=float)
        iv_u = uniq["iv"].to_numpy(dtype=float)
        if ks_u.size < min_points:
            continue
        strikes_per.append(ks_u)
        ivs_per.append(iv_u)
        taus.append(tau_med)

    if not taus:
        raise ValueError(
            "No expiry slice with enough points for SABR calibration "
            f"(need at least {min_points} distinct strikes per expiry)."
        )

    order = np.argsort(np.asarray(taus, dtype=float))
    taus_arr = np.asarray(taus, dtype=float)[order]
    strikes_ordered = [strikes_per[i] for i in order]
    ivs_ordered = [ivs_per[i] for i in order]
    return strikes_ordered, ivs_ordered, taus_arr


def prepare_historical_sabr_smiles(
    day_df: pd.DataFrame,
    *,
    spot_col: str = "underlying_last",
    min_strikes_per_expiry: int = 3,
    filter_kw: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, float, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Filter one day and build per-expiry strike/IV arrays."""
    kw = dict(filter_kw or {})
    day_sub = filter_day_for_surface(day_df, **kw)
    spot = _spot_from_day(day_sub, spot_col)
    strikes_per, ivs_per, expiry_taus = _build_smiles_by_expiry(
        day_sub,
        min_points=min_strikes_per_expiry,
    )
    return day_sub, spot, strikes_per, ivs_per, expiry_taus


def build_historical_sabr_surface(
    day_df: pd.DataFrame,
    k_grid: np.ndarray,
    tau_grid: np.ndarray,
    *,
    r: float = 0.0,
    q: float = 0.0,
    beta: float = 0.5,
    spot_col: str = "underlying_last",
    min_strikes_per_expiry: int = 3,
    filter_kw: dict[str, Any] | None = None,
) -> HistoricalSABRResult:
    """Calibrate SABR per market expiry and fill IV on ``(k_grid, tau_grid)``."""
    kw = dict(filter_kw or {})
    day_sub, spot, strikes_per, ivs_per, expiry_taus = prepare_historical_sabr_smiles(
        day_df,
        spot_col=spot_col,
        min_strikes_per_expiry=min_strikes_per_expiry,
        filter_kw=kw,
    )

    calibrated, details = calibrate_params_for_expiries(
        spot,
        r,
        q,
        expiry_taus,
        strikes_per,
        ivs_per,
        beta=beta,
    )

    m_grid = np.exp(np.asarray(k_grid, dtype=float).ravel())
    taus_out = np.asarray(tau_grid, dtype=float).ravel()

    surface = implied_vol_surface_from_calibrated_slices(
        spot,
        r,
        q,
        m_grid,
        taus_out,
        expiry_taus,
        calibrated,
        beta=beta,
    )

    return HistoricalSABRResult(
        surface=surface,
        day_sub=day_sub,
        expiry_taus=expiry_taus,
        calibrated_params=calibrated,
        spot=spot,
        calibration_details=details,
    )


def rmse_masked(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Root mean square error where both ``a`` and ``b`` are finite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    m = np.isfinite(a) & np.isfinite(b)
    n = int(np.count_nonzero(m))
    if n == 0:
        return float("nan"), 0
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d))), n


def compare_kernel_sabr_surfaces(
    Z_kernel: np.ndarray,
    Z_sabr: np.ndarray,
) -> dict[str, float | int]:
    """Summary stats for kernel vs SABR on the same grid."""
    rmse, n = rmse_masked(Z_kernel, Z_sabr)
    return {
        "rmse": rmse,
        "n_overlap": n,
        "n_kernel_finite": int(np.count_nonzero(np.isfinite(Z_kernel))),
        "n_sabr_finite": int(np.count_nonzero(np.isfinite(Z_sabr))),
    }
