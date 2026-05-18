"""ATM skew and curvature along the log-moneyness axis."""

from __future__ import annotations

import numpy as np

from implied_volatility_diffusion.core.unified_grid import UnifiedGrid


def _atm_neighbor_index(log_k: np.ndarray, *, atm_log_moneyness: float) -> int:
    k = np.asarray(log_k, dtype=float).ravel()
    i = int(np.argmin(np.abs(k - float(atm_log_moneyness))))
    if i <= 0 or i >= k.size - 1:
        raise ValueError(
            "ATM target log-moneyness must lie strictly inside the grid "
            f"(need neighbors); got index {i} for size {k.size}."
        )
    return i


def atm_skew_and_curvature(
    iv: np.ndarray,
    grid: UnifiedGrid,
    *,
    atm_log_moneyness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference ATM skew and curvature along log-moneyness, per tenor."""
    arr = np.asarray(iv, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"iv shape {arr.shape} must match grid.shape {grid.shape}")

    k = grid.log_moneyness
    i0 = _atm_neighbor_index(k, atm_log_moneyness=atm_log_moneyness)
    km, k0, kp = float(k[i0 - 1]), float(k[i0]), float(k[i0 + 1])

    sigma_m = arr[i0 - 1, :]
    sigma_0 = arr[i0, :]
    sigma_p = arr[i0 + 1, :]

    skew = (sigma_p - sigma_m) / (kp - km)
    dk_left = k0 - km
    dk_right = kp - k0
    # Three-point second derivative at k0 for possibly non-uniform spacing.
    curvature = (
        2.0
        * (dk_right * (sigma_m - sigma_0) + dk_left * (sigma_p - sigma_0))
        / (dk_left * dk_right * (dk_left + dk_right))
    )

    return grid.tau.copy(), skew, curvature


def atm_skew_and_curvature_batch(
    iv_batch: np.ndarray,
    grid: UnifiedGrid,
    *,
    atm_log_moneyness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of ATM skew/curvature over a batch of surfaces."""
    batch = np.asarray(iv_batch, dtype=float)
    if batch.shape[-2:] != grid.shape:
        raise ValueError(f"trailing iv_batch axes {batch.shape[-2:]} must match grid.shape {grid.shape}")
    flat = batch.reshape(-1, *grid.shape)
    skew_stack = np.empty((flat.shape[0], grid.tau.size), dtype=float)
    curv_stack = np.empty_like(skew_stack)
    tau_ref = None
    for n in range(flat.shape[0]):
        tau, sk, cu = atm_skew_and_curvature(flat[n], grid, atm_log_moneyness=atm_log_moneyness)
        tau_ref = tau
        skew_stack[n] = sk
        curv_stack[n] = cu
    assert tau_ref is not None
    return (
        tau_ref,
        skew_stack.mean(axis=0),
        skew_stack.std(axis=0, ddof=1) if skew_stack.shape[0] > 1 else np.zeros_like(skew_stack[0]),
        curv_stack.mean(axis=0),
        curv_stack.std(axis=0, ddof=1) if curv_stack.shape[0] > 1 else np.zeros_like(curv_stack[0]),
    )


__all__ = [
    "atm_skew_and_curvature",
    "atm_skew_and_curvature_batch",
]
