# SABR interpolation on market data

This note describes how to use the SABR (Hagan lognormal) implementation in this repo to **calibrate to observed implied-volatility smiles** and **interpolate** to a dense strike and maturity grid.

**Reference:** Hagan, Kumar, Lesniewski, Woodward, *Managing Smile Risk*, Wilmott Magazine (2002).

## What “interpolation” means here

1. For each market expiry, you have sparse quotes: strikes \(K\) and Black implied vols \(\sigma_{\mathrm{mkt}}(K,\tau)\).
2. You fix SABR \(\beta\) (common choices: `0`, `0.5`, `1`) and calibrate \((\alpha,\rho,\nu)\) so the SABR formula matches those quotes (least squares).
3. You evaluate the **same** SABR formula on a finer grid of moneyness \(K/S\) and maturity \(\tau\) to fill gaps between quotes. That is the interpolation step.

Implementation lives in:

- `src/implied_volatility_diffusion/models/sabr/hagan.py` — `sabr_hagan_lognormal_iv`
- `src/implied_volatility_diffusion/models/sabr/calibration.py` — `calibrate_sabr_to_implied_vols`, `calibrate_params_for_expiries`, `implied_vol_surface_from_calibrated_slices`

## Environment

From the repository root (see the project README for `uv` setup):

```bash
uv sync --group dev
```

Run Python with the project on `PYTHONPATH` or install editable (`uv sync` already links the package).

## Full surface: per-expiry calibration then dense grid

Assume one valuation date with:

- `spot`, continuous risk-free rate `r`, dividend yield `q`
- market expiries `tau_i` (years)
- for each expiry: 1D arrays of **strikes** and **Black implied vols** (same length)

```python
import numpy as np
from implied_volatility_diffusion.models.sabr import (
    calibrate_params_for_expiries,
    implied_vol_surface_from_calibrated_slices,
)

spot, r, q = 100.0, 0.03, 0.0
beta = 0.5  # fixed; calibrate (alpha, rho, nu) per expiry

expiry_taus = np.array([0.25, 0.5, 1.0], dtype=float)

strikes_per_expiry = [
    np.array([85.0, 92.5, 100.0, 107.5, 115.0], dtype=float),
    np.array([85.0, 92.5, 100.0, 107.5, 115.0], dtype=float),
    np.array([85.0, 92.5, 100.0, 107.5, 115.0], dtype=float),
]

ivs_per_expiry = [
    np.array([0.32, 0.29, 0.27, 0.28, 0.31], dtype=float),  # replace with real IVs
    np.array([0.31, 0.28, 0.26, 0.27, 0.30], dtype=float),
    np.array([0.30, 0.27, 0.25, 0.26, 0.29], dtype=float),
]

calibrated_params, details = calibrate_params_for_expiries(
    spot,
    r,
    q,
    expiry_taus,
    strikes_per_expiry,
    ivs_per_expiry,
    beta=beta,
)
# calibrated_params.shape == (n_expiries, 3)  -> (alpha, rho, nu) per row

moneyness = np.linspace(0.75, 1.25, 51)
tau_axis = np.linspace(0.05, 1.0, 40)

iv_surface = implied_vol_surface_from_calibrated_slices(
    spot,
    r,
    q,
    moneyness,
    tau_axis,
    expiry_taus,
    calibrated_params,
    beta=beta,
)
# iv_surface[i, j] is IV at moneyness[i], tau_axis[j]
```

`implied_vol_surface_from_calibrated_slices` assigns each grid maturity \(\tau\) the calibrated triple from the **nearest** market expiry in `expiry_taus`, then evaluates `sabr_hagan_lognormal_iv` on the grid. Adjust `expiry_taus` density or extend this mapping if you need smoother term-structure stitching between market pillars.

## Single expiry: calibrate one smile only

If you only need one tenor \(\tau\):

```python
import numpy as np
from implied_volatility_diffusion.models.sabr import (
    calibrate_sabr_to_implied_vols,
    sabr_hagan_lognormal_iv,
)

spot, r, q, tau = 100.0, 0.03, 0.0, 0.5
beta = 0.5
forward = spot * np.exp((r - q) * tau)

strikes = np.linspace(80.0, 120.0, 25)
# Replace with your observed IVs (same length as strikes).
market_ivs = 0.25 + 0.05 * np.abs(strikes / forward - 1.0)  # placeholder shape only

alpha, rho, nu, result = calibrate_sabr_to_implied_vols(
    forward, tau, strikes, market_ivs, beta=beta
)

dense_strikes = np.linspace(75.0, 125.0, 101)
iv_interp = np.array(
    [sabr_hagan_lognormal_iv(forward, float(k), tau, alpha, beta, rho, nu) for k in dense_strikes],
    dtype=float,
)
```

### Loading cleaned data

Read the processed parquet with **`load_cleaned_data`** from `historical_data_smoothing_interpolation.py` (expects at least `quote_date`, `k`, `tau`, `iv`, `vega`, `strike`, `underlying_last`, and related columns). Slice one valuation day with `df[df["quote_date"] == sample_date]`.

### SABR interpolation on that day

For one `quote_date`, **`historical_sabr_interpolation.py`** applies the same SABR workflow as above: **`filter_day_for_surface`**, group quotes by **`expire_date`** (or rounded **`tau`** if expiry is missing), **`calibrate_params_for_expiries`**, then **`implied_vol_surface_from_calibrated_slices`** on your chosen dense **`k_grid` × `tau_grid`** (with `k = log(K/S)` and `m = exp(k)` internally). The evaluation grid is **independent** of any kernel smoother; pick resolution and ranges for the SABR surface you want.

```python
import numpy as np
import pandas as pd

from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import load_cleaned_data
from implied_volatility_diffusion.data.historical_sabr_interpolation import build_historical_sabr_surface

df = load_cleaned_data("data/processed/processed.parquet")
day_df = df[df["quote_date"] == pd.Timestamp("2021-12-14")].copy()

k_grid = np.linspace(-0.2, 0.2, 41)
tau_grid = np.linspace(0.05, 1.0, 24)

sabr_out = build_historical_sabr_surface(day_df, k_grid, tau_grid, r=0.03, q=0.0, beta=0.5)
Z_sabr = sabr_out.surface
```

### Optional: compare to a kernel surface

If you also build a kernel surface on the **same** `(k_grid, tau_grid)` (see `build_kernel_surface` in `historical_data_smoothing_interpolation.py`), you can summarize overlap with **`compare_kernel_sabr_surfaces(Z_kernel, Z_sabr)`** (`rmse`, finite counts). Different grids per method are fine if you only need side-by-side plots—interpolate or align off-line before a scalar RMSE.

## Practical notes

- **Quote quality:** remove bad prints, align call/put conventions, and ensure strikes are positive before calibration.
- **\(\beta\):** often fixed (not jointly fit) for stability on equity-style surfaces.
- **Term structure:** if the notebook demo looks “flat” in \(\tau\), it may be using **one** global \((\alpha,\rho,\nu)\) for all maturities; per-expiry calibration (above) restores maturity variation.
