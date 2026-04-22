# SABR README

This document centralizes SABR-related guidance in this project, combining:

- the high-level SABR baseline summary from the root `README.md`
- the implementation and usage details from `docs/sabr_interpolation.md`

## Purpose

SABR is the classical baseline used to generate and calibrate Black implied-volatility
surfaces before downstream diffusion/transformer modeling.

In this repo, the baseline is based on the SABR lognormal approximation
(Hagan 2002) and supports:

- single-smile calibration
- per-expiry calibration on market data
- dense-grid surface interpolation
- historical cleaned-data workflow

## Core modules

- `src/implied_volatility_diffusion/synthetic_ivs_generator/sabr.py`
  - `sabr_lognormal_iv`
  - `calibrate_sabr_to_implied_vols`
- `src/implied_volatility_diffusion/synthetic_ivs_generator/sabr_iv_surface.py`
  - `calibrate_params_for_expiries`
  - `implied_vol_surface_from_calibrated_slices`
  - `implied_vol_surface_for_params`
  - LHS helpers for synthetic SABR surfaces
- `src/implied_volatility_diffusion/data/historical_sabr_interpolation.py`
  - historical-day SABR calibration and dense surface build
  - kernel-vs-SABR comparison helpers

## Environment

From repository root:

```bash
uv sync --group dev
```

For notebooks:

```bash
uv sync --group notebooks
```

## Historical data pipeline and cleaned input

Cleaning is done by:

- `src/implied_volatility_diffusion/data/data_pipeline.py`
  - `process_file(...)` -> `clean_data(...)`

Configured paths are in:

- `config/data_pipeline_config.yaml`

Default processed dataset:

- `data/processed/processed.parquet`

Rebuild processed parquet:

```bash
python -m implied_volatility_diffusion.data.data_pipeline
```

Load cleaned data in Python:

```python
from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import load_cleaned_data
df = load_cleaned_data("data/processed/processed.parquet")
```

## Typical SABR interpolation workflow on market data

1. Slice one valuation date (`quote_date`) from cleaned data.
2. Group quotes by expiry (`expire_date` or fallback rounded `tau`).
3. Calibrate `(alpha, rho, nu)` per expiry for fixed `beta`.
4. Evaluate SABR on a dense grid of `k_grid` and `tau_grid`.

`build_historical_sabr_surface(...)` wraps this flow.

## Example: historical day interpolation

```python
import numpy as np
import pandas as pd

from implied_volatility_diffusion.data.historical_data_smoothing_interpolation import load_cleaned_data
from implied_volatility_diffusion.data.historical_sabr_interpolation import build_historical_sabr_surface

df = load_cleaned_data("data/processed/processed.parquet")
day_df = df[df["quote_date"] == pd.Timestamp("2021-06-17")].copy()

k_grid = np.linspace(-0.2, 0.2, 41)
tau_grid = np.linspace(0.05, 1.0, 24)

sabr_out = build_historical_sabr_surface(day_df, k_grid, tau_grid, r=0.03, q=0.0, beta=0.5)
Z_sabr = sabr_out.surface
```

## Optional kernel comparison

Kernel and SABR are independent interpolators on the same cleaned historical quotes.
If you evaluate both on the same `(k_grid, tau_grid)`, use:

- `compare_kernel_sabr_surfaces(Z_kernel, Z_sabr)`

for overlap and RMSE summary.

## Detailed reference

For the full mathematical and API walkthrough, see:

- `docs/sabr_interpolation.md`

