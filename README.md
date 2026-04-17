# Implied-Volatility-Surface-Diffusion-Transformer (IVS-DiT)

## Setup

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

For Jupyter:

```bash
uv sync --group notebooks
```

## Tests

```bash
uv run pytest
```

Fallback with venv + pip:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install pytest scipy pyyaml matplotlib
.venv/bin/python -m pytest -q
```

## Configuration

[`config/heston_iv_surface.yaml`](config/heston_iv_surface.yaml) Market, Heston parameter box, Latin Hypercube, Black-Scholes implied-vol inversion, and Heston-COS pricer settings.
[`config/sabr_iv_surface.yaml`](config/sabr_iv_surface.yaml) Market assumptions, SABR parameter box, Latin Hypercube settings, and shared moneyness and maturity grid.
[`config/iv_surface_grid.yaml`](config/iv_surface_grid.yaml) Shared grid and plot surface settings.

## Heston-COS synthetic implied-volatility generator

**Goal:** produce many diverse **Black-Scholes implied-volatility** surfaces that are consistent with the **Heston stochastic-volatility** model for training and benchmarking.

**Parameter sampling:** `lhs_heston_params` and `lhs_heston_params_multi_batch` sample rows in fixed order `HESTON_PARAM_ORDER`. Ranges come from `heston_ranges`; optional `lhs.log_uniform` maps selected positive parameters through log-uniform margins for better scale coverage.

**Pricing:** for each `(strike, tau)` on the grid, `heston_call_cos` (Fang and Oosterlee Fourier-cosine method) evaluates the discounted European call under Heston.

**Implied volatility:** model call price is inverted with `implied_volatility` in `black_scholes.py` using py_lets_be_rational, a short Newton refinement, and Brent fallback.

**Batch surfaces:** `implied_vol_surfaces_lhs(cfg)` returns `(params, m, tau, iv)` with `iv` shape `(n_samples, n_moneyness, n_tau)`; total samples are `n_samples * n_batches` when multi-batch LHS is enabled.

## SABR baseline surface generator

**Goal:** generate and calibrate **Black implied-volatility surfaces** under the **SABR lognormal approximation (Hagan 2002)** so the project has a classical baseline before full diffusion/transformer training.

**SABR core math** in `src/implied_volatility_diffusion/synthetic_ivs_generator/sabr.py`:

- `sabr_hagan_lognormal_iv` evaluates one implied vol for `(forward, strike, tau)`.
- `calibrate_sabr_to_implied_vols` calibrates `(alpha, rho, nu)` for one expiry smile with bounded `scipy.optimize.least_squares`.

**Surface assembly** in `src/implied_volatility_diffusion/synthetic_ivs_generator/sabr_iv_surface.py`:

- `implied_vol_surface_for_params` builds one full surface from one SABR parameter vector.
- `lhs_sabr_params` and `implied_vol_surfaces_lhs` generate synthetic SABR surfaces from LHS draws.
- `calibrate_params_for_expiries` calibrates one SABR smile per market expiry.
- `implied_vol_surface_from_calibrated_slices` maps calibrated expiry slices to the grid (nearest-expiry mapping by default).

## Generic surface engine

`src/implied_volatility_diffusion/iv_surface.py` is model-agnostic: it builds grid axes from config, samples parameter vectors with Latin Hypercube sampling, and assembles batches of surfaces. Both SABR and Heston reuse this shared grid and sampling pattern.

## Notebook

`notebooks/sabr_synthetic_ivs.ipynb` demonstrates loading SABR config, generating one surface and batches of synthetic surfaces, and plotting a 3D surface and selected smiles.

## Plot example (script mode)

```python
from implied_volatility_diffusion.synthetic_ivs_generator.sabr_iv_surface import (
    implied_vol_surface_for_params,
    plot_sabr_smiles,
    plot_sabr_surface,
)

params = [0.2, -0.2, 0.35]  # alpha, rho, nu
m, tau, iv = implied_vol_surface_for_params(params, cfg)
fig_surface = plot_sabr_surface(m, tau, iv, cfg=cfg)
fig_smiles = plot_sabr_smiles(m, tau, iv, smile_taus=[0.2, 0.6, 1.0])
```


