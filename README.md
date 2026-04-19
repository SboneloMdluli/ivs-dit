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

## Configuration

[`config/heston_iv_surface.yaml`](config/heston_iv_surface.yaml) Market, Heston parameter box, Latin Hypercube, Black–Scholes implied-vol inversion, and Heston-COS pricer settings.
[`config/iv_surface_grid.yaml`](config/iv_surface_grid.yaml) Shared grid and plot surface


## Heston-COS synthetic Implied-Volatility Generator

**Goal:** produce many diverse **Black–Scholes implied volatility** surfaces that are consistent with the **Heston stochastic volatility** model for training and benchmarking.

**Parameter sampling**: `lhs_heston_params` / `lhs_heston_params_multi_batch` sample rows in fixed order `HESTON_PARAM_ORDER`. Ranges come from `heston_ranges`; optional `lhs.log_uniform` maps selected positive parameters through log-uniform margins for better coverage of scales. Each row is then **Feller-clipped**: `sigma_v` is reduced row-by-row to `sqrt(max(0, 2·κ·θ − lhs.feller_eps))` so `2·κ·θ ≥ sigma_v² + feller_eps` holds for every draw.

**Pricing**: For each `(strike, τ)` on the grid, `heston_call_cos` (Fang & Oosterlee Fourier–cosine method) evaluates the **discounted European call** under Heston. The implementation uses the Heston characteristic function, cumulant-based truncation of the log-spot interval `[a, b]`, and strike-aware widening so short maturities still bracket `log(K)` reliably.

**Batch surfaces**: `implied_vol_surfaces_lhs(cfg)` returns `(params, m, tau, iv)` with `iv` of shape `(n_samples, n_moneyness, n_tau)` (total samples = `n_samples × n_batches` when multi-batch LHS is enabled). `implied_vol_surface_for_params` computes a single surface for one parameter vector. Pass a **merged** config (for example from `load_heston_iv_surface_config`).

`iv_surface.py` is **model agnostic**: it builds grid axes from config, draws parameter vectors with Latin Hypercube sampling, and for each grid point calls a `model call pricer` (strike and time to maturity → discounted European price) and a Black–Scholes implied-vol inverter. Heston is just one implementation of the pricer.
