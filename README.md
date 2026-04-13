# Implied-Volatility-Surface-Diffusion-Transformer (IVS-DiT)

## Setup

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run pytest
```

For Jupyter:

```bash
uv sync --group notebooks
```

## Tests

```bash
uv run pytest
```

## Heston-COS synthetic Implied-Volatility Generator

**Goal:** produce many diverse **Black–Scholes implied volatility** surfaces that are consistent with the **Heston stochastic volatility** model for training and benchmarking.

**Parameter sampling**: `lhs_heston_params` / `lhs_heston_params_multi_batch` sample rows in fixed order `HESTON_PARAM_ORDER`. Ranges come from `heston_ranges` in `config/heston_iv_surface.yaml`; optional `lhs.log_uniform` maps selected positive parameters through log-uniform margins for better coverage of scales.

**Config layout**: `config/heston_iv_surface.yaml` holds market, Heston ranges, LHS, implied-vol inversion, and `heston_cos_pricer` (COS settings). `config/iv_surface_grid.yaml` holds the shared **`grid`** (moneyness and maturity axes) and **`plot_surface`** (matplotlib defaults). Use `load_heston_iv_surface_config(config_dir)` to merge both into one dict, or `merge_config_files` from `ivs_config` for arbitrary multi-file merges.

**Pricing**: For each `(strike, τ)` on the grid, `heston_call_cos` (Fang & Oosterlee Fourier–cosine method) evaluates the **discounted European call** under Heston. The implementation uses the Heston characteristic function, cumulant-based truncation of the log-spot interval `[a, b]`, and strike-aware widening so short maturities still bracket `log(K)` reliably.

**Implied volatility**: The model call price is inverted to Black–Scholes σ with `implied_volatility` in `black_scholes.py`: Jäckel’s rational guess via `py_lets_be_rational`, a short Newton refinement on the price residual, and Brent fallback when vega is tiny or the wing is difficult.

**Batch surfaces**: `implied_vol_surfaces_lhs(cfg)` returns `(params, m, tau, iv)` with `iv` of shape `(n_samples, n_moneyness, n_tau)` (total samples = `n_samples × n_batches` when multi-batch LHS is enabled). `implied_vol_surface_for_params` computes a single surface for one parameter vector.

## Generic surface engine

`iv_surface.py` is **model agnostic**: it builds grid axes from config, draws parameter vectors with Latin Hypercube sampling, and for each grid point calls a `model call pricer` (strike and time to maturity -> discounted European price) and a Black–Scholes implied-vol inverter. Heston is just one implementation of the pricer.