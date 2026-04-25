# Heston Surface Generation

This document describes how synthetic implied-volatility surfaces are generated under the Heston model in this project.

## Goal

Produce diverse Black-Scholes implied-volatility surfaces consistent with Heston stochastic volatility dynamics for downstream model training and benchmarking.

## High-Level Flow

1. Sample Heston parameter vectors with Latin Hypercube Sampling (LHS).
2. Price call options on a shared `(moneyness, tau)` grid using Heston COS pricing.
3. Invert prices to Black-Scholes implied volatilities.
4. Return batched IV surfaces for all sampled parameter sets.

## Parameter Sampling

- Parameters follow the fixed order `HESTON_PARAM_ORDER`.
- Main entrypoints:
  - `lhs_heston_params`
  - `lhs_heston_params_multi_batch`
- Parameter ranges come from config (`heston_ranges` section).
- Optional `lhs.log_uniform` can sample selected positive parameters in log space for better coverage.

## Pricing Step

- For each sampled row and each grid point `(K, tau)`, the pipeline prices a European call using Heston COS.
- COS implementation uses the Fang-Oosterlee Fourier-cosine method.
- Relevant modules live under:
  - `src/implied_volatility_diffusion/models/heston/`
  - `src/implied_volatility_diffusion/synthetic/`

## Implied Volatility Inversion

- Model prices are converted to implied volatility with routines in `src/implied_volatility_diffusion/pricing/implied_vol.py`.
- Inversion stack includes:
  - `py_lets_be_rational`-based implied vol routines
  - local numeric refinement/fallback handling

## Output Contract

`implied_vol_surfaces_heston_lhs(cfg, ...)` returns:

- `params`: shape `(n_samples_total, n_params)`
- `m`: shape `(n_moneyness,)`
- `tau`: shape `(n_tau,)`
- `iv`: shape `(n_samples_total, n_moneyness, n_tau)`

Where `n_samples_total = n_samples * n_batches` when multi-batch LHS is enabled.
