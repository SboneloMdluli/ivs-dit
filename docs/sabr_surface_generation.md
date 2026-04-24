# SABR Surface Generation

This document describes the SABR baseline workflow for generating and calibrating implied-volatility surfaces.

## Goal

Provide a classical baseline based on the SABR lognormal approximation (Hagan 2002) before or alongside diffusion/transformer-style IVS models.

## Core Components

- SABR model implementation and calibration logic:
  - `src/implied_volatility_diffusion/models/sabr/`
- Synthetic surface generation utilities:
  - `src/implied_volatility_diffusion/synthetic/sabr.py`
  - `src/implied_volatility_diffusion/synthetic/surface.py`

## Main Capabilities

- Evaluate SABR implied volatility for `(forward, strike, tau)` points.
- Generate synthetic SABR parameter sets via LHS.
- Build full surfaces from sampled SABR parameters.
- Calibrate expiry-by-expiry SABR slices to observed smiles, then map slices to a target grid.

## Typical Workflow

1. Define market/grid assumptions in `config/sabr_iv_surface.yaml`.
2. Sample parameter vectors with LHS.
3. Generate implied vol values over shared moneyness/maturity axes.
4. (Optional) Calibrate to market data by expiry and rebuild grid surfaces from calibrated slices.

## Related Guide

For a practical interpolation walkthrough on market data, see [`docs/sabr_interpolation.md`](sabr_interpolation.md).
