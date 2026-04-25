# Option Data Pipeline

This document summarizes the historical option-data pipeline used to build datasets for IV surface construction, smoothing/interpolation, and model validation.

## Objective

Construct a multi-year option-chain dataset from vendor raw files, standardize schema, engineer IV-related features, and output clean modeling-ready parquet files.

## Data Source

- Raw option-chain files: `data/raw/optiondx/` (TXT format, source: `www.optionsdx.com`)
- Typical coverage: multi-year monthly snapshots

Preprocessed datasets (raw + cleaned) are available here:

- [Google Drive dataset folder](https://drive.google.com/drive/folders/1RyOj4Ylcqgo5ItAcTGJWsiuKayZ-qvYI?usp=drive_link)

## Pipeline Stages

1. **Ingestion**
   - Read vendor TXT files into tabular frames.
   - Normalize column names and schema.
   - Parse and standardize date fields.
   - Coerce numeric columns robustly to handle vendor inconsistencies.

2. **Quote Normalization**
   - Split calls and puts into a standardized row format.

3. **Feature Engineering**
   - Time to maturity: `tau = DTE / 365`
   - Log moneyness (or related moneyness transforms, by workflow)
   - Mid price: `(bid + ask) / 2`
   - Spread and relative spread
   - Total implied variance: `w = tau * sigma^2`
   - Smoothing weights (liquidity/vega-based composites)

4. **Data Cleaning**
   - Keep rows with valid bid/ask and IV fields.
   - Enforce positive maturity and quote sanity conditions (`ask >= bid`, spread constraints).

5. **Outputs**
   - Minimally processed dataset: `data/raw/raw.parquet`
   - Cleaned and feature-enriched dataset: `data/processed/cleaned.parquet`

## Notes

- Exact filters and feature variants can differ by notebook/experiment.
- See notebooks in `notebooks/` for exploratory validation and interpolation steps.
