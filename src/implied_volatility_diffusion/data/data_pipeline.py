from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def discover_file(sourse_dir: str | Path, pattern: str = "spx_eod_*.txt") -> list[Path]:
    """Discover raw option data files in the source directory matching the pattern."""
    source_dir = Path(sourse_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Source directory {source_dir} does not exist.")
    
    files = sorted(source_dir.glob(pattern))
    print(f"Discovered {len(files)} files in {source_dir} matching pattern {pattern}.")
    if not files:
        raise ValueError(f"No files found in {source_dir} matching pattern {pattern}.")
    
    return files

def load_file(path: str | Path) -> pd.DataFrame:
    """Load a single raw option data file into a DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"File {path} does not exist.")
    
    try:
        df = pd.read_csv(path, low_memory=False)
        # print(f"Loaded file {path} with shape {df.shape}.")
        df.columns = [str(c).strip().strip("[]").strip() for c in df.columns]
        df['source_file'] = path.name  # Add source file name as a column for traceability  
        date_cols = ["QUOTE_DATE", "EXPIRE_DATE", "QUOTE_READTIME"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        for col in df.columns:
            if col == "source_file":
                continue

            if (
                col.startswith("C_")
                or col.startswith("P_")
                or col in {
                    "UNDERLYING_LAST",
                    "STRIKE",
                    "DTE",
                    "STRIKE_DISTANCE",
                    "STRIKE_DISTANCE_PCT",
                    "QUOTE_UNIXTIME",
                    "QUOTE_TIME_HOURS",
                    "EXPIRE_UNIX",
                }
            ):
                df[col] = df[col].astype(str).str.strip()
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        raise ValueError(f"Error loading file {path}: {e}") from e
    
def explode_options(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the options to separate calls and puts."""
    
    base_cols = [
        "QUOTE_DATE",
        "EXPIRE_DATE",
        "UNDERLYING_LAST",
        "STRIKE",
        "DTE",
        "source_file",
    ]

    call_cols = [
        "C_BID", "C_ASK", "C_LAST", "C_VOLUME", "C_IV", "C_DELTA", "C_GAMMA", "C_VEGA"
    ]
    put_cols = [
        "P_BID", "P_ASK", "P_LAST", "P_VOLUME", "P_IV", "P_DELTA", "P_GAMMA", "P_VEGA"
    ]

    missing_base = [c for c in base_cols if c not in df.columns]
    if missing_base:
        raise KeyError(f"Missing base columns: {missing_base}")

    missing_call = [c for c in call_cols if c not in df.columns]
    missing_put = [c for c in put_cols if c not in df.columns]
    if missing_call:
        raise KeyError(f"Missing call columns: {missing_call}")
    if missing_put:
        raise KeyError(f"Missing put columns: {missing_put}")
    
    calls = df[base_cols + call_cols].copy()
    calls.columns = base_cols + ["bid", "ask", "last", "volume", "iv", "delta", "gamma", "vega"]
    calls["option_type"] = "call"

    puts = df[base_cols + put_cols].copy()
    puts.columns = base_cols + ["bid", "ask", "last", "volume", "iv", "delta", "gamma", "vega"]
    puts["option_type"] = "put"

    out = pd.concat([calls, puts], ignore_index=True)

    out = out.rename(
        columns={
            "QUOTE_DATE": "quote_date",
            "EXPIRE_DATE": "expire_date",
            "UNDERLYING_LAST": "underlying_last",
            "STRIKE": "strike",
            "DTE": "dte",
        }
    )

    out["quote_date"] = pd.to_datetime(out["quote_date"], errors="coerce")
    out["expire_date"] = pd.to_datetime(out["expire_date"], errors="coerce")

    numeric_cols = [
        "underlying_last",
        "strike",
        "dte",
        "bid",
        "ask",
        "last",
        "volume",
        "iv",
        "delta",
        "gamma",
        "vega",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """add features such as mid price, spread, moneyness, log-moneyness, total variance, and weights."""
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    
    df['rel_spread'] = np.where(df['mid'] > 0, df['spread'] / df['mid'], np.nan)
    df['tau'] = df['dte'] / 365.0

    df['moneyness'] = np.where(
        df['underlying_last'] > 0, 
        df['strike'] / df['underlying_last'], 
        np.nan
    )
    df["k"] = np.where(
        (df['strike'] > 0) & (df['underlying_last'] > 0),
        np.log(df['strike'] / df['underlying_last']),
        np.nan
    )

    df['abs_k'] = df['k'].abs()

    df['total_variance'] = np.where(
        (df['tau'] > 0) & (df['iv'] > 0),
        (df['iv'] ** 2) * df['tau'],
        np.nan
    )

    df['liq_weight'] = np.where(
        df['rel_spread'] > 0,
        1.0 / df['rel_spread'],
        np.nan
    )

    df['vega_weight'] = np.where(
        df['vega'] > 0,
        df['vega'],
        np.nan
    )

    df['smooth_weight'] = df['liq_weight'] * df['vega_weight']

    return df

def clean_data(
        df: pd.DataFrame,
        min_tau: float = 1.0 / 365,
        min_iv: float = 0.01,
        max_iv: float = 3.0,
        max_rel_spread: float = 2.0,
) -> pd.DataFrame:
    """clean the data by applying filters on mid price, time to maturity, implied volatility, strike price, underlying price, bid-ask spread, and relative spread."""
    df = df.copy()

    mask = df['mid'].notna() & (df['mid'] > 0)
    mask &= df['tau'].notna() & (df['tau'] >= min_tau)
    mask &= df['iv'].notna() & (df['iv'] >= min_iv) & (df['iv'] <= max_iv)
    mask &= df['strike'].notna() & (df['strike'] > 0)
    mask &= df['underlying_last'].notna() & (df['underlying_last'] > 0)
    mask &= df['bid'].notna() & (df['bid'] >= 0)
    mask &= df['ask'].notna() & (df['ask'] > 0)
    mask &= df['ask'] >= df['bid']

    if 'rel_spread' in df.columns:
        mask &= df['rel_spread'].notna() & (df['rel_spread'] <= max_rel_spread)

    return df.loc[mask].reset_index(drop=True)

def process_file(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single raw option data file and return the raw and processed DataFrames."""
    df_raw = load_file(file_path)
    processed = explode_options(df_raw)
    processed = add_features(processed)
    processed = clean_data(processed)

    return df_raw, processed

def add_or_write_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """Add to an existing parquet file or write a new one if it doesn't exist."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(output_path, index=False)
        print(f"Appended {df.shape[0]} rows to existing file {output_path}. New shape: {combined.shape}.")
    else:
        df.to_parquet(output_path, index=False)
        print(f"Wrote new file {output_path} with shape {df.shape}.")

def build_dataset(
        source_dir: str | Path,
        raw_output_path: str | Path,
        processed_output_path: str | Path,
        limit_files: int | None = None,
        batch_size: int = 6,
) -> None:
    """build the dataset by processing all raw option data files in the source directory and saving the raw and processed datasets to parquet files."""
    files = discover_file(source_dir)

    if limit_files is not None:
        files = files[:limit_files]

    print(f"Processing {len(files)} files from {source_dir}...")

    raw_output_path = Path(raw_output_path)
    processed_output_path = Path(processed_output_path)

    if raw_output_path.exists():
        raw_output_path.unlink()
    if processed_output_path.exists():
        processed_output_path.unlink()

    raw_buffer = []
    processed_buffer = []
    
    for i, path in enumerate(files, 1):
        print(f"Processing file {i}/{len(files)}: {path.name}")
        try:
            raw_df, processed_df = process_file(path)
            raw_buffer.append(raw_df)
            processed_buffer.append(processed_df)

            if len(raw_buffer) >= batch_size:
                raw_chunk= pd.concat(raw_buffer, ignore_index=True)
                process_chunk = pd.concat(processed_buffer, ignore_index=True)

                add_or_write_parquet(raw_chunk, raw_output_path)
                add_or_write_parquet(process_chunk, processed_output_path)

                print(
                    f'Saved batch ending at file {i}: '
                    f'raw chunk shape {raw_chunk.shape}, '
                    f'processed chunk shape {process_chunk.shape}'
                )

                raw_buffer.clear()
                processed_buffer.clear()
        except Exception as e:
            print(f"Error processing file {path}: {e}")

    if raw_buffer:
        raw_chunk = pd.concat(raw_buffer, ignore_index=True)
        process_chunk = pd.concat(processed_buffer, ignore_index=True)

        add_or_write_parquet(raw_chunk, raw_output_path)
        add_or_write_parquet(process_chunk, processed_output_path)

        print(
            f'Saved final batch: '
            f'raw chunk shape {raw_chunk.shape}, '
            f'processed chunk shape {process_chunk.shape}'
        )

    final_raw = pd.read_parquet(raw_output_path)
    final_processed = pd.read_parquet(processed_output_path)

    print(f"Saved raw dataset to {raw_output_path} with shape {final_raw.shape}.")
    print(f"Saved processed dataset to {processed_output_path} with shape {final_processed.shape}.")

def main():
    config_path = Path("config/data_pipeline_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    source_dir = config['paths']['source_dir']
    raw_output_path = config['paths']['raw_output_path']
    processed_output_path = config['paths']['processed_output_path']

    limit_files = config.get('run', {}).get('limit_files', None)
    batch_size = config.get('run', {}).get('batch_size', 6)

    discovered = discover_file(source_dir)
    print(f"Discovered {len(discovered)} files in {source_dir}.")

    build_dataset(
        source_dir=source_dir,
        raw_output_path=raw_output_path,
        processed_output_path=processed_output_path,
        limit_files=limit_files,
        batch_size=batch_size
    )

if __name__ == "__main__":
    main()