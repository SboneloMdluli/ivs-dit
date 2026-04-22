from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def gaussian_kernel_2d(
    x: np.ndarray,
    y: np.ndarray,
    h1: float,
    h2: float,
) -> np.ndarray:
    """2D Gaussian kernel weights for points (x, y) with bandwidths h1, h2."""
    return (np.exp(-x * x / (2 * h1)) * np.exp(-y * y / (2 * h2))) / (2 * np.pi)


def interpolate_and_extrapolate(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Interpolate/extrapolate y values at x_new based on (x_data, y_data)."""
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")
    if len(x_data) < 2:
        raise ValueError("Need at least two points for interpolation.")

    idx = np.argsort(x_data)
    x_sorted = x_data[idx]
    y_sorted = y_data[idx]

    f = interp1d(
        x_sorted,
        y_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return f(x_new)

<<<<<<< HEAD

def interpolate_surface(
    surface_in: np.ndarray,
    x_in: np.ndarray,
    tau_in: np.ndarray,
    x_out: np.ndarray,
    tau_out: np.ndarray,
) -> np.ndarray:
    """Interpolate/extrapolate surface values from (x_in, tau_in) grid to (x_out, tau_out) grid."""
=======
def interpolate_surface(
        surface_in: np.ndarray,
        x_in: np.ndarray,
        tau_in: np.ndarray,
        x_out: np.ndarray,
        tau_out: np.ndarray,
) -> np.ndarray:
    """
    Interpolate/extrapolate surface values from (x_in, tau_in) grid to (x_out, tau_out) grid.    
    """

>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
    tmp = np.zeros((len(x_out), len(tau_in)), dtype=float)
    for j in range(len(tau_in)):
        tmp[:, j] = interpolate_and_extrapolate(x_in, surface_in[:, j], x_out)

    out = np.zeros((len(x_out), len(tau_out)), dtype=float)
    for i in range(len(x_out)):
        out[i, :] = interpolate_and_extrapolate(tau_in, tmp[i, :], tau_out)

    return out

<<<<<<< HEAD

=======
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
def load_cleaned_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if "quote_date" not in df.columns:
        raise ValueError("Expected 'quote_date' column in the data file.")

    df = df.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")

    if "expire_date" in df.columns:
        df["expire_date"] = pd.to_datetime(df["expire_date"], errors="coerce")

    return df

<<<<<<< HEAD

def filter_day_for_surface(
    day_df: pd.DataFrame,
    k_range: tuple[float, float] = (-0.5, 0.5),
    tau_range: tuple[float, float] = (0.01, 2.0),
    iv_range: tuple[float, float] = (0.01, 2.0),
) -> pd.DataFrame:
    """Filter day_df for valid points in specified (k, tau, iv) ranges."""
    required_cols = {"k", "tau", "iv", "vega"}
    missing_cols = [c for c in required_cols if c not in day_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in day_df: {missing_cols}")

    out = day_df.copy()
    out = out.dropna(subset=["quote_date", "k", "tau", "iv", "vega"]).copy()
    out = out[
        out["k"].between(*k_range) & out["tau"].between(*tau_range) & out["iv"].between(*iv_range) & (out["vega"] > 0)
    ].copy()
    return out


def build_weight_for_smoothing(
    day_df: pd.DataFrame,
    weight_col: str = "vega",
    clip_upper: float | None = None,
    sqrt_weight: bool = False,
=======
def filter_day_for_surface(
        day_df: pd.DataFrame,
        k_range: tuple[float, float] = (-0.5, 0.5),
        tau_range: tuple[float, float] = (0.01, 2.0),
        iv_range: tuple[float, float] = (0.01, 2.0),
) -> pd.DataFrame:
    """Filter day_df for valid points in specified (k, tau, iv) ranges."""
    required_cols = {"k", "tau", "iv", 'vega'}
    missing_cols = [c for c in required_cols if c not in day_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in day_df: {missing_cols}")
    
    out = day_df.copy()
    out = out.dropna(subset=["quote_date", "k", "tau", "iv", "vega"]).copy()
    out = out[
        out["k"].between(*k_range)
        & out["tau"].between(*tau_range)
        & out["iv"].between(*iv_range)
        & (out["vega"] > 0)
    ].copy()
    return out

def build_weight_for_smoothing(
        day_df: pd.DataFrame,
        weight_col: str = "vega",
        clip_upper: float | None = None,
        sqrt_weight: bool = False,
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
) -> np.ndarray:
    """Build weight array for smoothing based on specified column, with optional clipping and sqrt."""
    if weight_col not in day_df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in day_df.")
<<<<<<< HEAD

=======
    
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
    w = day_df[weight_col].to_numpy(dtype=float)

    if sqrt_weight:
        w = np.sqrt(np.maximum(w, 0))

    if clip_upper is not None:
        w = np.clip(w, 0, clip_upper)

    return w

<<<<<<< HEAD

def smooth_surface(
    iv: np.ndarray,
    x_in: np.ndarray,
    tau_in: np.ndarray,
    weights: np.ndarray,
    x_grid: np.ndarray,
    tau_grid: np.ndarray,
    h1: float = 0.01,
    h2: float = 0.10,
    min_mass: float = 1e-8,
=======
def smooth_surface(
        iv: np.ndarray,
        x_in: np.ndarray,
        tau_in: np.ndarray,
        weights: np.ndarray,
        x_grid: np.ndarray,
        tau_grid: np.ndarray,
        h1: float = 0.01,
        h2: float = 0.10,
        min_mass: float = 1e-8,
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
) -> np.ndarray:
    """Smooth IV surface using weighted Gaussian kernel."""
    iv = np.asarray(iv, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    tau_in = np.asarray(tau_in, dtype=float)
    weights = np.asarray(weights, dtype=float)

    Z = np.full((len(x_grid), len(tau_grid)), np.nan, dtype=float)
    values = iv

    for i, x0 in enumerate(x_grid):
        dx = x0 - x_in
        for j, tau0 in enumerate(tau_grid):
            dt = tau0 - tau_in
            kernel_weights = gaussian_kernel_2d(dx, dt, h1, h2) * weights
            mass = np.sum(kernel_weights)
<<<<<<< HEAD

=======
            
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
            if mass < min_mass:
                continue

            z = np.sum(kernel_weights * values) / mass
            Z[i, j] = z
    return Z

<<<<<<< HEAD

def build_kernel_surface(
    day_df: pd.DataFrame,
    x_col: str = "k",
    weight_col: str = "vega",
    x_grid: np.ndarray | None = None,
    tau_grid: np.ndarray | None = None,
    h1: float = 0.01,
    h2: float = 0.10,
    clip_upper: float | None = None,
    sqrt_weight: bool = False,
=======
def build_kernel_surface(
        day_df: pd.DataFrame,
        x_col: str = "k",
        weight_col: str = "vega",
        x_grid: np.ndarray | None = None,
        tau_grid: np.ndarray | None = None,
        h1: float = 0.01,
        h2: float = 0.10,
        clip_upper: float | None = None,
        sqrt_weight: bool = False,
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build smoothed IV surface for a day using Gaussian kernel."""
    if x_grid is None or tau_grid is None:
        raise ValueError("x_grid and tau_grid must be provided.")
    if x_col not in day_df.columns:
        raise ValueError(f"x-axis column not found: {x_col}")

    day_sub = filter_day_for_surface(day_df)

    w = build_weight_for_smoothing(
        day_sub,
        weight_col=weight_col,
        clip_upper=clip_upper,
        sqrt_weight=sqrt_weight,
    )

    Z = smooth_surface(
        iv=day_sub["iv"].values,
        x_in=day_sub[x_col].values,
        tau_in=day_sub["tau"].values,
        weights=w,
        x_grid=x_grid,
        tau_grid=tau_grid,
        h1=h1,
        h2=h2,
    )

    return Z, day_sub

<<<<<<< HEAD

## Plot functions:


def plot_smile(
    x_grid: np.ndarray,
=======
## Plot functions:

def plot_smile(
       x_grid: np.ndarray,
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
    tau_grid: np.ndarray,
    Z: np.ndarray,
    sample_date: pd.Timestamp,
    title: str,
    num_sections: int = 5,
) -> None:
    j_idx = [round(i * (len(tau_grid) - 1) / (num_sections - 1)) for i in range(num_sections)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for j in j_idx:
        ax.plot(
            x_grid,
            Z[:, j],
            marker="o",
            ms=3,
            lw=1.2,
            label=rf"$\tau={tau_grid[j]:.4f}$",
        )

    ax.set_xlabel(r"Log-moneyness $k=\log(K/S)$")
    ax.set_ylabel(r"Implied volatility $\sigma_{\mathrm{imp}}$")
    ax.set_title(f"{title} ({sample_date.date()})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_surface_3d(
    KGRID: np.ndarray,
    TAUGRID: np.ndarray,
    Z: np.ndarray,
    sample_date: pd.Timestamp,
    title: str = "Implied Volatility Surface",
    cmap: str = "turbo",
    elev: float = 25,
    azim: float = 235,
) -> None:
<<<<<<< HEAD
    """3D surface plot using the existing project style.
=======
    """
    3D surface plot using the existing project style.
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93

    Parameters
    ----------
    KGRID : np.ndarray
        Meshgrid for x-axis (log-moneyness k)
    TAUGRID : np.ndarray
        Meshgrid for tau
    Z : np.ndarray
        IV surface values
    sample_date : pd.Timestamp
        Date for title
    title : str
        Plot title
    cmap : str
        Colormap
    elev : float
        Elevation angle
    azim : float
        Azimuth angle
    """
<<<<<<< HEAD
=======

>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        KGRID,
        TAUGRID,
        Z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
    )

    ax.set_xlabel("Log-moneyness k = log(K/S)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied vol")

    ax.set_title(f"{title} ({sample_date})")

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    cax = fig.add_axes([0.91, 0.2, 0.022, 0.58])
    fig.colorbar(surf, cax=cax, label=r"$\sigma_{\mathrm{imp}}$")

<<<<<<< HEAD
    plt.show()
=======
    plt.show()
>>>>>>> f0f20627ccab22c18edc2922d20089d6c8dcee93
