"""Grid axes built from a ``{start_point, step, end_point}`` config dict."""

from typing import Any, Mapping

import numpy as np


def build_grid_axis(spec: Mapping[str, Any]) -> np.ndarray:
    """Build a 1D grid from ``start_point``, ``step``, ``end_point``.

    ``end_point`` is always included. ``prepend`` is an optional list of
    additional points merged in and deduplicated.
    """
    start = float(spec["start_point"])
    step = float(spec["step"])
    end = float(spec["end_point"])

    if step <= 0.0:
        raise ValueError("grid step must be positive")
    if end + 1e-15 < start:
        raise ValueError("end_point must be >= start_point")

    pts: list[float] = [start]
    tol = 1e-12 * max(1.0, abs(end))
    while True:
        nxt = pts[-1] + step
        if nxt > end + tol:
            break
        pts.append(float(nxt))
    if abs(pts[-1] - end) > tol:
        pts.append(end)
    base = np.asarray(pts, dtype=float)
    prepend = spec.get("prepend")
    if prepend:
        extra = np.asarray([float(x) for x in prepend], dtype=float)
        base = np.unique(np.sort(np.concatenate([extra, base])))
    return base


def grid_axes(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """NumPy moneyness / maturity axes from ``cfg['grid']`` (legacy API)."""
    grid = cfg["grid"]
    return build_grid_axis(grid["moneyness"]), build_grid_axis(grid["tau"])
