"""Tests for the closed-form VP forward noising scheduler."""

import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "implied_volatility_diffusion" / "noise_scheduler.py"
_SPEC = importlib.util.spec_from_file_location("noise_scheduler", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Failed to load noise_scheduler module")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
VPNoiseScheduler = _MOD.VPNoiseScheduler


def test_vp_scheduler_large_t_converges_to_standard_normal_moments() -> None:
    """At large t, mean approaches 0 and variance approaches 1."""
    rng = np.random.default_rng(7)
    x0 = np.full(200_000, 3.0, dtype=float)
    scheduler = VPNoiseScheduler(beta=1.0)

    xt = scheduler.add_noise(x0, t=20.0, rng=rng)
    mean = float(np.mean(xt))
    var = float(np.var(xt))

    assert abs(mean) < 1.5e-2
    assert abs(var - 1.0) < 3.0e-2


def test_vp_scheduler_variance_is_bounded_for_large_t() -> None:
    """Variance should not explode; VP process is bounded by max(var0, 1)."""
    rng = np.random.default_rng(11)
    x0 = rng.normal(loc=0.0, scale=2.5, size=200_000)  # Var ~ 6.25
    var0 = float(np.var(x0))
    scheduler = VPNoiseScheduler(beta=0.8)

    for t in [1.0, 5.0, 25.0, 100.0]:
        xt = scheduler.add_noise(x0, t=t, rng=rng)
        vart = float(np.var(xt))
        assert vart <= max(var0, 1.0) + 0.1
