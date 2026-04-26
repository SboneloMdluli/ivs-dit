"""Tests for the closed-form VP forward noising scheduler."""

import numpy as np

from implied_volatility_diffusion import VPNoiseScheduler


def test_vp_scheduler_converges_to_standard_normal() -> None:
    """At large t, mean approaches 0 and variance approaches 1."""
    rng = np.random.default_rng(7)
    x0 = np.full(200_000, 3.0, dtype=float)
    beta = 1.0
    t = 20.0

    scheduler = VPNoiseScheduler(beta=beta)
    xt = scheduler.add_noise(x0, t=t, rng=rng)

    mean_emp = float(np.mean(xt))
    var_emp = float(np.var(xt))

    # Theoretical values
    decay = np.exp(-0.5 * beta * t)
    mean_theory = decay * 3.0
    var_theory = 1.0 - np.exp(-beta * t)

    assert np.isclose(mean_emp, mean_theory, atol=1e-2)
    assert np.isclose(var_emp, var_theory, atol=2e-2)


def test_vp_scheduler_variance_follows_closed_form_and_is_bounded() -> None:
    """Variance should follow closed form and remain bounded."""
    rng = np.random.default_rng(11)
    x0 = rng.normal(loc=0.0, scale=2.5, size=200_000)
    var0 = float(np.var(x0))

    beta = 0.8
    scheduler = VPNoiseScheduler(beta=beta)

    for t in [1.0, 5.0, 25.0, 100.0]:
        xt = scheduler.add_noise(x0, t=t, rng=rng)
        var_emp = float(np.var(xt))

        var_theory = np.exp(-beta * t) * var0 + (1.0 - np.exp(-beta * t))

        # Check exact behavior
        assert np.isclose(var_emp, var_theory, atol=2e-2)

        # Explicit boundedness
        assert var_emp <= max(var0, 1.0) + 1e-6
