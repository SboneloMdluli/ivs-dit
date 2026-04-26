"""Tests for the per-pixel ``log(sigma)`` normalizer."""

from __future__ import annotations

import numpy as np
import pytest

from implied_volatility_diffusion import (
    SurfaceNormalizer,
    denormalize_surface,
    iv_to_log_iv,
    normalize_surface,
)


def _make_stack(shape: tuple[int, int], n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-2.0, -0.5, size=shape)
    sigma = rng.uniform(0.05, 0.3, size=shape)
    log_iv = rng.normal(loc=mu, scale=sigma, size=(n, *shape))
    return np.exp(log_iv), mu, sigma


def test_fit_recovers_per_pixel_mean_and_std() -> None:
    shape = (5, 4)
    iv, mu_true, sigma_true = _make_stack(shape, n=10_000, seed=1)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    assert norm.fitted
    assert norm.mean.shape == shape
    assert norm.std.shape == shape
    assert np.allclose(norm.mean, mu_true, atol=0.02)
    assert np.allclose(norm.std, sigma_true, atol=0.02)


def test_transform_then_inverse_roundtrip() -> None:
    shape = (6, 5)
    iv, _, _ = _make_stack(shape, n=200, seed=2)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    z = norm.transform(iv)
    assert z.shape == iv.shape
    back = norm.inverse_transform(z)
    assert np.allclose(back, iv, atol=1e-12)


def test_zero_z_is_historical_mean() -> None:
    shape = (4, 3)
    iv, _, _ = _make_stack(shape, n=500, seed=3)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    zero_z = np.zeros(shape)
    mean_iv = norm.inverse_transform(zero_z)
    assert np.allclose(np.log(mean_iv), norm.mean)


def test_partial_fit_matches_single_pass_fit() -> None:
    shape = (3, 3)
    iv, _, _ = _make_stack(shape, n=237, seed=4)
    full = SurfaceNormalizer(grid_shape=shape).fit(iv)

    streaming = SurfaceNormalizer(grid_shape=shape)
    for chunk in np.array_split(iv, 5):
        streaming.partial_fit(chunk)

    assert np.allclose(streaming.mean, full.mean, atol=1e-10)
    assert np.allclose(streaming.std, full.std, atol=1e-10)


def test_nan_cells_are_ignored() -> None:
    shape = (2, 2)
    iv = np.ones((5, *shape))
    iv[0, 0, 0] = np.nan
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    assert norm.count[0, 0] == 4
    assert norm.count[1, 1] == 5
    # log(1) = 0; pixel mean must be 0 up to floor
    assert np.isclose(norm.mean[0, 0], 0.0, atol=1e-12)


def test_sigma_floor_prevents_divide_by_zero() -> None:
    shape = (2, 2)
    iv = np.full((100, *shape), 0.25)
    norm = SurfaceNormalizer(grid_shape=shape, sigma_floor=1e-4).fit(iv)
    assert np.all(norm.std >= 1e-4)
    z = norm.transform(iv)
    assert np.all(np.isfinite(z))


def test_save_and_load_roundtrip(tmp_path) -> None:
    shape = (4, 3)
    iv, _, _ = _make_stack(shape, n=50, seed=5)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    target = tmp_path / "norm.npz"
    norm.save(target)

    reloaded = SurfaceNormalizer.load(target)
    assert reloaded.grid_shape == shape
    assert np.allclose(reloaded.mean, norm.mean)
    assert np.allclose(reloaded.std, norm.std)
    z_a = norm.transform(iv)
    z_b = reloaded.transform(iv)
    assert np.allclose(z_a, z_b)


def test_shape_mismatch_raises() -> None:
    norm = SurfaceNormalizer(grid_shape=(3, 3))
    with pytest.raises(ValueError, match="pixel shape"):
        norm.fit(np.ones((10, 4, 4)))


def test_iv_to_log_iv_floors_tiny_values() -> None:
    iv = np.array([0.0, 1.0e-20, 0.1, np.nan])
    log_iv = iv_to_log_iv(iv, floor=1.0e-6)
    assert np.isclose(log_iv[0], np.log(1.0e-6))
    assert np.isclose(log_iv[1], np.log(1.0e-6))
    assert np.isclose(log_iv[2], np.log(0.1))
    assert np.isnan(log_iv[3])


def test_denormalize_method_is_inverse_of_normalize() -> None:
    shape = (4, 5)
    iv, _, _ = _make_stack(shape, n=64, seed=10)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    z = norm.normalize(iv)
    back = norm.denormalize(z)
    assert np.allclose(back, iv, atol=1e-12)


def test_denormalize_zero_z_is_per_pixel_mean() -> None:
    shape = (3, 4)
    iv, _, _ = _make_stack(shape, n=200, seed=11)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    sigma_mean = norm.denormalize(np.zeros(shape))
    assert np.allclose(sigma_mean, np.exp(norm.mean))


def test_denormalize_return_log_iv_matches_linear_transform() -> None:
    shape = (3, 3)
    iv, _, _ = _make_stack(shape, n=50, seed=12)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    z = norm.normalize(iv)
    expected = z * norm.std + norm.mean
    log_iv = norm.denormalize(z, return_log_iv=True)
    assert np.allclose(log_iv, expected, atol=1e-12)


def test_denormalize_surface_standalone_matches_method() -> None:
    shape = (5, 6)
    iv, _, _ = _make_stack(shape, n=128, seed=13)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    z = norm.normalize(iv)
    fn_out = denormalize_surface(z, norm.mean, norm.std)
    meth_out = norm.denormalize(z)
    assert np.allclose(fn_out, meth_out, atol=1e-12)


def test_normalize_surface_standalone_roundtrip() -> None:
    shape = (2, 3)
    iv, _, _ = _make_stack(shape, n=40, seed=14)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    z = normalize_surface(iv, norm.mean, norm.std)
    back = denormalize_surface(z, norm.mean, norm.std)
    assert np.allclose(back, iv, atol=1e-12)


def test_denormalize_propagates_nan() -> None:
    shape = (2, 2)
    iv, _, _ = _make_stack(shape, n=20, seed=15)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    z = np.zeros(shape)
    z[0, 0] = np.nan
    out = norm.denormalize(z)
    assert np.isnan(out[0, 0])
    assert np.all(np.isfinite(out[1:, :]))


def test_denormalize_surface_batch_broadcast() -> None:
    shape = (4, 4)
    iv, _, _ = _make_stack(shape, n=32, seed=16)
    norm = SurfaceNormalizer(grid_shape=shape).fit(iv)
    batch = np.stack([norm.normalize(iv[i]) for i in range(iv.shape[0])])
    out = denormalize_surface(batch, norm.mean, norm.std)
    assert out.shape == iv.shape
    assert np.allclose(out, iv, atol=1e-12)
