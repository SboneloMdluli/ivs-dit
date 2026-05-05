"""Tests for latent spatial grid helpers (NumPy only)."""

import numpy as np
import pytest

from implied_volatility_diffusion.diffusion.latent_grid import (
    crop_surface,
    halving_spatial_factor,
    latent_padded_hw,
    latent_spatial_hw,
    pad_surface,
)


def test_halving_spatial_factor() -> None:
    assert halving_spatial_factor(2) == 4


def test_halving_spatial_factor_invalid() -> None:
    with pytest.raises(ValueError, match="num_downsample"):
        halving_spatial_factor(0)


def test_pad_crop_roundtrip_2d() -> None:
    x = np.arange(35, dtype=np.float64).reshape(5, 7)
    xp, pads, (oh, ow) = pad_surface(x, num_downsample=2)
    assert oh == 5 and ow == 7
    ph, pw = latent_padded_hw(5, 7, 2)
    assert xp.shape == (ph, pw)
    back = crop_surface(xp, pads, target_h=oh, target_w=ow)
    np.testing.assert_array_equal(back, x)


def test_pad_crop_roundtrip_batched() -> None:
    x = np.random.default_rng(0).standard_normal((3, 2, 5, 11))
    xp, pads, (oh, ow) = pad_surface(x, num_downsample=2)
    back = crop_surface(xp, pads, target_h=oh, target_w=ow)
    np.testing.assert_allclose(back, x)


def test_latent_spatial_hw_matches_padded() -> None:
    hl, wl = latent_spatial_hw(15, 20, num_downsample=2)
    ph, pw = latent_padded_hw(15, 20, 2)
    f = halving_spatial_factor(2)
    assert (hl, wl) == (ph // f, pw // f)
