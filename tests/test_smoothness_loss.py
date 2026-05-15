"""Tests for IV surface smoothness penalties in :class:`DiffusionLoss`."""

import torch

from implied_volatility_diffusion.diffusion.losses import _dirichlet_energy_index_mean_per_sample


def test_dirichlet_matches_first_order_only() -> None:
    iv = torch.randn(2, 1, 8, 8)
    d = _dirichlet_energy_index_mean_per_sample(iv[:, 0])
    assert d.shape == (2,)
    assert torch.all(d >= 0)
