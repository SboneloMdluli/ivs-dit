"""Tests for diffusion training timestep sampling."""

import torch

from implied_volatility_diffusion.diffusion.losses import DiffusionLoss, DiffusionLossConfig
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler


def test_sample_timesteps_uniform_in_range() -> None:
    sched = VPNoiseScheduler(timesteps=400)
    loss = DiffusionLoss(config=DiffusionLossConfig())
    t = loss.sample_timesteps(256, sched, device=torch.device("cpu"))
    assert t.shape == (256,)
    assert t.dtype == torch.long
    assert (t >= 0).all() and (t < sched.timesteps).all()
