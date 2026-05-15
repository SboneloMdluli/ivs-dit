"""Tests for diffusion training timestep sampling and epsilon MSE weights."""

import torch

from implied_volatility_diffusion.diffusion.losses import (
    DiffusionLoss,
    DiffusionLossConfig,
    _sample_timesteps_lognormal,
)
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler


def test_default_config_uses_lognormal_low_noise_preset() -> None:
    sched = VPNoiseScheduler(timesteps=400, beta_schedule="cosine")
    cfg = DiffusionLossConfig()
    assert cfg.timestep_sampling == "lognormal"
    assert cfg.lognormal_mu == -3.5
    assert cfg.lognormal_max_noise_frac == 0.15
    assert cfg.eps_loss_schedule == "alpha_bar"
    loss = DiffusionLoss(config=cfg)
    uniform = DiffusionLoss(config=DiffusionLossConfig(timestep_sampling="uniform"))
    n = 2048
    t_def = loss.sample_timesteps(n, sched, device=torch.device("cpu")).float().mean()
    t_uni = uniform.sample_timesteps(n, sched, device=torch.device("cpu")).float().mean()
    assert t_def < t_uni


def test_sample_timesteps_uniform_in_range() -> None:
    sched = VPNoiseScheduler(timesteps=400)
    loss = DiffusionLoss(config=DiffusionLossConfig(timestep_sampling="uniform"))
    t = loss.sample_timesteps(256, sched, device=torch.device("cpu"))
    assert t.shape == (256,)
    assert t.dtype == torch.long
    assert (t >= 0).all() and (t < sched.timesteps).all()


def test_lognormal_timesteps_bias_toward_low_t() -> None:
    sched = VPNoiseScheduler(timesteps=400, beta_schedule="cosine")
    uniform = DiffusionLoss(config=DiffusionLossConfig(timestep_sampling="uniform"))
    lognormal = DiffusionLoss(
        config=DiffusionLossConfig(
            timestep_sampling="lognormal",
            lognormal_mu=-2.0,
            lognormal_sigma=1.0,
            lognormal_max_noise_frac=None,
        )
    )
    n = 4096
    t_u = uniform.sample_timesteps(n, sched, device=torch.device("cpu")).float().mean()
    t_ln = lognormal.sample_timesteps(n, sched, device=torch.device("cpu")).float().mean()
    assert t_ln < t_u


def test_lognormal_noise_cap_reduces_mean_t() -> None:
    sched = VPNoiseScheduler(timesteps=400, beta_schedule="cosine")
    n = 4096
    t_open = (
        _sample_timesteps_lognormal(n, sched, device=torch.device("cpu"), mu=-2.0, sigma=1.0, max_noise_frac=None)
        .float()
        .mean()
    )
    t_capped = (
        _sample_timesteps_lognormal(n, sched, device=torch.device("cpu"), mu=-2.0, sigma=1.0, max_noise_frac=0.15)
        .float()
        .mean()
    )
    assert t_capped < t_open


def test_lognormal_helper_respects_schedule_bounds() -> None:
    sched = VPNoiseScheduler(timesteps=100)
    t = _sample_timesteps_lognormal(
        64,
        sched,
        device=torch.device("cpu"),
        mu=-1.5,
        sigma=0.8,
    )
    assert (t >= 0).all() and (t < sched.timesteps).all()


def test_eps_loss_alpha_bar_upweights_near_x0() -> None:
    sched = VPNoiseScheduler(timesteps=400)
    loss = DiffusionLoss(config=DiffusionLossConfig(eps_loss_schedule="alpha_bar"))
    t_low = torch.tensor([0, 1, 2], dtype=torch.long)
    t_high = torch.tensor([397, 398, 399], dtype=torch.long)
    w_low = loss._eps_loss_weights(sched, t_low, "epsilon")
    w_high = loss._eps_loss_weights(sched, t_high, "epsilon")
    assert (w_low > w_high).all()


def test_alpha_bar_sq_emphasizes_low_noise_more_than_alpha_bar() -> None:
    sched = VPNoiseScheduler(timesteps=400, beta_schedule="cosine")
    t_low = torch.tensor([5], dtype=torch.long)
    t_high = torch.tensor([350], dtype=torch.long)
    loss_ab = DiffusionLoss(config=DiffusionLossConfig(eps_loss_schedule="alpha_bar"))
    loss_sq = DiffusionLoss(config=DiffusionLossConfig(eps_loss_schedule="alpha_bar_sq"))
    ratio_ab = (
        loss_ab._eps_loss_weights(sched, t_low, "epsilon")[0]
        / loss_ab._eps_loss_weights(sched, t_high, "epsilon")[0]
    )
    ratio_sq = (
        loss_sq._eps_loss_weights(sched, t_low, "epsilon")[0]
        / loss_sq._eps_loss_weights(sched, t_high, "epsilon")[0]
    )
    assert ratio_sq > ratio_ab
