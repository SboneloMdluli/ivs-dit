"""Tests for the U-Net's previous-IVS conditioning path.
"""

import pytest
import torch

from implied_volatility_diffusion.diffusion.backbones.unet import UNet


def _make_unet(*, cond_channels: int) -> UNet:
    return UNet(
        in_channels=1,
        out_channels=1,
        cond_channels=cond_channels,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_embed_dim=32,
        attention_levels=(),
        attention_heads=1,
    )


def test_unconditional_unet_ignores_cond_and_preserves_shape() -> None:
    torch.manual_seed(0)
    # Create an unconditional U-Net with no conditioning channels
    unet = _make_unet(cond_channels=0).eval()
    x = torch.randn(2, 1, 16, 24)
    t = torch.randint(0, 1000, (2,))

    with torch.no_grad():
        y_no_cond = unet(x, t)
        y_with_cond = unet(x, t, cond=torch.randn(2, 1, 16, 24))

    assert y_no_cond.shape == x.shape
    assert torch.allclose(y_no_cond, y_with_cond)


def test_conditional_unet_uses_cond_and_preserves_shape() -> None:
    torch.manual_seed(0)
    # Create a conditional U-Net with one conditioning channel
    unet = _make_unet(cond_channels=1).eval()
    x = torch.randn(2, 1, 16, 24)
    t = torch.randint(0, 1000, (2,))
    cond_a = torch.randn(2, 1, 16, 24)
    cond_b = cond_a + torch.randn_like(cond_a)

    with torch.no_grad():
        y_a = unet(x, t, cond=cond_a)
        y_b = unet(x, t, cond=cond_b)

    assert y_a.shape == x.shape
    assert torch.isfinite(y_a).all()
    # Different conditioning surfaces must produce different denoiser outputs;
    # otherwise the concat path is dead.
    assert not torch.allclose(y_a, y_b)



def test_conditional_unet_rejects_mismatched_spatial_shape() -> None:
    unet = _make_unet(cond_channels=1).eval()
    x = torch.randn(2, 1, 16, 24)
    t = torch.zeros(2, dtype=torch.long)
    bad_cond = torch.randn(2, 1, 16, 25)

    with pytest.raises(ValueError, match="spatial dims"):
        unet(x, t, cond=bad_cond)


def test_conditional_unet_rejects_mismatched_channel_count() -> None:
    unet = _make_unet(cond_channels=1).eval()
    x = torch.randn(2, 1, 16, 24)
    t = torch.zeros(2, dtype=torch.long)
    bad_cond = torch.randn(2, 2, 16, 24)

    with pytest.raises(ValueError, match="channel dim"):
        unet(x, t, cond=bad_cond)


def test_conditional_unet_rejects_non_4d_cond() -> None:
    unet = _make_unet(cond_channels=1).eval()
    x = torch.randn(2, 1, 16, 24)
    t = torch.zeros(2, dtype=torch.long)
    bad_cond = torch.randn(2, 16, 24)

    with pytest.raises(ValueError, match="4D"):
        unet(x, t, cond=bad_cond)


def test_conditional_unet_gradients_flow_to_cond() -> None:
    """Sanity check that ``cond`` is part of the autograd graph."""
    torch.manual_seed(0)
    unet = _make_unet(cond_channels=1)
    x = torch.randn(2, 1, 16, 24)
    t = torch.zeros(2, dtype=torch.long)
    cond = torch.randn(2, 1, 16, 24, requires_grad=True)

    y = unet(x, t, cond=cond)
    y.sum().backward()

    assert cond.grad is not None
    assert torch.isfinite(cond.grad).all()
    assert cond.grad.abs().sum() > 0
