"""Tests for KL-regularized autoencoder."""

import pytest

torch = pytest.importorskip("torch")

from implied_volatility_diffusion.diffusion.kl_autoencoder import KLAutoencoder
from implied_volatility_diffusion.diffusion.latent_grid import latent_spatial_hw


def _make(double_z: bool = True) -> KLAutoencoder:
    return KLAutoencoder(
        in_channels=1,
        latent_channels=4,
        base_channels=16,
        num_downsample=2,
        double_z=double_z,
    )


def test_reconstruction_shape_arbitrary_grid() -> None:
    ae = _make()
    for h, w in [(11, 17), (16, 16), (3, 5)]:
        x = torch.randn(2, 1, h, w)
        y = ae(x)
        assert y.shape == x.shape, f"shape mismatch at ({h},{w}): {y.shape}"


def test_latent_shape_matches_helper() -> None:
    ae = _make()
    h, w = 15, 21
    hl, wl = latent_spatial_hw(h, w, ae.num_downsample)
    x = torch.randn(1, 1, h, w)
    z, mu, _ = ae.encode(x)
    assert z.shape == (1, 4, hl, wl)
    assert mu.shape == z.shape


def test_forward_return_output_fields() -> None:
    ae = _make()
    x = torch.randn(2, 1, 16, 16)
    out = ae(x, return_output=True)
    assert out.reconstruction.shape == x.shape
    assert out.z.shape[1] == 4
    assert out.posterior_mean.shape == out.z.shape
    assert out.rec_loss.ndim == 0
    assert out.kl_loss.ndim == 0
    assert torch.isfinite(out.rec_loss)
    assert torch.isfinite(out.kl_loss)
    assert out.kl_loss.item() >= 0.0


def test_total_loss_backward() -> None:
    ae = _make()
    x = torch.randn(1, 1, 12, 12)
    out = ae(x, return_output=True)
    loss = out.rec_loss + 1e-6 * out.kl_loss
    loss.backward()
    assert ae.enc_stem.weight.grad is not None
    assert ae.dec_out.weight.grad is not None


def test_deterministic_mode_no_kl() -> None:
    ae = _make(double_z=False)
    x = torch.randn(1, 1, 8, 8)
    out = ae(x, return_output=True)
    assert out.reconstruction.shape == x.shape
    assert out.kl_loss.item() == 0.0


def test_encode_decode_consistency() -> None:
    ae = _make()
    x = torch.randn(1, 1, 10, 14)
    z, _, pads = ae.encode(x)
    recon = ae.decode(z, orig_hw=(10, 14), pads=pads)
    assert recon.shape == x.shape


def test_invalid_input_channels() -> None:
    ae = _make()
    with pytest.raises(ValueError):
        ae.encode(torch.randn(1, 3, 8, 8))
