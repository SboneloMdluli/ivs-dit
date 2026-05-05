import pytest
import torch

from implied_volatility_diffusion.diffusion.backbones.transformer import GridTransformer
from implied_volatility_diffusion.diffusion.backbones import build_backbone, iter_backbone_names

def _make_transformer(
        *, cond_channels: int = 0) -> GridTransformer:
    return GridTransformer(
        grid_shape=(6, 8),
        in_channels=1,
        out_channels=1,
        cond_channels=cond_channels,
        d_model=32,
        nhead=4,
        layers=2,
        dim_feedforward=64,
        dropout=0.1,
    )

def test_grid_transformer_preserved_surface_shape() -> None:
    torch.manual_seed(0)
    model = _make_transformer().eval()
    x = torch.randn(3, 1, 6, 8)
    t = torch.randint(0, 100, (3,))

    with torch.no_grad():
        y = model(x, t)

    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert torch.isfinite(y).all(), "Output contains non-finite values"

def test_grid_transformer_registry() -> None:
    assert "transformer" in iter_backbone_names(), "Transformer backbone not registered"
    assert "grid_transformer" in iter_backbone_names(), "GridTransformer backbone not registered"

    model = build_backbone(
        "transformer",
        {
            "grid_shape": (6, 8),
            "d_model": 32,
            "nhead": 4,
            "layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.1,
        }
    )

    x = torch.randn(2, 1, 6, 8)
    t = torch.zeros(2, dtype=torch.long)

    assert model(x, t).shape == x.shape, "Registry-built model output shape mismatch"

def test_conditional_grid_transformer() -> None:
    torch.manual_seed(0)
    
    model = _make_transformer(cond_channels=1).eval()
    x = torch.randn(2, 1, 6, 8)
    t = torch.randint(0, 100, (2,))
    cond_a = torch.randn(2, 1, 6, 8)
    cond_b = cond_a + torch.randn_like(cond_a)

    with torch.no_grad():
        y_a = model(x, t, cond_a)
        y_b = model(x, t, cond_b)

    assert y_a.shape == x.shape, f"Expected output shape {x.shape}, got {y_a.shape}"
    assert torch.isfinite(y_a).all(), "Output contains non-finite values"
    assert not torch.allclose(y_a, y_b), "Outputs should differ for different conditioning inputs"

def test_conditional_grid_transformer_rejects_missing_cond() -> None:
    model = _make_transformer(cond_channels=1).eval()
    x = torch.randn(2, 1, 6, 8)
    t = torch.randint(0, 100, (2,), dtype=torch.long)

    with pytest.raises(ValueError, match="sparse dims."):
        model(x, t, cond=torch.randn(2, 1, 6, 7))

    with pytest.raises(ValueError, match="channel dimensions."):
        model(x, t, cond=torch.randn(2, 2, 6, 8))

    with pytest.raises(ValueError, match="4 dimensions."):
        model(x, t, cond=torch.randn(2, 1, 6))

def test_conditional_grid_transformer_gradients() -> None:
    torch.manual_seed(0)
    
    model = _make_transformer(cond_channels=1)
    x = torch.randn(2, 1, 6, 8, requires_grad=True)
    t = torch.zeros(2, dtype=torch.long)
    cond = torch.randn(2, 1, 6, 8, requires_grad=True)

    y = model(x, t, cond)
    y.sum().backward()

    assert cond.grad is not None, "Gradients not flowing to conditioning input"
    assert torch.isfinite(cond.grad).all(), "Conditioning gradients contain non-finite values"
    assert cond.grad.abs().sum() > 0, "Conditioning gradients are zero, expected non-zero values"