"""

"""

from typing import Any, Mapping

import torch
from torch import nn

from implied_volatility_diffusion.diffusion.backbones.base import (
    DenoisingBackbone,
    register_backbone,
)

from implied_volatility_diffusion.diffusion.backbones.embeddings import TimeEmbeddingMLP

class GridTransformer(DenoisingBackbone):
    """Transformer-based backbone for diffusion models."""

    def __init__(
            self,
            *,
            grid_shape: tuple[int, int] = (41, 40),
            in_channels: int = 1,
            out_channels: int = 1,
            cond_channels: int = 0,
            d_model: int = 128,
            nhead: int = 4,
            layers: int = 4,
            dim_feedforward: int | None = None,
            dropout: float = 0.2,
            time_embedding_dim: int | None = None,
    ) -> None:
        """Initialize the GridTransformer backbone."""
        super().__init__()

        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.cond_channels = int(cond_channels)
        self.num_tokens = self.grid_shape[0] * self.grid_shape[1]

        # Factorized 2D positional encoding: row (moneyness) + col (tau).
        # Token at grid position (i, j) receives row_embed[i] + col_embed[j],
        # giving the model an explicit 2D inductive bias for IV surface structure.
        self.row_embed = nn.Parameter(torch.zeros(self.grid_shape[0], d_model))
        self.col_embed = nn.Parameter(torch.zeros(self.grid_shape[1], d_model))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        token_in_dim = self.in_channels + self.cond_channels
        self.input_proj = nn.Sequential(
            nn.Linear(token_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.time_embedding = TimeEmbeddingMLP(
            time_embedding_dim or d_model,
            hidden_dim=d_model,
        )

        self.time_proj = nn.Sequential(
            nn.Linear(self.time_embedding.out_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward or (d_model * 4),
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.out_channels),
        )

    def forward(
            self,
            x: torch.Tensor,
            time: torch.Tensor,
            cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the GridTransformer backbone."""
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError("Conditioning channels specified but no conditioning provided.")
            if cond.ndim != 4:
                raise ValueError(f"cond must have 4 dimensions, got {cond.ndim}.")
            if cond.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    f"cond spatial dims {tuple(cond.shape[-2:])} do not match x "
                    f"{tuple(x.shape[-2:])}; check for sparse dims."
                )
            if cond.shape[1] != self.cond_channels:
                raise ValueError(
                    f"cond has {cond.shape[1]} channel(s) but model expects "
                    f"{self.cond_channels} channel dimensions."
                )
            x = torch.cat([x, cond.to(dtype=x.dtype, device=x.device)], dim=1)

        b, _, h, w = x.shape
        values = x.flatten(2).transpose(1, 2)
        h_tokens = self.input_proj(values)
        # Build factorized 2D positional embedding: (H, W, d_model) → (1, H*W, d_model)
        pos_2d = (self.row_embed[:, None, :] + self.col_embed[None, :, :]).reshape(1, self.num_tokens, -1)
        h_tokens = h_tokens + pos_2d.to(dtype=h_tokens.dtype, device=h_tokens.device)
        h_tokens = h_tokens + self.time_proj(self.time_embedding(time).unsqueeze(1))
        out = self.out(self.encoder(h_tokens))

        return out.transpose(1, 2).reshape(b, self.out_channels, h, w)
    
def _build_transformer(cfg: Mapping[str, Any]) -> 'GridTransformer':
    """Build a GridTransformer from a configuration dictionary."""
    return GridTransformer(**dict(cfg))

register_backbone('transformer', _build_transformer)
register_backbone('grid_transformer', _build_transformer)

__all__ = ['GridTransformer']
    
    