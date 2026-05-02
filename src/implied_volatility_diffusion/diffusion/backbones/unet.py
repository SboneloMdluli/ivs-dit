"""Time-conditioned U-Net denoiser for the reverse diffusion process.

Design follows the *Annotated Diffusion* (Hugging Face) and Ho et al. (2020)
references, sampler can drop in
a different backbone (e.g. a ViT) by implementing
:class:`~implied_volatility_diffusion.diffusion.backbones.base.DenoisingBackbone`.

reference : https://huggingface.co/blog/annotated-diffusion
"""

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from implied_volatility_diffusion.diffusion.backbones.base import (
    DenoisingBackbone,
    register_backbone,
)
from implied_volatility_diffusion.diffusion.backbones.embeddings import TimeEmbeddingMLP


def _groupnorm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a divisor that's safe for small channel counts."""
    g = min(num_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class ResnetBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv ResNet block with FiLM-style time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _groupnorm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

        self.norm2 = _groupnorm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over flattened spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.norm = _groupnorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = int(num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads

        q = q.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        k = k.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        v = v.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)

        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(-1, -2).reshape(b, c, h, w)

        return x + self.proj(out)


def _avg_pool_down(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Avg-pool downsampling that records spatial size for upsampling."""
    size = (int(x.shape[-2]), int(x.shape[-1]))
    return F.avg_pool2d(x, kernel_size=2, ceil_mode=True), size


class _Upsample(nn.Module):
    """Nearest-neighbour upsampling to a recorded target size, then 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.conv(F.interpolate(x, size=size, mode="nearest"))


class UNet(DenoisingBackbone):
    """Time-conditioned U-Net for IV-surface diffusion.

    Args:
        in_channels: Channels of the noisy input (1 for a single surface).
        out_channels: Channels of the prediction target (matches ``in_channels``
            for ``epsilon`` / ``x0`` parameterisation).
        cond_channels: Channels of an optional conditioning surface that is
            concatenated to the noisy input on the channel axis before the
            input projection. ``0`` (default) means the backbone is fully
            unconditional. ``1`` is used by the next-day forecaster which
            conditions on the previous-day IV surface (in z-space).
        base_channels: Width of the first stage (multiplied by ``channel_mults``).
        channel_mults: Channel multipliers per resolution stage. Length controls
            the number of down/up sampling steps.
        num_res_blocks: ResNet blocks per stage.
        time_embed_dim: Width of the sinusoidal time embedding.
        attention_levels: Stage indices (0-based, into ``channel_mults``) that
            should include a self-attention block. Empty means no attention.
        attention_heads: Number of attention heads inside attention blocks.
        dropout: Dropout applied inside ResNet blocks.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 0,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_embed_dim: int = 256,
        attention_levels: Sequence[int] = (2,),
        attention_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not channel_mults:
            raise ValueError("channel_mults must be non-empty")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        if cond_channels < 0:
            raise ValueError("cond_channels must be >= 0")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.cond_channels = int(cond_channels)
        self.prediction_type = "epsilon"

        self.time_embedding = TimeEmbeddingMLP(time_embed_dim)
        t_dim = self.time_embedding.out_dim

        self.input_proj = nn.Conv2d(
            self.in_channels + self.cond_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        attn_levels = set(int(level) for level in attention_levels)
        channels = [base_channels * int(m) for m in channel_mults]
        self.num_res_blocks = int(num_res_blocks)
        self.num_levels = len(channels)
        self._channels = tuple(channels)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.do_down = []
        prev_c = base_channels
        for level, c in enumerate(channels):
            stage = nn.ModuleList()
            attn_stage = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(ResnetBlock(prev_c, c, time_dim=t_dim, dropout=dropout))
                attn_stage.append(
                    SelfAttention2d(c, num_heads=attention_heads) if level in attn_levels else nn.Identity()
                )
                prev_c = c
            self.down_blocks.append(stage)
            self.down_attn.append(attn_stage)
            self.do_down.append(level < len(channels) - 1)

        mid_c = channels[-1]
        self.mid_block1 = ResnetBlock(mid_c, mid_c, time_dim=t_dim, dropout=dropout)
        self.mid_attn = SelfAttention2d(mid_c, num_heads=attention_heads)
        self.mid_block2 = ResnetBlock(mid_c, mid_c, time_dim=t_dim, dropout=dropout)

        # Up path mirrors down: blocks consume one skip each, and we upsample
        # *after* the blocks at non-bottom levels so the next stage matches sizes.
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        cur_c = mid_c
        for level in reversed(range(self.num_levels)):
            stage_c = channels[level]
            stage = nn.ModuleList()
            attn_stage = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                in_c = cur_c + stage_c
                stage.append(ResnetBlock(in_c, stage_c, time_dim=t_dim, dropout=dropout))
                attn_stage.append(
                    SelfAttention2d(stage_c, num_heads=attention_heads) if level in attn_levels else nn.Identity()
                )
                cur_c = stage_c
            self.up_blocks.append(stage)
            self.up_attn.append(attn_stage)
            if level > 0:
                self.upsamples.append(_Upsample(cur_c))
            else:
                self.upsamples.append(nn.Identity())  # type: ignore[arg-type]

        self.out_norm = _groupnorm(cur_c)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(cur_c, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the diffusion target from a noisy IV surface.

        Args:
            x: ``(B, in_channels, H, W)`` noisy surface.
            t: ``(B,)`` long tensor of discrete timesteps.
            cond: ``(B, cond_channels, H, W)`` conditioning surface (e.g. the
                previous-day IV surface in z-space). Required when
                ``cond_channels > 0`` and must share the spatial grid of ``x``;
                ignored when ``cond_channels == 0``.
        """
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError(f"UNet was built with cond_channels={self.cond_channels} but `cond` was not provided")
            if cond.dim() != 4:
                raise ValueError(f"`cond` must be 4D (B, C, H, W); got shape {tuple(cond.shape)}")
            if cond.shape[0] != x.shape[0] or cond.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    f"`cond` must match `x` on batch and spatial dims; got cond={tuple(cond.shape)}, x={tuple(x.shape)}"
                )
            if cond.shape[1] != self.cond_channels:
                raise ValueError(
                    f"`cond` channel dim ({cond.shape[1]}) must equal cond_channels ({self.cond_channels})"
                )

            # Concatenate the conditioning surface to the noisy input
            # noisy surface, conditining
            x = torch.cat([x, cond.to(dtype=x.dtype, device=x.device)], dim=1)

        t_emb = self.time_embedding(t)
        h = self.input_proj(x)

        skips: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []
        for level in range(self.num_levels):
            for block, attn in zip(self.down_blocks[level], self.down_attn[level]):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            if self.do_down[level]:
                h, size = _avg_pool_down(h)
                sizes.append(size)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for stage_idx, level in enumerate(reversed(range(self.num_levels))):
            for block, attn in zip(self.up_blocks[stage_idx], self.up_attn[stage_idx]):
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            if level > 0:
                size = sizes.pop()
                h = self.upsamples[stage_idx](h, size)

        return self.out_conv(self.out_act(self.out_norm(h)))


def _build_unet(cfg: Mapping[str, Any]) -> UNet:
    return UNet(**dict(cfg))


register_backbone("unet", _build_unet)


__all__ = ["ResnetBlock", "SelfAttention2d", "UNet"]
