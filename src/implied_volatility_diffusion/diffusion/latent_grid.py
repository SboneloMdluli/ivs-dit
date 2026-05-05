"""Latent spatial grid geometry for the LDM first stage (NumPy only)."""

from __future__ import annotations

import numpy as np


def halving_spatial_factor(num_downsample: int) -> int:
    """Return ``2 ** num_downsample`` (each stride-2 step halves H and W)."""
    if num_downsample < 1:
        raise ValueError("num_downsample must be >= 1")
    return int(2**num_downsample)


def latent_padded_hw(h: int, w: int, num_downsample: int) -> tuple[int, int]:
    """Smallest ``(H', W')`` ≥ ``(h, w)`` that are multiples of the halving factor."""
    f = halving_spatial_factor(num_downsample)
    ph = ((int(h) + f - 1) // f) * f
    pw = ((int(w) + f - 1) // f) * f
    return ph, pw


def latent_spatial_hw(h: int, w: int, num_downsample: int) -> tuple[int, int]:
    """Latent map size ``(H_lat, W_lat)`` after ``num_downsample`` halvings."""
    ph, pw = latent_padded_hw(h, w, num_downsample)
    f = halving_spatial_factor(num_downsample)
    return ph // f, pw // f


def symmetric_pad_widths(
    h: int,
    w: int,
    *,
    multiple_h: int,
    multiple_w: int,
) -> tuple[int, int, int, int]:
    """``(pad_top, pad_bottom, pad_left, pad_right)`` for ``np.pad`` / ``F.pad``."""
    ph = (-int(h)) % int(multiple_h)
    pw = (-int(w)) % int(multiple_w)
    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


def pad_surface(
    x: np.ndarray,
    num_downsample: int,
    *,
    constant_values: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    """Pad last two axes so ``H, W`` are multiples of ``2**num_downsample``."""
    if x.ndim < 2:
        raise ValueError("x must have at least two dimensions (…, H, W)")
    f = halving_spatial_factor(num_downsample)
    orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
    pads = symmetric_pad_widths(orig_h, orig_w, multiple_h=f, multiple_w=f)
    pad_top, pad_bottom, pad_left, pad_right = pads
    npad = [(0, 0)] * (x.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
    out = np.pad(x, npad, mode="constant", constant_values=constant_values)
    return out, pads, (orig_h, orig_w)


def crop_surface(
    x: np.ndarray,
    pads: tuple[int, int, int, int],
    *,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Undo symmetric padding on the last two axes."""
    pad_top, _pad_bottom, pad_left, _pad_right = pads
    sl = (..., slice(pad_top, pad_top + int(target_h)), slice(pad_left, pad_left + int(target_w)))
    return x[sl]


__all__ = [
    "crop_surface",
    "halving_spatial_factor",
    "latent_padded_hw",
    "latent_spatial_hw",
    "pad_surface",
    "symmetric_pad_widths",
]
