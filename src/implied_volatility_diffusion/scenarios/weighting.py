"""VolGAN exponential scenario weights from arbitrage penalties."""

import numpy as np
import torch


def volgan_exponential_weights(penalties: np.ndarray, beta: float) -> np.ndarray:
    """Normalized weights ``w_i ∝ exp(-β Φ_i)``."""
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    phi = np.asarray(penalties, dtype=float).reshape(-1)
    if phi.size == 0:
        raise ValueError("penalties must be non-empty")
    if beta == 0.0:
        return np.full(phi.shape, 1.0 / phi.size, dtype=float)
    log_w = -beta * phi
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def adaptive_beta(weights: np.ndarray, scale: float = 500.0) -> float:
    """VolGAN adaptive β = scale / max(w)."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    w_max = float(np.max(w))
    if w_max <= 0.0:
        return 0.0
    return scale / w_max


def relative_entropy(weights: np.ndarray) -> float:
    """KL divergence of weights from the uniform distribution."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = w.size
    if n <= 1:
        return 0.0
    safe_w = np.where(w > 0.0, w, 1.0)
    log_w = np.where(w > 0.0, np.log(safe_w), 0.0)
    return float(np.log(n) + np.sum(w * log_w))


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / Σ w_i²."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    w2_sum = float(np.sum(w**2))
    if w2_sum <= 0.0:
        return 0.0
    return 1.0 / w2_sum


def fraction_arbitrage_free(penalties: np.ndarray, tol: float = 0.0) -> float:
    """Fraction of scenarios with penalty ≤ ``tol``."""
    phi = np.asarray(penalties, dtype=float).reshape(-1)
    if phi.size == 0:
        return 0.0
    return float(np.mean(phi <= tol))


def volgan_exponential_weights_torch(penalties: torch.Tensor, beta: float) -> torch.Tensor:
    """Torch ``softmax(-β Φ)`` variant of :func:`volgan_exponential_weights`."""
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    phi = penalties.reshape(-1)
    if phi.numel() == 0:
        raise ValueError("penalties must be non-empty")
    if beta == 0.0:
        return torch.full_like(phi, 1.0 / phi.numel())
    return torch.softmax(-beta * phi, dim=0)
