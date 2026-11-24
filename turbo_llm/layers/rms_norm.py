"""RMSNorm as used by the Qwen2 decoder."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization over the final hidden dimension."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.weight.numel():
            raise ValueError(
                f"expected final dimension {self.weight.numel()}, got {tuple(hidden_states.shape)}"
            )

        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)
