"""Rotary position embeddings used by the Qwen2 decoder."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    """Rotate the final dimension by swapping halves and negating the first."""
    if values.ndim == 0 or values.shape[-1] % 2:
        raise ValueError("RoPE requires an even final dimension")
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Generate and apply Qwen-style RoPE to query and key head tensors.

    Query/key input layout is ``(batch, heads, sequence, head_dim)``.
    Position IDs have shape ``(batch, sequence)``. Values are not rotated.
    """

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.head_dim = head_dim
        self.theta = theta
        inverse_frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inverse_frequencies, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape (batch, heads, sequence, head_dim)")
        if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
            raise ValueError("query and key must share batch, sequence, and head dimensions")
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"expected head dimension {self.head_dim}, got {query.shape[-1]}")
        if position_ids.shape != (query.shape[0], query.shape[2]):
            raise ValueError("position_ids must have shape (batch, sequence)")

        # Keep position/frequency math in FP32, as in the Qwen reference.
        positions = position_ids.to(device=query.device, dtype=torch.float32)
        frequencies = positions.unsqueeze(-1) * self.inv_freq.to(query.device)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().to(query.dtype).unsqueeze(1)
        sin = embedding.sin().to(query.dtype).unsqueeze(1)

        query_rotated = query * cos + rotate_half(query) * sin
        key_rotated = key * cos.to(key.dtype) + rotate_half(key) * sin.to(key.dtype)
        return query_rotated, key_rotated
