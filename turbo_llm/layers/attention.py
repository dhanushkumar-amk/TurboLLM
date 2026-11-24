"""Grouped-query causal self-attention for Qwen-shaped head tensors."""

from __future__ import annotations

import torch
from torch import nn


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat KV heads into query-head groups: (B, Hkv, S, D) -> (B, Hq, S, D)."""
    if hidden_states.ndim != 4:
        raise ValueError("KV tensor must have shape (batch, heads, sequence, head_dim)")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


class GroupedQueryAttention(nn.Module):
    """Causal attention over projected Q/K/V tensors.

    Inputs use ``(batch, heads, sequence, head_dim)``. Query heads are
    grouped evenly over key/value heads, as in grouped-query attention.
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        dropout: float = 0.0,
        backend: str = "eager",
    ) -> None:
        super().__init__()
        if min(num_attention_heads, num_key_value_heads, head_dim) <= 0:
            raise ValueError("head counts and head_dim must be positive")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("query head count must be divisible by KV head count")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if backend not in {"eager", "sdpa"}:
            raise ValueError("backend must be 'eager' or 'sdpa'")
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.dropout = dropout
        self.backend = backend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Q, K, and V must each have shape (batch, heads, sequence, head_dim)")
        batch, query_heads, query_length, dim = query.shape
        key_batch, kv_heads, key_length, key_dim = key.shape
        if (key_batch, kv_heads, key_dim) != (
            batch, self.num_key_value_heads, self.head_dim
        ):
            raise ValueError("K shape does not match configured batch, KV heads, or head_dim")
        if value.shape != key.shape:
            raise ValueError("K and V must have the same shape")
        if (query_heads, dim) != (self.num_attention_heads, self.head_dim):
            raise ValueError("Q shape does not match configured query heads or head_dim")
        if query_length > key_length:
            raise ValueError("query length cannot exceed key/value length")

        # Supports a full prompt (Q and K same length) and incremental decoding
        # where Q is the final token and K/V contain the complete cache.
        query_positions = torch.arange(
            key_length - query_length, key_length, device=query.device
        )
        key_positions = torch.arange(key_length, device=query.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        causal_mask = causal_mask[None, None, :, :]

        if self.backend == "sdpa":
            sdpa_mask: torch.Tensor = causal_mask
            if attention_mask is not None:
                additive_causal = torch.zeros_like(causal_mask, dtype=query.dtype)
                additive_causal = additive_causal.masked_fill(~causal_mask, float("-inf"))
                sdpa_mask = additive_causal + attention_mask
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                enable_gqa=True,
            )
            return output.transpose(1, 2).contiguous()

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = nn.functional.dropout(
            probabilities, p=self.dropout, training=self.training
        )
        return torch.matmul(probabilities, value).transpose(1, 2).contiguous()
