"""Qwen-style pre-normalized transformer decoder block."""

from __future__ import annotations

import torch
from torch import nn

from turbo_llm.layers.attention import GroupedQueryAttention
from turbo_llm.layers.mlp import SwiGLU
from turbo_llm.layers.rms_norm import RMSNorm
from turbo_llm.layers.rope import RotaryEmbedding


class DecoderLayer(nn.Module):
    """One decoder block with attention and MLP residual connections."""

    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        attention_bias: bool = True,
        attention_backend: str = "eager",
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias),
            "k_proj": nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias),
            "v_proj": nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias),
            "o_proj": nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False),
        })
        self.attention = GroupedQueryAttention(
            num_attention_heads, num_key_value_heads, head_dim,
            backend=attention_backend,
        )
        self.rotary = RotaryEmbedding(head_dim, rope_theta)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, sequence, _ = hidden_states.shape
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        query = self.self_attn["q_proj"](normalized).view(
            batch, sequence, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key = self.self_attn["k_proj"](normalized).view(
            batch, sequence, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.self_attn["v_proj"](normalized).view(
            batch, sequence, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query, key = self.rotary(query, key, position_ids)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        attention_output = self.attention(query, key, value, attention_mask=attention_mask)
        attention_output = attention_output.reshape(batch, sequence, -1)
        hidden_states = residual + self.self_attn["o_proj"](attention_output)
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, (key, value)
