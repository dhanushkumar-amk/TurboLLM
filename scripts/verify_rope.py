"""Compare TurboLLM RoPE with the pinned Hugging Face Qwen2 implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.layers.rope import RotaryEmbedding
from turbo_llm.shapes import QWEN25_05B


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    batch, sequence = 2, 5
    query = torch.randn(
        batch, QWEN25_05B.num_attention_heads, sequence, QWEN25_05B.head_dim,
        device=device, dtype=dtype,
    )
    key = torch.randn(
        batch, QWEN25_05B.num_key_value_heads, sequence, QWEN25_05B.head_dim,
        device=device, dtype=dtype,
    )
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)

    theta = config.rope_parameters["rope_theta"]
    rope = RotaryEmbedding(QWEN25_05B.head_dim, theta=theta).to(device)
    actual_query, actual_key = rope(query, key, position_ids)

    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, QWEN25_05B.head_dim, 2, device=device, dtype=torch.float32)
            / QWEN25_05B.head_dim)
    )
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos, sin = embedding.cos().to(dtype), embedding.sin().to(dtype)
    expected_query, expected_key = apply_rotary_pos_emb(query, key, cos, sin)

    query_error = (actual_query.float() - expected_query.float()).abs()
    key_error = (actual_key.float() - expected_key.float()).abs()
    print(f"Device: {device}")
    print(f"Q shape: {tuple(query.shape)} -> {tuple(actual_query.shape)}")
    print(f"K shape: {tuple(key.shape)} -> {tuple(actual_key.shape)}")
    print(f"Max absolute error (Q): {query_error.max().item():.8g}")
    print(f"Max absolute error (K): {key_error.max().item():.8g}")
    torch.testing.assert_close(actual_query, expected_query, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_key, expected_key, rtol=1e-3, atol=1e-3)
    print("RoPE matches the Hugging Face Qwen2 reference within tolerance.")


if __name__ == "__main__":
    main()
