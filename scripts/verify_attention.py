"""Compare TurboLLM GQA with the pinned Qwen2 attention layer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.layers.attention import GroupedQueryAttention
from turbo_llm.layers.rope import RotaryEmbedding
from turbo_llm.shapes import QWEN25_05B


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    reference = model.model.layers[0].self_attn
    config = model.config

    batch, sequence = 2, 5
    hidden = torch.randn(batch, sequence, QWEN25_05B.hidden_size, device=device, dtype=dtype)
    query = reference.q_proj(hidden).view(
        batch, sequence, QWEN25_05B.num_attention_heads, QWEN25_05B.head_dim
    ).transpose(1, 2)
    key = reference.k_proj(hidden).view(
        batch, sequence, QWEN25_05B.num_key_value_heads, QWEN25_05B.head_dim
    ).transpose(1, 2)
    value = reference.v_proj(hidden).view(
        batch, sequence, QWEN25_05B.num_key_value_heads, QWEN25_05B.head_dim
    ).transpose(1, 2)

    positions = torch.arange(sequence, device=device).expand(batch, -1)
    rotary = model.model.rotary_emb
    cos, sin = rotary(hidden, positions)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    query, key = apply_rotary_pos_emb(query, key, cos.squeeze(1), sin.squeeze(1))

    implementation = GroupedQueryAttention(
        QWEN25_05B.num_attention_heads,
        QWEN25_05B.num_key_value_heads,
        QWEN25_05B.head_dim,
    ).to(device).eval()
    actual = implementation(query, key, value)

    # Run the reference attention in eager form, with the same rotated Q/K/V.
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    causal_mask = torch.full(
        (sequence, sequence), float("-inf"), device=device, dtype=dtype
    ).triu(diagonal=1)[None, None, :, :]
    expected, _ = eager_attention_forward(
        reference,
        query,
        key,
        value,
        attention_mask=causal_mask,
        scaling=reference.scaling,
        dropout=0.0,
    )
    actual_projected = reference.o_proj(actual.reshape(batch, sequence, -1))
    expected_projected = reference.o_proj(expected.reshape(batch, sequence, -1))
    difference = (actual_projected.float() - expected_projected.float()).abs()
    print(f"Device: {device}")
    print(f"Q shape: {tuple(query.shape)}; K/V shape: {tuple(key.shape)}")
    print(f"Attention output shape: {tuple(actual.shape)}")
    print(f"Max absolute error after output projection: {difference.max().item():.8g}")
    torch.testing.assert_close(actual_projected, expected_projected, rtol=1e-3, atol=1e-3)
    print("GQA output matches the Qwen2 eager-attention reference within tolerance.")


if __name__ == "__main__":
    main()
