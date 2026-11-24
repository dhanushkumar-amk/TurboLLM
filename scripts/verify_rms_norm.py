"""Compare TurboLLM RMSNorm with the pinned Hugging Face Qwen2 reference."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.layers.rms_norm import RMSNorm
from turbo_llm.shapes import QWEN25_05B


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    reference = reference_model.model.layers[0].input_layernorm

    layer = RMSNorm(QWEN25_05B.hidden_size, eps=reference.variance_epsilon).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        layer.weight.copy_(reference.weight)

    torch.manual_seed(7)
    inputs = torch.randn(2, 5, QWEN25_05B.hidden_size, device=device, dtype=dtype)
    with torch.inference_mode():
        actual = layer(inputs)
        expected = reference(inputs)
    difference = (actual.float() - expected.float()).abs()

    print(f"Device: {device}")
    print(f"Shape: {tuple(inputs.shape)}")
    print(f"Max absolute error: {difference.max().item():.8g}")
    print(f"Mean absolute error: {difference.mean().item():.8g}")
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    print("RMSNorm matches the Hugging Face Qwen2 reference within tolerance.")


if __name__ == "__main__":
    main()
