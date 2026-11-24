"""Compare TurboLLM SwiGLU with the pinned Qwen2 MLP."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.layers.mlp import SwiGLU
from turbo_llm.model import MODEL_ID, MODEL_REVISION


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    reference = reference_model.model.layers[0].mlp
    layer = SwiGLU(896, 4864).to(device=device, dtype=dtype)
    layer.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 5, 896, device=device, dtype=dtype)
    with torch.inference_mode():
        expected, actual = reference(inputs), layer(inputs)
    error = (expected.float() - actual.float()).abs()
    print(f"Device: {device}; input shape: {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(actual.shape)}; max absolute error: {error.max().item():.8g}")
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print("SwiGLU matches the Hugging Face Qwen2 MLP within tolerance.")


if __name__ == "__main__":
    main()
