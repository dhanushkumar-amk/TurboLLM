"""Compare TurboLLM full-sequence logits with pinned Hugging Face Qwen2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.model import MODEL_ID, MODEL_REVISION, QwenModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FP32 isolates implementation correctness from FP16 accumulation/order noise.
    dtype = torch.float32
    reference = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()
    custom = QwenModel.from_pretrained(dtype=dtype, device=device)
    input_ids = torch.tensor([[151644, 872, 198, 3838, 374, 264, 5488, 13]], device=device)
    with torch.inference_mode():
        expected = reference(input_ids).logits
        actual = custom(input_ids).logits
    difference = (actual.float() - expected.float()).abs()
    print(f"Device: {device}; dtype: {dtype}")
    print(f"Logits shape: {tuple(actual.shape)}")
    print(f"Max absolute error: {difference.max().item():.8g}")
    print(f"Mean absolute error: {difference.mean().item():.8g}")
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    print("Full model logits match the Hugging Face reference within tolerance.")


if __name__ == "__main__":
    main()
