"""Run a small, pinned Hugging Face reference generation for TurboLLM."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "Explain what a key-value cache does in one short sentence."
MAX_NEW_TOKENS = 48


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this reference run expects the configured NVIDIA GPU.")

    device = torch.device("cuda:0")
    dtype = torch.float16
    print(f"Model: {MODEL_ID}@{MODEL_REVISION}")
    print(f"Device: {torch.cuda.get_device_name(device)} ({dtype})")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started

    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    print(f"Prompt: {PROMPT}")
    print(f"Output: {tokenizer.decode(new_tokens, skip_special_tokens=True).strip()}")
    print(f"Load time: {load_seconds:.2f}s")
    print(f"Generation time: {generation_seconds:.2f}s")
    print(f"Generated tokens: {new_tokens.numel()}")
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated(device) / 1024**2:.0f} MiB")


if __name__ == "__main__":
    main()
