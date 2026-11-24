"""Compare TurboLLM and Transformers on the same pinned Qwen prompt."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.generation import generate
from turbo_llm.model import MODEL_ID, MODEL_REVISION, QwenModel


def timed_hf(model, input_ids, count: int, use_cache: bool) -> tuple[float, list[int]]:
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=count,
            do_sample=False,
            use_cache=use_cache,
            repetition_penalty=1.0,
        )
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    return time.perf_counter() - started, output[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--cache", choices=("on", "off"), default="on")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain KV caching in one sentence."}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"].to(device)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    custom = QwenModel.from_pretrained(
        dtype=dtype, device=device, attention_backend="sdpa"
    )

    custom_times, hf_times = [], []
    custom_sequences, hf_sequences = [], []
    use_cache = args.cache == "on"
    for _ in range(args.runs + 1):
        custom_result = generate(
            custom, input_ids, max_new_tokens=args.new_tokens, eos_token_id=None,
            use_cache=use_cache,
        )
        hf_elapsed, hf_sequence = timed_hf(hf, input_ids, args.new_tokens, use_cache)
        if _:
            custom_times.append(custom_result.elapsed_seconds)
            hf_times.append(hf_elapsed)
            custom_sequences = custom_result.sequences[0].tolist()
            hf_sequences = hf_sequence

    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": str(device),
        "dtype": str(dtype),
        "kv_cache": use_cache,
        "prompt_tokens": int(input_ids.shape[1]),
        "new_tokens": args.new_tokens,
        "runs": args.runs,
        "warmups": 1,
        "median_total_seconds": {
            "transformers": statistics.median(hf_times),
            "turbo_llm_sdpa": statistics.median(custom_times),
        },
        "median_generated_tokens_per_second": {
            "transformers": args.new_tokens / statistics.median(hf_times),
            "turbo_llm_sdpa": args.new_tokens / statistics.median(custom_times),
        },
        "greedy_sequences_equal": hf_sequences == custom_sequences,
        "turbo_llm_text": tokenizer.decode(custom_sequences[input_ids.shape[1]:], skip_special_tokens=True),
        "transformers_text": tokenizer.decode(hf_sequences[input_ids.shape[1]:], skip_special_tokens=True),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
