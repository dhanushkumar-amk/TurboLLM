"""Compare eager GQA with PyTorch SDPA on the pinned Qwen workload."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.generation import generate
from turbo_llm.model import QwenModel


def set_backend(model: QwenModel, backend: str) -> None:
    for layer in model.model.layers:
        layer.attention.backend = backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, default=24)
    args = parser.parse_args()
    if args.runs < 1 or args.new_tokens < 1:
        parser.error("--runs and --new-tokens must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = QwenModel.from_pretrained(dtype=dtype, device=device)
    prompt = torch.tensor([[151644, 872, 198, 3838, 374, 264, 5488, 13]], device=device)

    timings: dict[str, list[float]] = {"eager": [], "sdpa": []}
    outputs = {}
    memory = {}
    for backend in ("eager", "sdpa"):
        set_backend(model, backend)
        # Warm-up is excluded from reported measurements.
        generate(model, prompt, max_new_tokens=2, eos_token_id=None)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.runs):
            result = generate(model, prompt, max_new_tokens=args.new_tokens, eos_token_id=None)
            timings[backend].append(result.elapsed_seconds)
            outputs[backend] = result.sequences[0].tolist()
        memory[backend] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )

    if outputs["eager"] != outputs["sdpa"]:
        raise AssertionError("eager and SDPA greedy outputs differ on the benchmark workload")
    medians = {backend: statistics.median(values) for backend, values in timings.items()}
    report = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": int(prompt.shape[1]),
        "new_tokens": args.new_tokens,
        "runs": args.runs,
        "warmup_runs_per_backend": 1,
        "median_seconds": medians,
        "tokens_per_second": {
            backend: args.new_tokens / duration for backend, duration in medians.items()
        },
        "peak_allocated_bytes": memory,
        "greedy_outputs_equal": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
