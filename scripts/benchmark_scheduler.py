"""Compare dynamic KV-cache batching with serial per-request generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.generation import generate
from turbo_llm.model import QwenModel
from turbo_llm.scheduler import ContinuousBatchScheduler


async def run_scheduler(model, prompts, count):
    scheduler = ContinuousBatchScheduler(
        model,
        max_batch_size=len(prompts),
        queue_limit=len(prompts),
        batch_window_seconds=0.005,
    )
    return await asyncio.gather(*[
        scheduler.submit(prompt, max_new_tokens=count, eos_token_id=None)
        for prompt in prompts
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    args = parser.parse_args()
    if min(args.requests, args.new_tokens, args.runs) < 1:
        parser.error("requests, new-tokens, and runs must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32
    model = QwenModel.from_pretrained(dtype=dtype, device=device, attention_backend="sdpa")
    torch.manual_seed(21)
    prompts = [
        torch.randint(100, 10_000, (1, 12 + index * 4), device=device)
        for index in range(args.requests)
    ]

    # Warm both paths once; report only measured repetitions.
    for prompt in prompts:
        generate(model, prompt, max_new_tokens=2, eos_token_id=None)
    asyncio.run(run_scheduler(model, prompts, 2))

    serial_times, scheduler_times = [], []
    for _ in range(args.runs):
        started = time.perf_counter()
        serial = [generate(model, prompt, max_new_tokens=args.new_tokens,
                           eos_token_id=None).sequences for prompt in prompts]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        serial_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        batched = asyncio.run(run_scheduler(model, prompts, args.new_tokens))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scheduler_times.append(time.perf_counter() - started)
        # Record token parity instead of hiding FP16 batch-shaped GEMM rounding.

    serial_median = statistics.median(serial_times)
    scheduler_median = statistics.median(scheduler_times)
    tokens = args.requests * args.new_tokens
    pairs_equal = [torch.equal(left, right) for left, right in zip(serial, batched)]
    common_prefix = []
    for left, right in zip(serial, batched):
        mismatch = next((index for index, (a, b) in enumerate(zip(left[0], right[0]))
                         if a != b), min(left.shape[1], right.shape[1]))
        common_prefix.append(mismatch - min(prompts[len(common_prefix)].shape[1], mismatch))
    print(json.dumps({
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "device": str(device),
        "dtype": str(dtype),
        "requests": args.requests,
        "prompt_lengths": [int(prompt.shape[1]) for prompt in prompts],
        "new_tokens_per_request": args.new_tokens,
        "runs": args.runs,
        "warmups": 1,
        "median_seconds": {"serial": serial_median, "continuous_batch": scheduler_median},
        "aggregate_tokens_per_second": {
            "serial": tokens / serial_median,
            "continuous_batch": tokens / scheduler_median,
        },
        "greedy_outputs_equal": all(pairs_equal),
        "common_generated_prefix_tokens": common_prefix,
    }, indent=2))


if __name__ == "__main__":
    main()
