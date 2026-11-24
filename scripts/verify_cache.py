"""Verify incremental KV-cache logits and generation against uncached runs."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.cache import PagedKVCache, estimate_kv_cache_bytes
from turbo_llm.generation import generate
from turbo_llm.model import QwenModel
from turbo_llm.scheduler import ContinuousBatchScheduler


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QwenModel.from_pretrained(dtype=torch.float16, device=device)
    prompt = torch.tensor([[151644, 872, 198, 3838, 374]], device=device)
    with torch.inference_mode():
        full_logits = model(prompt).logits[:, -1]
        prefill = model(prompt[:, :-1])
        cached_logits = model(prompt[:, -1:], prefill.past_key_values).logits[:, -1]
    torch.testing.assert_close(cached_logits, full_logits, rtol=5e-3, atol=3e-2)

    cached = generate(model, prompt, max_new_tokens=4, eos_token_id=None, use_cache=True)
    uncached = generate(model, prompt, max_new_tokens=4, eos_token_id=None, use_cache=False)
    if not torch.equal(cached.sequences, uncached.sequences):
        raise AssertionError("cached and uncached greedy token sequences differ")

    async def verify_continuous_cache_batch():
        scheduler = ContinuousBatchScheduler(
            model, max_batch_size=2, queue_limit=2, batch_window_seconds=0.01
        )
        prompts = [prompt, prompt[:, 1:]]
        scheduled_a, scheduled_b = await asyncio.gather(*[
            scheduler.submit(item, max_new_tokens=3, eos_token_id=None)
            for item in prompts
        ])
        expected_a = generate(model, prompts[0], max_new_tokens=3, eos_token_id=None).sequences
        expected_b = generate(model, prompts[1], max_new_tokens=3, eos_token_id=None).sequences
        torch.testing.assert_close(scheduled_a, expected_a)
        torch.testing.assert_close(scheduled_b, expected_b)
        return scheduler

    scheduler = asyncio.run(verify_continuous_cache_batch())

    pages = PagedKVCache(2, 3, 2, 2, 4)
    pages.create_request("a")
    for token in range(3):
        item = torch.full((2, 2, 4), token, dtype=torch.float16)
        pages.append("a", item, item + 10)
    keys, values = pages.read("a")
    assert keys.shape == (2, 2, 3, 4) and values.shape == keys.shape
    assert keys[0, 0, :, 0].tolist() == [0, 1, 2]
    pages.release("a")
    assert pages.free_count == 3

    estimate = estimate_kv_cache_bytes(
        num_layers=24, batch_size=1, num_key_value_heads=2,
        sequence_length=32768, head_dim=64, dtype=torch.float16,
    )
    print(f"Device: {device}; cached prompt logits match full-sequence logits.")
    print(f"Cached and uncached greedy output: {cached.sequences[0].tolist()}")
    print(f"Continuous batched KV-cache outputs match independent generation ({scheduler.requests_completed} requests).")
    print(f"Paged allocator append/read/release: passed; blocks free={pages.free_count}")
    print(f"Qwen KV cache at 32K, batch 1, FP16: {estimate / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
