"""Profile one fixed prefill-and-decode workload with torch.profiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.generation import generate
from turbo_llm.model import QwenModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = QwenModel.from_pretrained(dtype=dtype, device=device, attention_backend="sdpa")
    prompt = torch.tensor([[151644, 872, 198, 3838, 374, 264, 5488, 13]], device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=(torch.profiler.tensorboard_trace_handler(str(args.trace))
                        if args.trace else None),
    ) as profiler:
        generate(model, prompt, max_new_tokens=args.tokens, eos_token_id=None)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=20))
    if args.trace:
        print(f"Trace written to {args.trace.resolve()}")


if __name__ == "__main__":
    main()
