"""OpenAI-compatible streaming benchmark for TurboLLM and vLLM endpoints.

Example: python scripts/benchmark_serving.py --turbo-url http://localhost:8000 --vllm-url http://localhost:8001 --requests 40 --concurrency 1,4,8
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


async def request_one(name: str, base: str, model: str, prompt: str, output_tokens: int,
                      semaphore: asyncio.Semaphore, run_start: float) -> dict:
    async with semaphore:
        started = time.perf_counter()
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": output_tokens, "temperature": 0,
                           "stream": True, "stream_options": {"include_usage": True}}).encode()
        req = Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = await asyncio.to_thread(urlopen, req, timeout=600)
            first = None
            token_times: list[float] = []
            completion_tokens = 0
            prompt_tokens = 0
            chunks = 0
            while True:
                line = await asyncio.to_thread(response.readline)
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    item = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = item.get("usage") or {}
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                choices = item.get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if delta:
                    now = time.perf_counter()
                    first = now if first is None else first
                    token_times.append(now)
                    chunks += 1
            ended = time.perf_counter()
            ttft = (first - started) if first is not None else None
            gaps = [(b - a) for a, b in zip(token_times, token_times[1:])]
            return {"engine": name, "ok": True, "start_offset_s": started - run_start,
                    "ttft_s": ttft, "e2e_s": ended - started,
                    "itl_mean_s": statistics.mean(gaps) if gaps else None,
                    "itl_p95_s": percentile(gaps, .95) if gaps else None,
                    "output_tokens": completion_tokens or chunks,
                    "prompt_tokens": prompt_tokens, "error": None}
        except Exception as exc:
            return {"engine": name, "ok": False, "start_offset_s": started - run_start,
                    "ttft_s": None, "e2e_s": time.perf_counter() - started,
                    "itl_mean_s": None, "itl_p95_s": None, "output_tokens": 0,
                    "prompt_tokens": 0, "error": str(exc)}


def stats(rows: list[dict], ttft_slo: float, itl_slo: float) -> dict:
    ok = [r for r in rows if r["ok"]]
    def distribution(key: str) -> dict:
        values = [r[key] * 1000 for r in ok if r[key] is not None]
        return {f"p{p}": round(percentile(values, p / 100), 3) if values else None
                for p in (50, 95, 99)}
    duration = max((r["start_offset_s"] + r["e2e_s"] for r in rows), default=0)
    out_tokens = sum(r["output_tokens"] for r in ok)
    prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    good = [r for r in ok if r["ttft_s"] is not None and r["ttft_s"] < ttft_slo
            and (r["itl_mean_s"] is None or r["itl_mean_s"] < itl_slo)]
    return {"requests": len(rows), "completed": len(ok), "errors": len(rows) - len(ok),
            "ttft_ms": distribution("ttft_s"), "e2e_ms": distribution("e2e_s"),
            "itl_mean_ms": distribution("itl_mean_s"), "itl_p95_ms": distribution("itl_p95_s"),
            "output_tokens_per_sec": round(out_tokens / duration, 3) if duration else 0,
            "requests_per_sec": round(len(ok) / duration, 3) if duration else 0,
            "goodput_requests_per_sec": round(len(good) / duration, 3) if duration else 0,
            "goodput_requests": len(good), "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens, "duration_s": round(duration, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbo-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--turbo-model", default="turbo-llm")
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain how a KV cache improves autoregressive inference in three concise bullet points.")
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--itl-slo-ms", type=float, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    args = parser.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]
    if args.requests < 1 or any(c < 1 for c in levels):
        parser.error("requests and concurrency levels must be positive")
    async def run():
        results = []
        for level in levels:
            for name, url, model in (("TurboLLM", args.turbo_url, args.turbo_model),
                                     ("vLLM", args.vllm_url, args.vllm_model)):
                start = time.perf_counter()
                sem = asyncio.Semaphore(level)
                rows = await asyncio.gather(*(request_one(name, url, model, args.prompt,
                                                          args.output_tokens, sem, start)
                                              for _ in range(args.requests)))
                results.append({"engine": name, "concurrency": level,
                                "summary": stats(rows, args.ttft_slo_ms / 1000,
                                                 args.itl_slo_ms / 1000), "requests": rows})
        return results
    results = asyncio.run(run())
    hardware = {}
    try:
        import torch
        hardware = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        hardware = {"torch_version": None, "cuda_available": False, "gpu": None}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "hardware": hardware,
              "workload": {"requests_per_engine_per_level": args.requests,
                           "concurrency": levels, "output_tokens": args.output_tokens,
                           "prompt": args.prompt, "ttft_slo_ms": args.ttft_slo_ms,
                           "itl_slo_ms": args.itl_slo_ms,
                           "endpoints": {"TurboLLM": args.turbo_url, "vLLM": args.vllm_url}},
              "runs": results,
              "notes": ["Endpoint streaming measurements include server queueing and HTTP overhead.",
                        "ITL is derived from non-empty streamed text chunks; runtimes may emit multiple tokens per chunk.",
                        "GPU counters, cost, KV hit rate, MFU, and quality require separate instrumentation/evaluation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["engine", "concurrency", "request", "ok",
            "ttft_s", "e2e_s", "itl_mean_s", "itl_p95_s", "output_tokens", "prompt_tokens", "error"])
        writer.writeheader()
        for run in results:
            for i, row in enumerate(run["requests"], 1):
                writer.writerow({"concurrency": run["concurrency"], "request": i, **row})
    print(json.dumps({"output": str(args.output), "csv": str(csv_path),
                      "runs": [{"engine": r["engine"], "concurrency": r["concurrency"],
                                **r["summary"]} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
