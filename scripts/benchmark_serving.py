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
            text_parts: list[str] = []
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
                    text_parts.append(delta)
                    chunks += 1
            ended = time.perf_counter()
            ttft = (first - started) if first is not None else None
            gaps = [(b - a) for a, b in zip(token_times, token_times[1:])]
            generated = completion_tokens or chunks
            tpot = (ended - first) / max(1, generated - 1) if first is not None and generated > 1 else None
            return {"engine": name, "ok": True, "start_offset_s": started - run_start,
                    "ttft_s": ttft, "e2e_s": ended - started,
                    "itl_mean_s": statistics.mean(gaps) if gaps else None,
                    "itl_p95_s": percentile(gaps, .95) if gaps else None,
                    "tpot_s": tpot, "output_tokens": generated,
                    "prompt_tokens": prompt_tokens, "text": "".join(text_parts), "error": None}
        except Exception as exc:
            return {"engine": name, "ok": False, "start_offset_s": started - run_start,
                    "ttft_s": None, "e2e_s": time.perf_counter() - started,
                    "itl_mean_s": None, "itl_p95_s": None, "tpot_s": None, "output_tokens": 0,
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
            and r["tpot_s"] is not None and r["tpot_s"] < itl_slo]
    return {"requests": len(rows), "completed": len(ok), "errors": len(rows) - len(ok),
            "ttft_ms": distribution("ttft_s"), "e2e_ms": distribution("e2e_s"),
            "tpot_ms": distribution("tpot_s"), "itl_mean_ms": distribution("itl_mean_s"),
            "itl_p95_ms": distribution("itl_p95_s"),
            "output_tokens_per_sec": round(out_tokens / duration, 3) if duration else 0,
            "requests_per_sec": round(len(ok) / duration, 3) if duration else 0,
            "goodput_requests_per_sec": round(len(good) / duration, 3) if duration else 0,
            "goodput_requests": len(good), "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens, "duration_s": round(duration, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbo-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-url", help="Optional second OpenAI-compatible endpoint; omit for TurboLLM-only runs")
    parser.add_argument("--turbo-model", default="turbo-llm")
    parser.add_argument("--vllm-model")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured requests per workload/engine/load level")
    parser.add_argument("--workload-matrix", type=Path, help="JSON array of {name,prompt,output_tokens}; defaults to short and long prompts")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain how a KV cache improves autoregressive inference in three concise bullet points.")
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--itl-slo-ms", type=float, default=100)
    parser.add_argument("--gpu-hourly-cost", type=float, default=0,
                        help="Actual GPU+host hourly cost in your billing currency; 0 leaves cost unavailable")
    parser.add_argument("--currency", default="USD", help="Currency code for --gpu-hourly-cost")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    args = parser.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]
    if args.requests < 1 or args.warmup < 0 or any(c < 1 for c in levels):
        parser.error("requests/concurrency must be positive and warmup cannot be negative")
    workloads = json.loads(args.workload_matrix.read_text(encoding="utf-8-sig")) if args.workload_matrix else [
        {"name":"short_512_in_128_out", "prompt":"Summarize the key idea of inference-time KV caching. " * 40, "output_tokens":128},
        {"name":"long_2048_in_256_out", "prompt":"Explain model serving performance, covering prefill, decode, memory bandwidth, cache behavior, and scheduling. " * 170, "output_tokens":256},
    ]
    if not isinstance(workloads,list) or not workloads or any(not w.get("prompt") or int(w.get("output_tokens",0))<1 for w in workloads):
        parser.error("workload matrix must be a non-empty JSON array with prompt and positive output_tokens")
    if bool(args.vllm_url) != bool(args.vllm_model):
        parser.error("provide both --vllm-url and --vllm-model, or omit both for TurboLLM-only")
    targets = [("TurboLLM", args.turbo_url, args.turbo_model)]
    if args.vllm_url:
        targets.append(("vLLM", args.vllm_url, args.vllm_model))
    async def run():
        results = []
        for workload in workloads:
          for level in levels:
            order = list(targets)
            if len(order)>1 and (levels.index(level) + workloads.index(workload)) % 2: order.reverse()
            for name, url, model in order:
                sem = asyncio.Semaphore(level)
                warm_start=time.perf_counter()
                await asyncio.gather(*(request_one(name,url,model,workload["prompt"],int(workload["output_tokens"]),sem,warm_start) for _ in range(args.warmup)))
                start = time.perf_counter()
                rows = await asyncio.gather(*(request_one(name, url, model, workload["prompt"],
                                                          int(workload["output_tokens"]), sem, start)
                                              for _ in range(args.requests)))
                results.append({"engine": name, "concurrency": level, "workload": workload["name"],
                                "summary": stats(rows, args.ttft_slo_ms / 1000,
                                                 args.itl_slo_ms / 1000), "requests": rows})
        return results
    results = asyncio.run(run())
    engine_throughput = {
        name: {f"{item['workload']}:C{item['concurrency']}": item["summary"]["output_tokens_per_sec"]
        for item in results if item["engine"] == name}
        for name in dict.fromkeys(item["engine"] for item in results)
    }
    cost_per_million = {}
    if args.gpu_hourly_cost > 0:
        for name, levels in engine_throughput.items():
            cost_per_million[name] = {
                concurrency: round(args.gpu_hourly_cost * 1_000_000 / (rate * 3600), 6)
                if rate > 0 else None for concurrency, rate in levels.items()
            }
    hardware = {}
    try:
        import torch
        hardware = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        hardware = {"torch_version": None, "cuda_available": False, "gpu": None}
    paired_quality = {}
    for item in results:
        key=f"{item['workload']}:C{item['concurrency']}"
        paired_quality.setdefault(key,{})[item["engine"]]=[r.get("text","") for r in item["requests"]]
    quality_match={key: round(sum(a==b for a,b in zip(value.get("TurboLLM",[]),value.get("vLLM",[])))/max(1,min(len(value.get("TurboLLM",[])),len(value.get("vLLM",[]))))*100,2)
                   for key,value in paired_quality.items() if "TurboLLM" in value and "vLLM" in value}
    report = {"schema_version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
              "hardware": hardware,
              "workload": {"requests_per_engine_per_level": args.requests,
                           "warmup_requests_per_level": args.warmup,
                           "concurrency": levels,
                           "workloads": [{"name":w["name"],"output_tokens":int(w["output_tokens"])} for w in workloads],
                           "ttft_slo_ms": args.ttft_slo_ms,
                           "itl_slo_ms": args.itl_slo_ms,
                           "gpu_hourly_cost": args.gpu_hourly_cost,
                           "currency": args.currency,
                           "endpoints": {"TurboLLM": args.turbo_url, "vLLM": args.vllm_url}},
              "efficiency": {"gpu_utilization_percent": {name: None for name in engine_throughput},
                             "memory_bandwidth_utilization_percent": {name: None for name in engine_throughput},
                             "prefill_mfu_percent": {name: None for name in engine_throughput},
                             "kv_cache_usage_percent": {name: None for name in engine_throughput},
                             "prefix_cache_hit_rate_percent": {name: None for name in engine_throughput},
                             "cost_per_million_tokens": cost_per_million or None,
                             "unavailable_reason": "Endpoint runner does not collect host GPU, MFU, or engine cache counters. Supply telemetry fields from the serving host; cost also needs an actual hourly cost."},
              "runs": results,
              "quality": {"output_match": "Compare deterministic output text from matched request rows.",
                          "perplexity": None, "eval_score": None,
                          "note": "Use a fixed held-out evaluation set for accuracy/perplexity; serving timing alone cannot establish quality."},
              "notes": ["Endpoint streaming measurements include server queueing and HTTP overhead.",
                        "TPOT is estimated from TTFT-to-end duration divided by generated tokens; exact ITL requires token-ID timestamps, SSE chunk boundaries vary.",
                        "GPU counters, cost, KV hit rate, MFU, and quality require separate instrumentation/evaluation."]}
    report["quality"]["exact_output_match_percent_by_workload"] = quality_match
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["engine", "workload", "concurrency", "request", "ok",
            "start_offset_s", "ttft_s", "tpot_s", "e2e_s", "itl_mean_s", "itl_p95_s",
            "output_tokens", "prompt_tokens", "error"])
        writer.writeheader()
        for run in results:
            for i, row in enumerate(run["requests"], 1):
                writer.writerow({key: value for key, value in {
                    "engine": row.get("engine"), "workload": run.get("workload"),
                    "concurrency": run["concurrency"], "request": i,
                    **row,
                }.items() if key in writer.fieldnames})
    print(json.dumps({"output": str(args.output), "csv": str(csv_path),
                      "runs": [{"engine": r["engine"], "concurrency": r["concurrency"],
                                **r["summary"]} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
