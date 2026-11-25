# Local Inference Measurements

These are short smoke measurements, not stable benchmark claims. They were collected on the development PC documented in `environment.md` with the pinned Qwen2.5-0.5B-Instruct revision and CUDA FP16 unless noted. The desktop GPU was also serving display workloads.

## Eager GQA vs PyTorch SDPA

Command: `python scripts/benchmark_attention.py --runs 2 --new-tokens 8`

| Backend | Median total time | Throughput | Peak allocated VRAM |
|---|---:|---:|---:|
| Eager | 0.288 s | 27.8 tokens/s | 961 MiB |
| SDPA | 0.260 s | 30.8 tokens/s | 961 MiB |

Greedy outputs matched on this workload. This two-run sample showed about 9.8% lower median elapsed time for SDPA.

## Transformers vs TurboLLM: short single-request workload

Command: `python scripts/benchmark_reference.py --runs 2 --new-tokens 8 --dtype fp16`

For a 37-token prompt, the short sample measured 0.218 s / 36.7 tokens/s for Transformers and 0.258 s / 31.0 tokens/s for TurboLLM SDPA. Greedy token sequences matched with both engines using repetition penalty 1.0. A separate one-run FP32 check also matched generated text. TurboLLM was about 18% slower in this short comparison.

## Longer prompt and output

Command: `python scripts/benchmark_reference.py --runs 3 --new-tokens 96 --dtype fp16 --prompt "Explain in detail how KV caching works during autoregressive decoding, including prefill, decode, memory usage, and one tradeoff. Aim for around 100 words."`

Prompt length was 67 tokens. TurboLLM with caching produced 96 tokens in a median 3.471 s (27.66 tokens/s); Transformers took 2.771 s (34.64 tokens/s). The greedy outputs matched for the full generated sequence.

## Cache on versus off

The same 67-token prompt and 96-token limit with `--cache off` measured 3.558 s (26.98 tokens/s) for TurboLLM and 2.842 s (33.78 tokens/s) for Transformers. This FP16 run did not produce identical sequences (common prefix 61 tokens), so treat cache-on/off speed as indicative timing only, not an exact parity result.

## Continuous batching prototype

Command: `python scripts/benchmark_scheduler.py --dtype fp32 --runs 3`

Four synthetic requests with prompt lengths 12, 16, 20, and 24 tokens each generated 16 tokens. In FP32, serial generation measured 30.75 aggregate tokens/s and scheduler batching 63.78 tokens/s (about 2.07x on this small workload); all generated tokens matched. In FP16, the corresponding rates were 27.77 and 62.54 aggregate tokens/s, but one request diverged after its first generated token. Batched floating-point kernels can shift close logits, so the FP16 throughput run is not an exact correctness parity claim.

llama.cpp and vLLM comparisons remain unavailable; no results against them are claimed. Repeat measurements on idle hardware, record time-to-first-token and inter-token latency, and use representative workloads before drawing performance conclusions.

## What a fair vLLM comparison must control

Treat a comparison as valid only when these are recorded for **both** engines:

1. **Same model artifact:** model ID and immutable revision, tokenizer revision/chat template, dtype/quantization, and any speculative decoding or custom kernel settings.
2. **Same machine:** GPU model/count, VRAM, driver/CUDA, CPU/RAM, power limits, and whether another process is using the GPU. Run the engines separately on this single-GPU PC; do not load both model copies into 8 GB VRAM at once.
3. **Same workload:** a fixed prompt set with short/medium/long prompt lengths, fixed output caps, greedy sampling, matching stop behavior, and the same arrival/concurrency pattern. Report prompt and generated token counts.
4. **Repeatable protocol:** warm the model, run each load level at least five times, alternate engine order across repeats, and retain per-request raw data. Separate model load time from steady-state serving time.
5. **Primary outcomes:** TTFT p50/p95/p99, per-output-token ITL p50/p95/p99, E2E p50/p95/p99, output tokens/sec, requests/sec, errors/timeouts, and SLO goodput. Pick and state TTFT/ITL SLO thresholds before looking at results.
6. **Efficiency and quality:** peak/steady VRAM, GPU utilization, memory bandwidth and power sampled during the run; KV cache occupancy/prefix cache hit rate; cost per million tokens using the actual GPU-hour price; and a fixed quality evaluation/perplexity set. Quality must be checked when precision, quantization, kernels, or speculative decoding differ.

The current `benchmark_serving.py` captures serving latency and throughput from OpenAI-compatible SSE endpoints. Its ITL is based on client-observed **text chunks**, so it is provisional when servers group multiple generated tokens in one chunk. For precise ITL, instrument token IDs/timestamps in each server. The endpoint runner currently accepts one fixed prompt; use multiple prompt-length buckets before publishing a result. GPU/MFU/MBU, cache hit rate, cost, and quality are separate measurements and must not be inferred from latency.

### Run and export comparison images

Start TurboLLM on port 8000 and vLLM on port 8001 with the same pinned Qwen2.5-0.5B-Instruct revision, dtype, and GPU settings. On an 8 GB RTX 3050, stop one engine before loading the other if both copies do not fit. Then run:

```powershell
python scripts/benchmark_serving.py --turbo-url http://127.0.0.1:8000 --vllm-url http://127.0.0.1:8001 --turbo-model turbo-llm --vllm-model Qwen/Qwen2.5-0.5B-Instruct --requests 50 --concurrency 1,2,4,8 --output-tokens 64
python -m pip install matplotlib
python scripts/render_benchmark_charts.py benchmarks/results/latest.json
```

The report and raw request CSV are written under `benchmarks/results/`. Image files are written to `benchmarks/results/charts/`: `latency_tails.png`, `throughput.png`, and `goodput_heatmap.png`. Repeat the measurement run after warming each engine; do not count startup/download time as request latency.

## Uncached decode smoke measurement

Command: `python scripts/benchmark_reference.py --runs 1 --new-tokens 4 --dtype fp32 --cache off`

On the same 37-token prompt, one warm run plus one measured run took 0.107 s for Transformers and 0.127 s for TurboLLM. Both produced `KV caching is a`. The sample is too small for a stable performance claim; it confirms the uncached path runs and records a reproducible comparison command.
