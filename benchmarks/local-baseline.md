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

## Uncached decode smoke measurement

Command: `python scripts/benchmark_reference.py --runs 1 --new-tokens 4 --dtype fp32 --cache off`

On the same 37-token prompt, one warm run plus one measured run took 0.107 s for Transformers and 0.127 s for TurboLLM. Both produced `KV caching is a`. The sample is too small for a stable performance claim; it confirms the uncached path runs and records a reproducible comparison command.
