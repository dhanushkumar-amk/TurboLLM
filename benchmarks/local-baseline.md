# Local Inference Measurements

These are short smoke measurements, not stable benchmark claims. They were collected on the development PC documented in `environment.md` with the pinned Qwen2.5-0.5B-Instruct revision and CUDA FP16. The desktop GPU was also serving display workloads.

## Eager GQA vs PyTorch SDPA

Command: `python scripts/benchmark_attention.py --runs 2 --new-tokens 8`

| Backend | Median total time | Throughput | Peak allocated VRAM |
|---|---:|---:|---:|
| Eager | 0.288 s | 27.8 tokens/s | 961 MiB |
| SDPA | 0.260 s | 30.8 tokens/s | 961 MiB |

Greedy outputs matched on this workload. This two-run sample showed about 9.8% lower median elapsed time for SDPA. Run the script with at least 5–10 repetitions before treating the difference as reliable.

## Transformers vs TurboLLM

Command: `python scripts/benchmark_reference.py --runs 2 --new-tokens 8 --dtype fp16`

For a 37-token prompt, the short sample measured 0.218 s / 36.7 tokens/s for Transformers and 0.258 s / 31.0 tokens/s for TurboLLM SDPA. Greedy token sequences matched with both engines using repetition penalty 1.0. A separate one-run FP32 check also matched generated text. TurboLLM was about 18% slower in this short comparison; the implementation is a correctness-focused learning baseline, not a performance win over Transformers.

The benchmark script also supports `--dtype fp32`. Repeat measurements on idle hardware, include longer prompts and outputs, and record time-to-first-token and inter-token latency before publishing comparisons. No llama.cpp or vLLM result is reported: those runtimes were not installed/configured in this Windows environment.

## Uncached decode smoke measurement

Command: `python scripts/benchmark_reference.py --runs 1 --new-tokens 4 --dtype fp32 --cache off`

On the same 37-token prompt, one warm run plus one measured run took 0.107 s for Transformers and 0.127 s for TurboLLM. Both produced `KV caching is a`. The sample is too small for a stable performance claim; it confirms the uncached path runs and records a reproducible comparison command.
