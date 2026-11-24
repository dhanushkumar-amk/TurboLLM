# TurboLLM

TurboLLM is a learning project for building an LLM inference engine in PyTorch. The project is planned in stages: understand model execution, implement a correct forward pass, add KV-cache memory management and scheduling, then profile, serve, and benchmark the engine.

TurboLLM includes a learning-focused PyTorch inference engine, an API server, and reproducible benchmark and verification scripts. The 50-phase plan tracks implementation status.

## Project documents

- [Product requirements](PRD.md)
- [50-phase implementation plan](plans.md)
- [Learning roadmap](learning.md)
- [Architecture notes](architecture.md)
- [Local benchmark notes](benchmarks/local-baseline.md)
- [Reference generation record](benchmarks/reference-run.md)
- [Interview notes](interview.md)
- [Development hardware and software baseline](environment.md)

## Landing page

The landing page uses Next.js, TypeScript, Tailwind CSS, shadcn/ui, Motion, GSAP, and Lucide icons.

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

To check the project:

```bash
npm run lint
npm run build
```

## Current development machine

- AMD Ryzen 7 5800X
- 32 GB RAM
- NVIDIA GeForce RTX 3050 with 8 GB VRAM

These are development constraints, not benchmark claims. Performance targets will be set after a model and workload baseline are selected.

## Initial model

The first planned reference model is [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Its architecture exercises the RMSNorm, RoPE, SwiGLU, and GQA components in the roadmap. Record a pinned model revision before downloading weights or publishing benchmark results.

## PyTorch reference run

The ML dependencies are kept separate from the Next.js app dependencies. In PowerShell, create or activate `.venv`, then install and run:

```powershell
python -m pip install -r requirements-ml.txt
python scripts/reference_run.py
```

The runner pins model revision `7ae557604adf67be50417f59c2c2f167def9a775`, uses FP16 on CUDA, and asks for at most 48 new tokens. Hugging Face downloads the weights to its local cache. The printed load time, generation time, generated-token count, and peak allocated VRAM are a smoke-test baseline; repeat runs and control other GPU workloads before treating timing as a benchmark.

## Core implementation progress

The `turbo_llm` Python package starts the inference implementation. `ModelShape` records the pinned Qwen model dimensions and derives the 64-wide attention head and 7 query heads per KV head. `RMSNorm` implements the Qwen2 normalization formula in FP32 intermediate precision and returns the original activation dtype.

Verify the first layer against the pinned Hugging Face checkpoint with:

```powershell
python scripts/verify_rms_norm.py
```

The comparison copies the checkpoint's first input RMSNorm weights and checks a small FP16 CUDA input.

RoPE is implemented in `turbo_llm/layers/rope.py`. Verify it against Qwen's rotary embedding with:

```powershell
python scripts/verify_rope.py
```

The verifier checks query shape `(batch, 14, sequence, 64)` and key shape `(batch, 2, sequence, 64)` using the model's configured RoPE theta.

Grouped-query causal attention is implemented in `turbo_llm/layers/attention.py`. Verify it with the pinned model using:

```powershell
python scripts/verify_attention.py
```

The comparison feeds Q/K/V projected from a small hidden-state batch through both the TurboLLM implementation and Qwen's eager attention, including RoPE and a causal mask.

SwiGLU, the residual decoder layer, checkpoint loading, and the full Qwen forward pass are implemented in `turbo_llm`. Verify the MLP and full logits with:

```powershell
python scripts/verify_swiglu.py
python scripts/verify_model.py
```

The model verifier uses FP32 and eager reference attention to isolate forward-pass correctness from half-precision accumulation differences. On the reference GPU run, the checked logits matched exactly.

## Generation, cache, and serving

Run the focused cache checks and all unit/API tests with:

```powershell
python scripts/verify_cache.py
python -m unittest discover -s tests -v
```

The custom model supports greedy, temperature, top-k, top-p, and repetition-penalty sampling. KV cache size estimation, a contiguous decode cache, and a paged block allocator are included. `architecture.md` documents tensor layouts and current limitations.

Start the local API after installing `requirements-ml.txt`:

```powershell
uvicorn turbo_llm.server:app --host 127.0.0.1 --port 8000
```

The server provides `/health`, `/metrics`, `/v1/chat/completions` (including SSE streaming), and `/batch` for equal-length static batches. Non-streaming chat requests use the prototype continuous scheduler, which batches live decode steps, keeps compact per-request KV caches, and can admit new requests at token boundaries. Run `python scripts/smoke_server.py` to exercise the real pinned model through health, completion, and streaming routes.

## Profiling and benchmarks

```powershell
python scripts/profile_inference.py --tokens 8
python scripts/benchmark_attention.py --runs 5 --new-tokens 24
python scripts/benchmark_reference.py --runs 5 --new-tokens 24 --dtype fp16
python scripts/benchmark_reference.py --runs 5 --new-tokens 24 --dtype fp16 --cache off
python scripts/benchmark_scheduler.py --dtype fp32 --runs 3
```

The SDPA benchmark showed a preliminary latency improvement on the recorded short workload. TurboLLM remains slower than Transformers on the recorded single-request workload; continuous batching improved aggregate throughput in a small synthetic multi-request workload. See [local benchmark notes](benchmarks/local-baseline.md) for setup and limitations.
