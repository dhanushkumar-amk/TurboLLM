# TurboLLM — Product Requirements Document

**Status:** Draft  
**Owner:** Project team  
**Last updated:** 2026-11-23

## 1. Overview

TurboLLM is an educational, production-minded LLM inference engine built in PyTorch. It starts with a correct single-request implementation for a selected Hugging Face causal language model, then adds memory-efficient KV caching, request scheduling, performance optimizations, and an API for serving and benchmarking.

The project should make the major inference-engine components understandable and measurable. Hugging Face provides the correctness reference; vLLM is an external performance comparison. Neither is the implementation being built.

## 2. Problem and opportunity

Running a model through a high-level library is convenient, but it hides how transformer inference, KV-cache allocation, batching, and serving interact. Implementing those pieces in stages provides a practical way to learn the systems involved while producing a reproducible engine and honest performance results.

## 3. Product vision

Deliver a compact inference engine that can load a supported model, generate text correctly, reuse KV state efficiently, serve multiple concurrent requests, and expose enough benchmarks and documentation for another developer to reproduce and understand the results.

## 4. Goals

- Implement a transformer forward pass in PyTorch, including RMSNorm, RoPE, grouped-query attention (GQA), and SwiGLU.
- Validate model loading and logits against a Hugging Face reference.
- Support common sampling strategies and autoregressive generation.
- Add contiguous and paged KV caches with explicit memory management.
- Schedule concurrent requests with continuous batching and backpressure.
- Measure optimization and quantization effects on speed, memory, and quality.
- Provide an async HTTP API with streaming and operational health signals.
- Publish reproducible benchmarks and clear architecture and setup documentation.

## 5. Non-goals

- Supporting every model architecture, tokenizer, or hardware backend in the first release.
- Matching the performance or feature breadth of mature inference servers.
- Training or fine-tuning models.
- Claiming performance improvements without repeatable measurements.
- Making prefix caching or a dashboard a launch requirement; both are optional extensions.

## 6. Target users

- Developers learning how LLM inference engines work.
- Engineers evaluating cache, batching, and kernel tradeoffs on a defined setup.
- Contributors who need a small codebase with explicit correctness and performance checks.

## 7. Product principles

1. **Correctness before speed:** maintain a trusted reference path and test each optimization against it.
2. **One supported baseline first:** select and pin one model revision and environment before expanding scope.
3. **Visible resource ownership:** requests must have clear cache allocation, release, and cancellation behavior.
4. **Measured claims:** publish setup details and workload definitions with benchmark results.
5. **Understandable implementation:** keep components and tradeoffs documented, not hidden behind opaque abstractions.

## 8. User-facing capabilities

### 8.1 Model execution and generation

- Load a supported Hugging Face causal language model's weights and tokenizer.
- Apply the model's chat template when configured.
- Run a PyTorch forward pass with RMSNorm, RoPE, GQA, and SwiGLU.
- Generate tokens with greedy, temperature, top-k, and top-p sampling.
- Configure generation length, stopping behavior, and random seed where applicable.

### 8.2 Correctness and reference comparison

- Compare intermediate tensors and final logits against the pinned Hugging Face implementation.
- Define and document numerical tolerances for supported dtype/device combinations.
- Provide fixed inputs and reproducible settings for regression checks.

### 8.3 KV cache and memory management

- Support prefill and incremental decode using a contiguous KV cache.
- Estimate KV-cache memory from model dimensions, sequence length, batch size, and dtype.
- Support a paged KV cache using fixed-size blocks, a free-list allocator, and per-request block tables.
- Release cache state when requests complete, fail, or are cancelled.
- Handle capacity exhaustion explicitly; do not silently corrupt or lose request state.
- Prefix caching may be implemented as an optional extension after paged allocation is stable.

### 8.4 Scheduling and batching

- Track request lifecycle from admission and queueing through prefill, decode, and terminal state.
- Provide a static batching baseline and continuous batching scheduler.
- Handle variable prompt lengths with appropriate padding and attention masks.
- Support chunked prefill, admission control, queue limits, and backpressure.
- Allow requests to complete, time out, or cancel independently.

### 8.5 Optimization and quantization

- Evaluate PyTorch SDPA and FlashAttention where supported.
- Evaluate `torch.compile` and CUDA graphs on compatible configurations.
- Profile representative prefill and decode workloads using `torch.profiler`.
- Evaluate INT8 and INT4 quantization with quality comparison, including perplexity.
- Retain a clear baseline and correctness checks for optimized paths.

### 8.6 Serving and observability

- Expose an async FastAPI server with server-sent event (SSE) token streaming.
- Provide an OpenAI-compatible endpoint for the supported request subset.
- Support request cancellation and timeouts.
- Provide health checks and metrics for request counts, latency, token throughput, and resource use where available.
- Return actionable errors for invalid requests, capacity limits, and runtime failures.

### 8.7 Benchmarking and documentation

- Compare Hugging Face, TurboLLM, and vLLM on documented, comparable workloads.
- Report time to first token (TTFT), inter-token latency, tokens per second, P50/P95 latency, and peak memory.
- Record hardware, software versions, model revision, dtype, workload, concurrency, and runtime settings.
- Include setup instructions, architecture diagrams, supported features, limitations, and measured results in project documentation.
- A dashboard is optional and not required for the initial release.

## 9. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Load the selected model weights and tokenizer and report missing, unexpected, or mismatched weights clearly. | Must |
| FR-02 | Implement the baseline transformer operations and produce logits for tokenized input. | Must |
| FR-03 | Match Hugging Face reference logits within documented tolerances on fixed inputs. | Must |
| FR-04 | Provide greedy, temperature, top-k, and top-p sampling. | Must |
| FR-05 | Generate autoregressively with a configurable maximum token count and stopping conditions. | Must |
| FR-06 | Support uncached and contiguous-cache prefill/decode paths with equivalent outputs within tolerance. | Must |
| FR-07 | Report cache memory estimates and enforce capacity limits. | Must |
| FR-08 | Allocate and release paged KV blocks safely across request completion, errors, and cancellation. | Must |
| FR-09 | Queue and schedule requests, including continuous batching and variable-length inputs. | Must |
| FR-10 | Apply admission control and backpressure when queue or cache capacity is reached. | Must |
| FR-11 | Stream generated output through the HTTP API and support cancellation and timeouts. | Must |
| FR-12 | Expose health and operational metrics sufficient to diagnose serving behavior. | Should |
| FR-13 | Measure optimization and quantization paths against baseline correctness, speed, memory, and quality. | Should |
| FR-14 | Provide reproducible benchmark commands and publish environment/workload details. | Must |
| FR-15 | Implement prefix caching and a dashboard. | Could |

## 10. Non-functional requirements

- **Correctness:** fixed-input reference comparisons must be repeatable for a given supported environment.
- **Resource safety:** cache blocks and request state must be reclaimed on all terminal paths.
- **Clarity:** modules should have focused responsibilities and documented tensor shapes and cache behavior.
- **Reproducibility:** benchmark output must include enough configuration to rerun the workload.
- **Graceful limits:** unsupported hardware/features should produce a clear fallback or error.
- **Extensibility:** architecture support can expand after the initial model is stable without weakening baseline checks.

## 11. Phased delivery plan

### Phase 1 — Foundation and correctness

**Scope:** Select and pin a model baseline; implement model setup, weight loading, tokenizer/chat template, transformer operations, logits comparison, sampling, and basic generation.

**Exit criteria:**

- The model loads with a clear weight-loading report.
- Logits match Hugging Face within documented tolerance on fixed inputs.
- Greedy generation is repeatable for fixed inputs and settings.
- Sampling modes and basic stopping behavior are covered by reproducible checks.

### Phase 2 — KV cache and memory management

**Scope:** Add contiguous KV cache, prefill/decode separation, memory budgeting, paged blocks, allocator, block tables, cleanup, and OOM handling. Prefix caching is optional.

**Exit criteria:**

- Cached and uncached logits agree within tolerance at each decode step.
- Allocation and release preserve allocator invariants, including cancellation and error paths.
- Memory estimates are documented and compared with observed allocations.
- Capacity exhaustion is handled explicitly.

### Phase 3 — Scheduling and continuous batching

**Scope:** Add request lifecycle, queue, static batching baseline, continuous batching, variable-length attention handling, chunked prefill, admission control, and backpressure.

**Exit criteria:**

- Different-length requests can execute together and finish independently.
- Cancellation and timeout release all request resources.
- Queue and cache limits behave predictably under load.
- Continuous batching is compared with the static baseline on the same workload.

### Phase 4 — Performance optimization and quantization

**Scope:** Evaluate SDPA/FlashAttention, CUDA graphs, `torch.compile`, INT8/INT4 quantization, and profiler-guided improvements.

**Exit criteria:**

- Each retained optimization passes correctness checks and has a repeatable before/after measurement.
- Prefill and decode performance are reported separately.
- Quantization results include perplexity methodology and quality impact.
- Hardware and software compatibility limits are documented.

### Phase 5 — Serving, benchmarking, and presentation

**Scope:** Add FastAPI, async SSE streaming, cancellation, timeouts, OpenAI-compatible endpoint, health and metrics, benchmark suite, README, and architecture diagrams. Dashboard remains optional.

**Exit criteria:**

- API behavior is demonstrated for normal generation, streaming, invalid input, cancellation, timeout, and health checks.
- Benchmarks compare Hugging Face, TurboLLM, and vLLM using documented matching settings.
- Required latency, throughput, and memory metrics are published with environment details.
- README and diagrams explain setup, architecture, usage, results, and limitations.

## 12. Dependencies and sequencing

1. Pin the initial model, tokenizer, revision, dtype, device target, and environment.
2. Establish the Hugging Face reference harness before implementing optimized execution.
3. Complete basic uncached generation before adding KV cache.
4. Validate contiguous caching before introducing paged allocation.
5. Define cache ownership and request lifecycle before implementing continuous batching.
6. Build repeatable workload definitions before optimization and serving comparisons.
7. Benchmark only after correctness and resource cleanup are established.

## 13. Success metrics

- **Correctness:** reference logit agreement within the published tolerance for the supported baseline.
- **Reliability:** no leaked cache blocks across completion, cancellation, timeout, and error scenarios.
- **Serving:** successful independent handling of concurrent requests with bounded queue/cache behavior.
- **Performance:** reproducible TTFT, inter-token latency, throughput, P50/P95, and peak-memory results.
- **Quality:** documented perplexity change for quantized configurations.
- **Usability:** a new developer can install, run a sample generation, and reproduce a benchmark by following the README.

Numerical performance targets are intentionally TBD until the target hardware and baseline model are selected. Results must be reported as measurements, not implied guarantees.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Supporting too many model architectures too early | Pin one baseline model; expand only after its end-to-end checks pass. |
| Numerical drift from cache, batching, or kernels | Compare against the reference after each component change. |
| GPU feature differences | Record environment, provide supported fallbacks, and report feature constraints. |
| Cache leaks during cancellation or failures | Centralize request cleanup and verify allocator invariants on every terminal path. |
| Misleading benchmark comparisons | Document hardware, model, workload, concurrency, versions, and settings. |
| Quantization speed gains with unacceptable quality loss | Report perplexity and memory/latency together; keep baseline weights and path. |

## 15. Open decisions

- Which initial Hugging Face model and exact revision will define the baseline?
- What GPU/CPU environment and dtype are the initial support targets?
- What numerical tolerance should apply to each supported dtype and backend?
- Which OpenAI-compatible endpoint subset is required for the first server release?
- What benchmark workload sizes and concurrency levels will represent the target use case?

These decisions should be recorded before establishing performance targets or expanding model support.
