# TurboLLM Learning Roadmap

This is the study list for building TurboLLM. Follow it alongside [the implementation plan](plans.md): learn a concept just before we implement it, then connect the idea to the code and verify it against the pinned Qwen reference.

You do not need to master every topic before coding. We can move through this list in small steps. When a topic is marked **Later**, we will leave its theory for later and continue with the implementation work that depends on it only when needed.

## 1. Foundations: vectors, matrices, and tensors

- [ ] Scalars, vectors, matrices, and higher-dimensional tensors
- [ ] Matrix shapes and reading dimensions, for example `(batch, sequence, hidden)`
- [ ] Matrix multiplication: `(m × n) @ (n × p) -> (m × p)`
- [ ] Transpose, reshape, view, and why element order matters
- [ ] Broadcasting and elementwise operations
- [ ] Batch and sequence dimensions in language models
- [ ] Basic PyTorch tensors, devices, dtypes, and gradients

**Goal:** be comfortable reading a tensor shape and predicting the shape after an operation. Use matrix notation in explanations and diagrams whenever possible.

## 2. LLM and transformer basics

- [ ] What a token is; tokenizer vocabulary and token IDs
- [ ] Embedding lookup: token IDs become vectors
- [ ] Decoder-only transformer data flow
- [ ] Residual streams and skip connections
- [ ] Causal masking: a token cannot attend to future tokens
- [ ] Logits, softmax, next-token prediction, and autoregressive generation
- [ ] The difference between prefill (processing the prompt) and decode (generating tokens)

**Goal:** trace a prompt from text to token IDs, through model layers, to the next generated token.

## 3. Core model operations

- [ ] Linear layers as matrix multiplication plus bias (when present)
- [ ] Attention queries, keys, and values: Q, K, and V
- [ ] Attention scores and scaled dot-product attention
- [x] Rotary positional embeddings (RoPE) — implemented and compared with the Qwen reference; deeper matrix explanation can be studied when you want
- [x] Grouped-query attention (GQA): query heads share key/value heads — implemented and compared with Qwen's eager attention
- [x] Feed-forward networks and the SwiGLU activation — implemented and compared with Qwen
- [x] How one decoder block combines these pieces — implemented as a pre-norm residual decoder layer
- [x] How weights in a checkpoint map to operations in code — matching Qwen state-dict structure loads into TurboLLM

**Goal:** understand the operations needed to reproduce Qwen2.5-0.5B-Instruct's forward pass.

## 4. Normalization — study later as one group

We will defer the normalization theory until you are ready to study it. This group includes all normalization ideas relevant to the project:

- [ ] Why transformer layers use normalization
- [ ] Mean, variance, standard deviation, and root mean square (RMS)
- [ ] LayerNorm: subtract mean, divide by standard deviation, then learned scale/shift
- [ ] RMSNorm: divide by root mean square, then learned per-feature scale
- [ ] Pre-normalization and where Qwen applies RMSNorm in a decoder block
- [ ] Numerical stability, epsilon, FP16/BF16/FP32, and precision differences
- [ ] Compare TurboLLM normalization outputs with the Hugging Face reference

**Goal:** understand the matrix/vector calculation and why normalization stabilizes the activations. We have implemented RMSNorm, but the deeper lesson is intentionally postponed.

## 5. Tokenization and generation

- [ ] Tokenizer vocabulary, special tokens, and chat templates
- [ ] Greedy decoding
- [ ] Temperature and probability distributions
- [ ] Top-k and top-p sampling
- [ ] Stop tokens and maximum generation length
- [ ] Reproducible generation settings and seeds

**Goal:** turn model logits into a controlled sequence of output tokens.

## 6. KV cache and memory

- [ ] Why autoregressive generation repeats attention work
- [ ] What key/value tensors are stored for each layer and token
- [ ] KV-cache tensor shapes for Qwen's 24 layers and 2 KV heads
- [ ] Estimate cache memory from dimensions, sequence length, batch size, and dtype
- [ ] Prefill versus incremental decode with a cache
- [ ] Contiguous cache allocation
- [ ] Paged cache blocks, block tables, allocation, and release

**Goal:** generate tokens without recomputing all earlier keys and values, while keeping GPU memory use predictable.

## 7. Serving and scheduling

- [ ] Request lifecycle: queued, prefill, decode, completed, cancelled, failed
- [ ] Static batching and padding/masks
- [ ] Continuous batching and variable-length requests
- [ ] Backpressure, queue limits, cancellation, and cleanup
- [ ] HTTP APIs and streaming generated tokens
- [ ] Health checks and basic metrics

**Goal:** safely serve multiple requests while managing their cache memory.

## 8. Correctness, performance, and engineering

- [ ] Compare intermediate tensors and final logits to Hugging Face
- [ ] Numerical tolerances and why different dtypes can produce small differences
- [ ] Measure latency, tokens per second, and peak VRAM
- [ ] Warm-up runs, repeatability, and controlling other GPU workloads
- [ ] Profile before optimizing
- [ ] Understand PyTorch SDPA and other attention implementations
- [ ] Benchmark against a clearly defined baseline
- [ ] Record model revision, software versions, hardware, and workload

**Goal:** make correctness and performance claims reproducible and explainable.

## Qwen2.5-0.5B reference facts

These are model configuration values, not topics you need to memorize immediately:

| Setting | Value |
|---|---:|
| Hidden size | 896 |
| Decoder layers | 24 |
| Query attention heads | 14 |
| Key/value heads | 2 |
| Head dimension | 64 |
| MLP intermediate size | 4,864 |
| Vocabulary size | 151,936 |
| RMSNorm epsilon | 0.000001 |

## How we will use this list

For each implementation milestone, I’ll explain the relevant concept with matrix/tensor shapes, show where it appears in Qwen, implement a small piece, and compare it with the pinned reference. We’ll mark items complete as we cover and verify them. Normalization theory stays in section 4 until you ask to study it.
