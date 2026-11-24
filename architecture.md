# TurboLLM Architecture

TurboLLM is a learning inference engine for one pinned model: Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`.

## Inference path

```text
Messages -> Qwen tokenizer/chat template -> token IDs
  -> embedding (batch, sequence, 896)
  -> 24 decoder layers:
       RMSNorm -> Q/K/V projections -> RoPE on Q and K
       -> causal grouped-query attention (14 Q heads, 2 KV heads)
       -> residual -> RMSNorm -> SwiGLU (896 -> 4864 -> 896) -> residual
  -> final RMSNorm -> tied vocabulary projection (151,936 logits)
  -> greedy or sampled next token -> repeat
```

The implementation is in `turbo_llm/`. The eager attention backend is a readable reference. The optional PyTorch SDPA backend uses grouped-query attention directly and is selected by the local server. The model's weights load from the pinned Hugging Face checkpoint into TurboLLM's own forward implementation.

## KV cache

Each decoder layer retains rotated keys and values with shapes `(batch, 2, cached_sequence, 64)`. Prompt prefill creates the initial cache. Each decode call processes only the latest token and appends its K/V tensors. The paged allocator stores K/V pages in `(layers, K_or_V, blocks, KV_heads, block_size, head_dim)` layout and tracks request ownership with block tables.

At batch 1 and 32,768 tokens, FP16 cache storage is estimated as:

```text
2 tensors * 24 layers * 1 batch * 2 KV heads * 32,768 tokens * 64 values * 2 bytes
= 402,653,184 bytes (0.375 GiB)
```

The estimate excludes allocator metadata and temporary activations. The page allocator is an educational reference and gathers pages into contiguous tensors; it is not a fused paged-attention kernel.

## Serving

`turbo_llm.server:app` loads the pinned model at startup, uses SDPA on available CUDA devices, and exposes:

- `GET /health`
- `GET /metrics`
- `POST /v1/chat/completions` with a small OpenAI-compatible subset and SSE streaming
- `POST /batch` for equal-token-length static batches

The server bounds admitted requests and serializes GPU model calls by default to protect VRAM. Non-streaming chat requests use a continuous decode scheduler: new requests can join between token iterations, and active requests with different prompt lengths have their K/V caches padded and masked into one decode batch. Each request keeps its own compact cache and can finish independently. Streaming requests currently use the simpler per-request KV-cache path.
