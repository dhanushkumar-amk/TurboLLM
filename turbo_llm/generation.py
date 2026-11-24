"""Autoregressive decoding over the TurboLLM model."""

from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Sequence

import torch

from turbo_llm.model import QwenModel
from turbo_llm.sampling import sample_next_token


DEFAULT_EOS_TOKEN_IDS = (151645, 151643)


def _eos_tensor(eos_token_id: int | Sequence[int] | None, device: torch.device):
    if eos_token_id is None:
        return None
    ids = (eos_token_id,) if isinstance(eos_token_id, int) else tuple(eos_token_id)
    return torch.tensor(ids, device=device, dtype=torch.long)


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / max(self.elapsed_seconds, 1e-9)


@torch.inference_mode()
def generate_token_stream(
    model: QwenModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 64,
    eos_token_id: int | Sequence[int] | None = DEFAULT_EOS_TOKEN_IDS,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
):
    """Yield token IDs incrementally, retaining per-layer KV state."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus generated tokens exceeds model context length")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    sequences = input_ids
    output = model(input_ids)
    cache = output.past_key_values
    logits = output.logits[:, -1, :]
    for _ in range(max_new_tokens):
        token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=sequences,
            generator=generator,
        )
        yield token
        eos_ids = _eos_tensor(eos_token_id, token.device)
        if eos_ids is not None and bool(torch.isin(token, eos_ids).all()):
            break
        sequences = torch.cat((sequences, token), dim=1)
        output = model(token, cache)
        cache, logits = output.past_key_values, output.logits[:, -1, :]


@torch.inference_mode()
def generate(
    model: QwenModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 64,
    eos_token_id: int | Sequence[int] | None = DEFAULT_EOS_TOKEN_IDS,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
    seed: int | None = None,
) -> GenerationResult:
    """Generate from a batch of equal-length prompts."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus generated tokens exceeds model context length")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    result = input_ids
    started = time.perf_counter()
    if max_new_tokens == 0:
        return GenerationResult(result, input_ids.shape[1], 0, 0.0)

    output = model(input_ids)
    cache = output.past_key_values if use_cache else None
    logits = output.logits[:, -1, :]
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    eos_ids = _eos_tensor(eos_token_id, input_ids.device)
    generated = 0
    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=result,
            generator=generator,
        )
        if eos_ids is not None:
            next_token = torch.where(finished[:, None], eos_ids[0], next_token)
        result = torch.cat((result, next_token), dim=1)
        generated += 1
        if eos_ids is not None:
            finished |= torch.isin(next_token[:, 0], eos_ids)
            if bool(finished.all()):
                break
        if generated < max_new_tokens:
            output = model(next_token, cache) if use_cache else model(result)
            cache = output.past_key_values if use_cache else None
            logits = output.logits[:, -1, :]
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    return GenerationResult(
        sequences=result,
        prompt_tokens=input_ids.shape[1],
        generated_tokens=generated,
        elapsed_seconds=time.perf_counter() - started,
    )
