"""Greedy, temperature, top-k, and top-p next-token selection."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    previous_tokens: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return one token ID per batch row from final-position logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocabulary)")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if top_k < 0 or top_k > logits.shape[-1]:
        raise ValueError("top_k must be between zero and vocabulary size")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")

    scores = logits.float().clone()
    if previous_tokens is not None and repetition_penalty != 1.0:
        if previous_tokens.ndim != 2 or previous_tokens.shape[0] != logits.shape[0]:
            raise ValueError("previous_tokens must have shape (batch, sequence)")
        token_ids = previous_tokens.to(device=logits.device, dtype=torch.long)
        previous_scores = scores.gather(1, token_ids)
        adjusted = torch.where(
            previous_scores < 0,
            previous_scores * repetition_penalty,
            previous_scores / repetition_penalty,
        )
        scores.scatter_(1, token_ids, adjusted)
    if temperature == 0:
        return scores.argmax(dim=-1, keepdim=True)

    scores = scores / temperature
    if top_k:
        threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(-1, sorted_indices, sorted_scores)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
