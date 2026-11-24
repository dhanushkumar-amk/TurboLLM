"""Greedy, temperature, top-k, and top-p next-token selection."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
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
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
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
