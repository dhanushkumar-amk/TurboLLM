"""A compact Qwen2.5 model implementation for TurboLLM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from turbo_llm.layers.decoder import DecoderLayer
from turbo_llm.layers.rms_norm import RMSNorm


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@dataclass
class ModelOutput:
    logits: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class QwenModel(nn.Module):
    """Qwen2 causal LM forward pass with optional layer-wise KV cache."""

    def __init__(self, config, attention_backend: str = "eager") -> None:
        super().__init__()
        self.config = config
        rope_theta = config.rope_parameters["rope_theta"]
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                eps=config.rms_norm_eps,
                rope_theta=rope_theta,
                attention_bias=True,
                attention_backend=attention_backend,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        position_ids = torch.arange(
            past_length, past_length + input_ids.shape[1], device=input_ids.device
        ).expand(input_ids.shape[0], -1)
        hidden_states = self.model.embed_tokens(input_ids)
        next_cache = []
        for index, layer in enumerate(self.model.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, layer_cache = layer(hidden_states, position_ids, past)
            next_cache.append(layer_cache)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return ModelOutput(logits=logits, past_key_values=tuple(next_cache))

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        *,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        attention_backend: str = "eager",
    ) -> "QwenModel":
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_id, revision=revision)
        reference = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, low_cpu_mem_usage=True
        )
        model = cls(config, attention_backend=attention_backend).to(dtype=dtype)
        incompatible = model.load_state_dict(reference.state_dict(), strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key != "lm_head.weight"]
        missing = [key for key in incompatible.missing_keys if key != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
        del reference
        return model.to(device).eval()
