from __future__ import annotations

import unittest

import torch

from turbo_llm.cache import MemoryBudget, PagedKVCache, estimate_kv_cache_bytes
from turbo_llm.layers.attention import GroupedQueryAttention
from turbo_llm.layers.mlp import SwiGLU
from turbo_llm.layers.rms_norm import RMSNorm
from turbo_llm.layers.rope import RotaryEmbedding
from turbo_llm.sampling import sample_next_token
from turbo_llm.shapes import QWEN25_05B


class CoreTests(unittest.TestCase):
    def test_qwen_shape_contract(self):
        self.assertEqual(QWEN25_05B.head_dim, 64)
        self.assertEqual(QWEN25_05B.queries_per_kv_head, 7)

    def test_invalid_shape_contract(self):
        with self.assertRaises(ValueError):
            type(QWEN25_05B)(hidden_size=10, num_attention_heads=3)

    def test_rmsnorm_shape_and_scale(self):
        layer = RMSNorm(4)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(layer(values).shape, values.shape)
        with self.assertRaises(ValueError):
            layer(torch.ones(1, 3))

    def test_rope_preserves_shapes_and_rotates(self):
        rope = RotaryEmbedding(4)
        q = torch.ones(1, 2, 3, 4)
        k = torch.ones(1, 1, 3, 4)
        q_out, k_out = rope(q, k, torch.tensor([[0, 1, 2]]))
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)
        self.assertTrue(torch.equal(q_out[:, :, :1], q[:, :, :1]))

    def test_gqa_and_swiglu_shapes(self):
        layer = GroupedQueryAttention(4, 2, 8)
        output = layer(torch.randn(1, 4, 3, 8), torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
        self.assertEqual(output.shape, (1, 3, 4, 8))
        self.assertEqual(SwiGLU(8, 16)(torch.randn(2, 3, 8)).shape, (2, 3, 8))

    def test_sdpa_matches_eager_gqa(self):
        q = torch.randn(1, 4, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        eager = GroupedQueryAttention(4, 2, 8, backend="eager")(q, k, v)
        sdpa = GroupedQueryAttention(4, 2, 8, backend="sdpa")(q, k, v)
        torch.testing.assert_close(eager, sdpa, rtol=1e-5, atol=1e-5)

    def test_sampling_and_paged_cache_lifecycle(self):
        logits = torch.tensor([[0.0, 3.0, 1.0]])
        self.assertEqual(sample_next_token(logits).item(), 1)
        self.assertEqual(sample_next_token(logits, temperature=0.7, top_k=1).item(), 1)
        cache = PagedKVCache(1, 2, 1, 2, 2)
        cache.create_request("r")
        cache.append("r", torch.ones(1, 1, 2), torch.zeros(1, 1, 2))
        self.assertEqual(cache.read("r")[0].shape, (1, 1, 1, 2))
        cache.release("r")
        self.assertEqual(cache.free_count, 2)
        budget = MemoryBudget(16)
        budget.reserve("r", 8)
        self.assertEqual(budget.available_bytes, 8)
        with self.assertRaises(MemoryError):
            budget.reserve("too-large", 9)
        budget.release("r")
        self.assertEqual(budget.used_bytes, 0)
        self.assertEqual(estimate_kv_cache_bytes(
            num_layers=1, batch_size=1, num_key_value_heads=1,
            sequence_length=1, head_dim=2, dtype=torch.float16,
        ), 8)


if __name__ == "__main__":
    unittest.main()
