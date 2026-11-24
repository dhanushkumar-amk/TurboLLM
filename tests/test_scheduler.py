from __future__ import annotations

import asyncio
import time
import unittest

import torch

from turbo_llm.scheduler import ContinuousBatchScheduler


class SlowTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"max_position_embeddings": 32})()
        self.max_batch_seen = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        self.max_batch_seen = max(self.max_batch_seen, input_ids.shape[0])
        time.sleep(0.04)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
        logits[..., 2] = 1
        return type("Output", (), {"logits": logits})()


class SchedulerTests(unittest.TestCase):
    def test_new_request_joins_live_decode_batch(self):
        async def run():
            model = SlowTinyModel()
            scheduler = ContinuousBatchScheduler(
                model, max_batch_size=4, queue_limit=4, batch_window_seconds=0.001
            )
            first = asyncio.create_task(scheduler.submit(
                torch.tensor([[1, 2]]), max_new_tokens=3, eos_token_id=None
            ))
            await asyncio.sleep(0.015)
            second = asyncio.create_task(scheduler.submit(
                torch.tensor([[1, 2, 3]]), max_new_tokens=3, eos_token_id=None
            ))
            result_a, result_b = await asyncio.gather(first, second)
            self.assertEqual(result_a.shape[1], 5)
            self.assertEqual(result_b.shape[1], 6)
            self.assertEqual(model.max_batch_seen, 2)
            self.assertEqual(scheduler.requests_completed, 2)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
