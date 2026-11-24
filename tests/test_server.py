from __future__ import annotations

import unittest

import torch
from fastapi.testclient import TestClient

from turbo_llm.server import Engine, create_app


class TinyOutput:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = ()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"max_position_embeddings": 32})()

    def forward(self, input_ids, past_key_values=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        logits[..., 2] = 2
        return TinyOutput(logits)


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return torch.tensor([[1, 2]])

    def decode(self, tokens, skip_special_tokens=True):
        return "tiny response"

    def __call__(self, prompt, return_tensors="pt"):
        return type("Encoded", (), {"input_ids": torch.tensor([[1, 2]])})()


class ServerTests(unittest.TestCase):
    def test_health_metrics_and_chat(self):
        engine = Engine(TinyModel(), TinyTokenizer())
        with TestClient(create_app(engine)) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            response = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "tiny response")
            self.assertEqual(client.get("/metrics").json()["requests_total"], 1)

    def test_rejects_unknown_model(self):
        with TestClient(create_app(Engine(TinyModel(), TinyTokenizer()))) as client:
            response = client.post("/v1/chat/completions", json={
                "model": "not-supported", "messages": [{"role": "user", "content": "hi"}],
            })
            self.assertEqual(response.status_code, 400)

    def test_static_batch_and_streaming(self):
        with TestClient(create_app(Engine(TinyModel(), TinyTokenizer()))) as client:
            batch = client.post("/batch", json={"prompts": ["one", "two"], "max_tokens": 1})
            self.assertEqual(batch.status_code, 200, batch.text)
            self.assertEqual(len(batch.json()["choices"]), 2)
            stream = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "stream": True,
            })
            self.assertEqual(stream.status_code, 200, stream.text)
            self.assertIn("data: [DONE]", stream.text)


if __name__ == "__main__":
    unittest.main()
