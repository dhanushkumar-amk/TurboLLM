"""Run real-model HTTP health, completion, and streaming smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo_llm.server import create_app


def main() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        body = {"messages": [{"role": "user", "content": "Say hello briefly."}],
                "max_tokens": 3}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        print("Completion:", response.json()["choices"][0]["message"]["content"])
        body["stream"] = True
        stream = client.post("/v1/chat/completions", json=body)
        assert stream.status_code == 200 and "data: [DONE]" in stream.text, stream.text
        metrics = client.get("/metrics").json()
        print("Streaming: passed; metrics:", metrics)


if __name__ == "__main__":
    main()
