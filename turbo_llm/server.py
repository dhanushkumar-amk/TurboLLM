"""FastAPI OpenAI-compatible subset for local TurboLLM inference."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer

from turbo_llm.generation import generate, generate_token_stream
from turbo_llm.model import MODEL_ID, MODEL_REVISION, QwenModel


@dataclass
class Engine:
    model: QwenModel
    tokenizer: object
    max_concurrent: int = 1
    queue_limit: int = 16
    admission: asyncio.Semaphore = field(init=False)
    execution: asyncio.Semaphore = field(init=False)
    active: dict[str, str] = field(default_factory=dict)
    requests_total: int = 0
    errors_total: int = 0
    elapsed_total: float = 0.0

    def __post_init__(self) -> None:
        self.admission = asyncio.Semaphore(self.max_concurrent + self.queue_limit)
        self.execution = asyncio.Semaphore(self.max_concurrent)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    stream: bool = False


class BatchRequest(BaseModel):
    prompts: list[str] = Field(min_length=1, max_length=16)
    max_tokens: int = Field(default=32, ge=1, le=512)


class _StreamEnd:
    pass


_STREAM_END = _StreamEnd()


def _next_stream_item(iterator):
    return next(iterator, _STREAM_END)


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            model = await run_in_threadpool(
                QwenModel.from_pretrained, dtype=dtype, device=device,
                attention_backend="sdpa",
            )
            tokenizer = await run_in_threadpool(
                AutoTokenizer.from_pretrained, MODEL_ID, revision=MODEL_REVISION
            )
            app.state.engine = Engine(
                model=model,
                tokenizer=tokenizer,
                max_concurrent=int(os.getenv("TURBOLLM_MAX_CONCURRENT", "1")),
                queue_limit=int(os.getenv("TURBOLLM_QUEUE_LIMIT", "16")),
            )
        yield

    app = FastAPI(title="TurboLLM", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    async def health():
        current = app.state.engine
        return {"status": "ok" if current is not None else "loading", "model": MODEL_ID}

    @app.get("/metrics")
    async def metrics():
        current = app.state.engine
        if current is None:
            return {"requests_total": 0, "errors_total": 0, "active_requests": 0}
        return {
            "requests_total": current.requests_total,
            "errors_total": current.errors_total,
            "active_requests": len(current.active),
            "elapsed_seconds_total": round(current.elapsed_total, 3),
        }

    @app.post("/batch")
    async def static_batch(request: BatchRequest):
        current: Engine | None = app.state.engine
        if current is None:
            raise HTTPException(status_code=503, detail="model is loading")
        encoded = [current.tokenizer(prompt, return_tensors="pt").input_ids
                   for prompt in request.prompts]
        lengths = {item.shape[1] for item in encoded}
        if len(lengths) != 1:
            raise HTTPException(status_code=400, detail="static batch prompts must have equal token lengths")
        if not await _admit(current):
            raise HTTPException(status_code=429, detail="request queue is full")
        await current.execution.acquire()
        request_id = f"batch-{uuid.uuid4().hex}"
        current.active[request_id] = "prefill"
        current.requests_total += len(encoded)
        started = time.perf_counter()
        try:
            batch_ids = torch.cat(encoded, dim=0).to(next(current.model.parameters()).device)
            current.active[request_id] = "decode"
            result = await run_in_threadpool(
                generate, current.model, batch_ids, max_new_tokens=request.max_tokens
            )
            current.elapsed_total += time.perf_counter() - started
            return {"choices": [
                {"index": index, "text": current.tokenizer.decode(
                    result.sequences[index, result.prompt_tokens:], skip_special_tokens=True
                )}
                for index in range(len(encoded))
            ]}
        except Exception as exc:
            current.errors_total += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            current.active.pop(request_id, None)
            current.execution.release()
            current.admission.release()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        current: Engine | None = app.state.engine
        if current is None:
            raise HTTPException(status_code=503, detail="model is loading")
        if request.model not in (MODEL_ID, "turbo-llm"):
            raise HTTPException(status_code=400, detail=f"unsupported model: {request.model}")
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if not await _admit(current):
            raise HTTPException(status_code=429, detail="request queue is full")

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        current.active[request_id] = "queued"
        current.requests_total += 1
        execution_acquired = False
        stream_handed_off = False
        try:
            encoded = current.tokenizer.apply_chat_template(
                [message.model_dump() for message in request.messages],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = encoded["input_ids"]
            if encoded.shape[1] + request.max_tokens > current.model.config.max_position_embeddings:
                raise HTTPException(status_code=400, detail="prompt plus max_tokens exceeds model context length")
            device = next(current.model.parameters()).device
            encoded = encoded.to(device)
            await current.execution.acquire()
            execution_acquired = True
            current.active[request_id] = "prefill"
            if request.stream:
                current.active[request_id] = "decode"
                stream_handed_off = True
                return StreamingResponse(
                    _stream_response(current, request, encoded, request_id),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            started = time.perf_counter()
            current.active[request_id] = "decode"
            result = await run_in_threadpool(
                generate,
                current.model,
                encoded,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed,
            )
            current.elapsed_total += time.perf_counter() - started
            text = current.tokenizer.decode(
                result.sequences[0, result.prompt_tokens:], skip_special_tokens=True
            )
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": result.prompt_tokens,
                          "completion_tokens": result.generated_tokens,
                          "total_tokens": result.prompt_tokens + result.generated_tokens},
            }
        except asyncio.CancelledError:
            current.active[request_id] = "cancelled"
            raise
        except HTTPException:
            current.active[request_id] = "failed"
            raise
        except Exception as exc:
            current.active[request_id] = "failed"
            current.errors_total += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if not stream_handed_off:
                current.active[request_id] = "completed"
                current.active.pop(request_id, None)
                if execution_acquired:
                    current.execution.release()
                current.admission.release()

    return app


async def _admit(engine: Engine) -> bool:
    try:
        await asyncio.wait_for(engine.admission.acquire(), timeout=0.01)
        return True
    except TimeoutError:
        return False


async def _stream_response(engine: Engine, request: ChatCompletionRequest, input_ids: torch.Tensor,
                           request_id: str):
    tokenizer = engine.tokenizer
    prefix = ""
    generated_ids = []
    try:
        iterator = generate_token_stream(
            engine.model,
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
        )
        while True:
            token = await run_in_threadpool(_next_stream_item, iterator)
            if token is _STREAM_END:
                break
            generated_ids.extend(token[0].tolist())
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = text[len(prefix):] if text.startswith(prefix) else text
            prefix = text
            chunk = {"id": request_id, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": {"content": delta},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        engine.active.pop(request_id, None)
        engine.execution.release()
        engine.admission.release()


app = create_app()
