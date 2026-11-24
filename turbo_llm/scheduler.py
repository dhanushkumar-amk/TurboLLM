"""Continuous decode scheduler with equal-time-step dynamic microbatches.

This first scheduler favors understandable correctness: it pads active token
histories and reruns them on each decode tick. It demonstrates request arrival,
batch admission, independent completion, and cancellation, but does not use KV
pages and is not intended for high-throughput serving.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch
from starlette.concurrency import run_in_threadpool

from turbo_llm.model import QwenModel
from turbo_llm.sampling import sample_next_token


@dataclass
class _Job:
    input_ids: torch.Tensor
    max_new_tokens: int
    future: asyncio.Future
    eos_token_id: int | None
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int | None
    generated: int = 0
    generator: torch.Generator | None = None


class ContinuousBatchScheduler:
    """Admit requests between token steps and batch active requests together."""

    def __init__(
        self,
        model: QwenModel,
        *,
        pad_token_id: int = 151643,
        max_batch_size: int = 8,
        queue_limit: int = 32,
        batch_window_seconds: float = 0.005,
        execution_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if min(max_batch_size, queue_limit) <= 0 or batch_window_seconds < 0:
            raise ValueError("batch and queue sizes must be positive; window cannot be negative")
        self.model = model
        self.pad_token_id = pad_token_id
        self.max_batch_size = max_batch_size
        self.batch_window_seconds = batch_window_seconds
        self.execution_semaphore = execution_semaphore
        self.queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=queue_limit)
        self._worker_task: asyncio.Task | None = None
        self._worker_lock = asyncio.Lock()
        self.requests_completed = 0
        self.requests_cancelled = 0

    async def submit(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = 151645,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("each scheduled prompt must have shape (1, sequence)")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus generated tokens exceeds model context length")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = _Job(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            future=future,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise MemoryError("continuous-batch request queue is full") from exc
        async with self._worker_lock:
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = asyncio.create_task(self._run())
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            self.requests_cancelled += 1
            raise

    async def _run(self) -> None:
        active: list[_Job] = []
        while True:
            if not active:
                try:
                    first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    self.queue.task_done()
                    active.append(first)
                except TimeoutError:
                    async with self._worker_lock:
                        if self.queue.empty():
                            self._worker_task = None
                            return
                    continue
            if self.batch_window_seconds:
                await asyncio.sleep(self.batch_window_seconds)
            while len(active) < self.max_batch_size:
                try:
                    active.append(self.queue.get_nowait())
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            active = [job for job in active if not job.future.cancelled()]
            if not active:
                continue
            try:
                token_ids, attention_mask, position_ids = self._make_batch(active)
                if self.execution_semaphore is None:
                    output = await run_in_threadpool(
                        _model_forward, self.model, token_ids, attention_mask, position_ids
                    )
                else:
                    async with self.execution_semaphore:
                        output = await run_in_threadpool(
                            _model_forward, self.model, token_ids, attention_mask, position_ids
                        )
                logits = output.logits[:, -1, :]
                still_active = []
                for row, job in enumerate(active):
                    if job.generator is None and job.seed is not None:
                        job.generator = torch.Generator(device=logits.device).manual_seed(job.seed)
                    token = sample_next_token(
                        logits[row:row + 1],
                        temperature=job.temperature,
                        top_k=job.top_k,
                        top_p=job.top_p,
                        repetition_penalty=job.repetition_penalty,
                        previous_tokens=job.input_ids,
                        generator=job.generator,
                    )
                    job.input_ids = torch.cat((job.input_ids, token.to(job.input_ids.device)), dim=1)
                    job.generated += 1
                    finished = (
                        job.future.cancelled()
                        or job.generated >= job.max_new_tokens
                        or (job.eos_token_id is not None and token.item() == job.eos_token_id)
                    )
                    if finished:
                        if not job.future.done():
                            job.future.set_result(job.input_ids)
                        self.requests_completed += 1
                    else:
                        still_active.append(job)
                active = still_active
            except Exception as exc:
                for job in active:
                    if not job.future.done():
                        job.future.set_exception(exc)
                active = []

    def _make_batch(self, jobs: list[_Job]):
        device = next(self.model.parameters()).device
        max_length = max(job.input_ids.shape[1] for job in jobs)
        ids = torch.full(
            (len(jobs), max_length), self.pad_token_id,
            dtype=jobs[0].input_ids.dtype, device=device,
        )
        mask = torch.zeros((len(jobs), max_length), dtype=torch.bool, device=device)
        for row, job in enumerate(jobs):
            sequence = job.input_ids[0].to(device)
            ids[row, max_length - sequence.numel():] = sequence
            mask[row, max_length - sequence.numel():] = True
        positions = mask.long().cumsum(-1).sub(1).clamp_min(0)
        return ids, mask, positions


@torch.inference_mode()
def _model_forward(model, input_ids, attention_mask, position_ids):
    return model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
