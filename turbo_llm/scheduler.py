"""Continuous decode scheduler with per-request contiguous KV caches."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Sequence

import torch
from starlette.concurrency import run_in_threadpool

from turbo_llm.model import QwenModel
from turbo_llm.sampling import sample_next_token
from turbo_llm.generation import DEFAULT_EOS_TOKEN_IDS


@dataclass
class _Job:
    input_ids: torch.Tensor
    max_new_tokens: int
    future: asyncio.Future
    eos_token_id: int | Sequence[int] | None
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int | None
    generated: int = 0
    generator: torch.Generator | None = None
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


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
        eos_token_id: int | Sequence[int] | None = DEFAULT_EOS_TOKEN_IDS,
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
            previous_active = [job for job in active if not job.future.cancelled()]
            incoming: list[_Job] = []
            if not previous_active:
                try:
                    first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    self.queue.task_done()
                    incoming.append(first)
                except TimeoutError:
                    async with self._worker_lock:
                        if self.queue.empty():
                            self._worker_task = None
                            return
                    continue
            if self.batch_window_seconds:
                await asyncio.sleep(self.batch_window_seconds)
            while len(incoming) < self.max_batch_size:
                try:
                    incoming.append(self.queue.get_nowait())
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            incoming = [job for job in incoming if not job.future.cancelled()]
            if not previous_active and not incoming:
                continue
            try:
                new_active = await self._prefill(incoming) if incoming else []
                previous_active = await self._decode(previous_active) if previous_active else []
                active = previous_active + new_active
            except Exception as exc:
                for job in previous_active + incoming:
                    if not job.future.done():
                        job.future.set_exception(exc)
                active = []

    async def _forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
        if self.execution_semaphore is None:
            return await run_in_threadpool(
                _model_forward, self.model, input_ids, attention_mask,
                position_ids, past_key_values,
            )
        async with self.execution_semaphore:
            return await run_in_threadpool(
                _model_forward, self.model, input_ids, attention_mask,
                position_ids, past_key_values,
            )

    async def _prefill(self, jobs: list[_Job]) -> list[_Job]:
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
        output = await self._forward(ids, mask, positions)
        active = []
        for row, job in enumerate(jobs):
            pad_count = max_length - job.input_ids.shape[1]
            job.past_key_values = tuple(
                (key[row:row + 1, :, pad_count:, :].contiguous(),
                 value[row:row + 1, :, pad_count:, :].contiguous())
                for key, value in output.past_key_values
            )
            self._sample_and_advance(job, output.logits[row:row + 1, -1, :])
            if not self._finish_if_done(job):
                active.append(job)
        return active

    async def _decode(self, jobs: list[_Job]) -> list[_Job]:
        jobs = [job for job in jobs if not job.future.cancelled()]
        if not jobs:
            return []
        device = next(self.model.parameters()).device
        cache_lengths = [job.past_key_values[0][0].shape[2] for job in jobs]
        max_cache_length = max(cache_lengths)
        batch_size = len(jobs)
        batched_cache = []
        for layer_index in range(len(jobs[0].past_key_values)):
            key_rows, value_rows = [], []
            for job, cache_length in zip(jobs, cache_lengths):
                key, value = job.past_key_values[layer_index]
                pad = max_cache_length - cache_length
                key_rows.append(torch.nn.functional.pad(key, (0, 0, pad, 0)))
                value_rows.append(torch.nn.functional.pad(value, (0, 0, pad, 0)))
            batched_cache.append((torch.cat(key_rows, dim=0), torch.cat(value_rows, dim=0)))

        mask = torch.zeros((batch_size, max_cache_length + 1), dtype=torch.bool, device=device)
        positions = torch.empty((batch_size, 1), dtype=torch.long, device=device)
        for row, cache_length in enumerate(cache_lengths):
            mask[row, max_cache_length - cache_length:] = True
            positions[row, 0] = cache_length
        input_ids = torch.cat([job.input_ids[:, -1:].to(device) for job in jobs], dim=0)
        output = await self._forward(
            input_ids, mask, positions, tuple(batched_cache)
        )
        active = []
        for row, (job, cache_length) in enumerate(zip(jobs, cache_lengths)):
            start = max_cache_length - cache_length
            job.past_key_values = tuple(
                (key[row:row + 1, :, start:, :].contiguous(),
                 value[row:row + 1, :, start:, :].contiguous())
                for key, value in output.past_key_values
            )
            self._sample_and_advance(job, output.logits[row:row + 1, -1, :])
            if not self._finish_if_done(job):
                active.append(job)
        return active

    def _sample_and_advance(self, job: _Job, logits: torch.Tensor) -> None:
        if job.generator is None and job.seed is not None:
            job.generator = torch.Generator(device=logits.device).manual_seed(job.seed)
        token = sample_next_token(
            logits,
            temperature=job.temperature,
            top_k=job.top_k,
            top_p=job.top_p,
            repetition_penalty=job.repetition_penalty,
            previous_tokens=job.input_ids,
            generator=job.generator,
        )
        job.input_ids = torch.cat((job.input_ids, token.to(job.input_ids.device)), dim=1)
        job.generated += 1

    def _finish_if_done(self, job: _Job) -> bool:
        last_token = int(job.input_ids[0, -1])
        finished = (
            job.future.cancelled()
            or job.generated >= job.max_new_tokens
            or (job.eos_token_id is not None and last_token in (
                (job.eos_token_id,) if isinstance(job.eos_token_id, int) else job.eos_token_id
            ))
        )
        if finished:
            if not job.future.done():
                job.future.set_result(job.input_ids)
            self.requests_completed += 1
        return finished


@torch.inference_mode()
def _model_forward(model, input_ids, attention_mask, position_ids, past_key_values):
    return model(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
