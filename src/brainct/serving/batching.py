"""Cross-request micro-batching of CNN calls.

Under concurrency, many small model calls (single slices, or the tail chunk of a series) each
pay ONNX Runtime's per-call overhead and run with poor core utilisation. The batcher collects
pending slice arrays for up to `max_wait_ms` (or until `max_batch` slices), runs one call,
and splits the outputs back to each caller. With no concurrency it adds at most max_wait_ms.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Forward = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class _Job:
    x: np.ndarray
    future: asyncio.Future


class MicroBatcher:
    def __init__(
        self,
        forward: Forward,
        max_batch: int = 32,
        max_wait_ms: float = 4.0,
        on_batch: Callable[[int], None] | None = None,
    ):
        self.forward = forward
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.on_batch = on_batch
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def submit(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(logits, embedding) for x [n, K, H, W]; large inputs are split into max_batch chunks."""
        loop = asyncio.get_running_loop()
        futures = []
        for i in range(0, len(x), self.max_batch):
            fut = loop.create_future()
            await self._queue.put(_Job(x[i : i + self.max_batch], fut))
            futures.append(fut)
        parts = await asyncio.gather(*futures)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = [await self._queue.get()]
            size = len(jobs[0].x)
            deadline = loop.time() + self.max_wait
            while size < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout)
                except TimeoutError:
                    break
                if size + len(job.x) > self.max_batch:
                    await self._run(jobs)  # flush, then start the next batch with this job
                    jobs, size = [job], len(job.x)
                    deadline = loop.time() + self.max_wait
                    continue
                jobs.append(job)
                size += len(job.x)
            await self._run(jobs)

    async def _run(self, jobs: list[_Job]) -> None:
        x = np.concatenate([j.x for j in jobs]) if len(jobs) > 1 else jobs[0].x
        try:
            logits, emb = await asyncio.to_thread(self.forward, x)
        except Exception as exc:  # deliver the failure to every waiting request
            for j in jobs:
                if not j.future.done():
                    j.future.set_exception(exc)
            return
        if self.on_batch is not None:
            self.on_batch(len(x))
        start = 0
        for j in jobs:
            n = len(j.x)
            if not j.future.done():
                j.future.set_result((logits[start : start + n], emb[start : start + n]))
            start += n
