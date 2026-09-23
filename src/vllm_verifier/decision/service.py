"""Bounded request queue feeding a single owner and physical question batches."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from ..config import Settings
from ..errors import ServiceError
from ..models import SystemOneRequest, SystemOneResponse, VisionRequest, VisionResponse
from .artifact import MODEL
from .runtime import KaiRuntime

logger = logging.getLogger(__name__)


class DecisionRuntime(Protocol):
    def evaluate_batch(
        self,
        requests: list[SystemOneRequest],
    ) -> list[SystemOneResponse | ServiceError]: ...


@dataclass
class Job:
    request: SystemOneRequest
    future: asyncio.Future[SystemOneResponse]


class DirectDecisionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = self
        self.model_name = MODEL
        self.runtime: DecisionRuntime | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision-owner")
        self.queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=settings.max_requests)
        self.worker: asyncio.Task[None] | None = None
        self.closed = False

    async def start(self) -> None:
        if self.settings.model not in {
            MODEL,
            "Decision-1.0-Kai-0.6B",
            Settings.model_fields["model"].default,
        }:
            await self.close()
            raise ValueError("The decision runtime currently supports Decision-1.0-Kai-0.6B")
        try:
            self.runtime = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                lambda: KaiRuntime(
                    cache_dir=self.settings.decision_cache_dir,
                    device=self.settings.decision_device,
                    batch_size=self.settings.decision_batch_size,
                    batch_tokens=self.settings.decision_batch_tokens,
                ),
            )
            self.worker = asyncio.create_task(self._work(), name="decision-scheduler")
        except BaseException:
            await self.close()
            raise

    async def ready(self) -> bool:
        return bool(
            self.runtime is not None and not self.closed and self.worker and not self.worker.done()
        )

    async def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.model not in {MODEL, "Decision-1.0-Kai-0.6B", "jev-latest", "diffusion-jev"}:
            raise ServiceError(422, "unknown_model", "Use jev-latest or the served model")
        if not await self.ready():
            raise ServiceError(503, "engine_unavailable", "Decision engine is not ready")
        future: asyncio.Future[SystemOneResponse] = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(Job(request.model_copy(deep=True), future))
        except asyncio.QueueFull as exc:
            raise ServiceError(529, "overloaded", "Decision queue is full") from exc
        return await future

    async def evaluate_vision(self, request: VisionRequest) -> VisionResponse:
        raise ServiceError(422, "unsupported_modality", "Kai currently supports text only")

    async def _work(self) -> None:
        jobs: list[Job] = []
        try:
            while True:
                jobs = [await self.queue.get()]
                if self.settings.decision_batch_wait_ms:
                    await asyncio.sleep(self.settings.decision_batch_wait_ms / 1000)
                # Public requests have at most 64 questions: eight requests fit 512 decisions.
                while len(jobs) < self.settings.decision_batch_requests:
                    try:
                        jobs.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                live = [job for job in jobs if not job.future.done()]
                if live:
                    assert self.runtime is not None
                    try:
                        results = await asyncio.get_running_loop().run_in_executor(
                            self.executor,
                            self.runtime.evaluate_batch,
                            [job.request for job in live],
                        )
                        if len(results) != len(live):
                            raise RuntimeError("Decision engine returned an incomplete batch")
                    except Exception:
                        logger.exception("Decision batch failed")
                        results = [
                            ServiceError(502, "engine_error", "Candidate scoring failed")
                            for _ in live
                        ]
                    for job, result in zip(live, results, strict=True):
                        if not job.future.done():
                            if isinstance(result, ServiceError):
                                job.future.set_exception(result)
                            else:
                                job.future.set_result(result)
                for _ in jobs:
                    self.queue.task_done()
                jobs = []
        finally:
            while not self.queue.empty():
                jobs.append(self.queue.get_nowait())
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(
                        ServiceError(503, "engine_unavailable", "Decision engine stopped")
                    )
                self.queue.task_done()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        # Also handle cancellation before the worker coroutine's first instruction.
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if not job.future.done():
                job.future.set_exception(
                    ServiceError(503, "engine_unavailable", "Decision engine stopped")
                )
            self.queue.task_done()
        # Running Metal operations cannot be preempted; no subsequent queued job is started.
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        self.runtime = None
