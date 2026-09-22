"""Async API boundary around a single owning MLX thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..backend import Completion
from ..config import Settings
from ..errors import ServiceError
from .mlx import MLXRuntime


class MLXBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-owner")
        self.runtime: MLXRuntime | None = None
        self.closed = False

    async def start(self) -> None:
        def load() -> MLXRuntime:
            return MLXRuntime(
                self.settings.model,
                revision=self.settings.mlx_revision,
                canvas_tokens=self.settings.mlx_canvas_tokens,
                max_model_len=self.settings.mlx_max_model_len,
            )

        try:
            self.runtime = await asyncio.get_running_loop().run_in_executor(self.executor, load)
        except BaseException:
            await self.close()
            raise

    async def complete(self, messages: list[dict[str, Any]]) -> Completion:
        if self.closed or self.runtime is None:
            raise ServiceError(503, "engine_unavailable", "Mac engine is not ready")
        # Vision is rejected before any inference; this runtime only validates text generation.
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ServiceError(
                422, "unsupported_modality", "Mac engine currently supports text only"
            )
        runtime = self.runtime

        def generate() -> Completion:
            try:
                tokens = runtime.encode(messages)
                return runtime.complete_tokens(tokens, self.settings.max_output_tokens)
            except ValueError as exc:
                raise ServiceError(
                    422, "engine_input_error", "Input exceeds engine limits"
                ) from exc
            except (RuntimeError, MemoryError) as exc:
                raise ServiceError(502, "engine_error", "Local inference failed") from exc
            finally:
                runtime.stats.clear()

        return await asyncio.get_running_loop().run_in_executor(self.executor, generate)

    async def ready(self) -> bool:
        return self.runtime is not None and not self.closed

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # MLX generation cannot be preempted safely; drain the running job on shutdown.
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        self.runtime = None
