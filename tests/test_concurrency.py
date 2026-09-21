import asyncio

import httpx
import pytest

from vllm_verifier.app import create_app
from vllm_verifier.backend import Completion
from vllm_verifier.config import Settings
from vllm_verifier.errors import ServiceError
from vllm_verifier.models import Noul, SystemOneRequest, Usage
from vllm_verifier.service import DecisionService


class BlockingBackend:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.cancelled = 0

    async def complete(self, _):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return Completion('{"noul":0.6}', Usage(), "stop")
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    async def ready(self):
        return True

    async def close(self):
        pass


def request(count=3):
    return SystemOneRequest(
        model="jev-latest",
        state="test",
        questions={str(i): Noul(type="noul", instructions="Yes?") for i in range(count)},
    )


async def test_global_concurrency_across_requests():
    backend = BlockingBackend()
    service = DecisionService(Settings(_env_file=None, max_concurrency=2), backend)
    calls = [asyncio.create_task(service.evaluate(request())) for _ in range(3)]
    await backend.started.wait()
    await asyncio.sleep(0.01)
    assert backend.peak == 2
    backend.release.set()
    await asyncio.gather(*calls)
    assert backend.active == 0
    assert backend.peak == 2


async def test_deadline_cancels_running_and_queued_work():
    backend = BlockingBackend()
    settings = Settings(_env_file=None, request_timeout=0.03, max_concurrency=1)
    app = create_app(settings, backend)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            response = await c.post("/v1/systemone", json=request().model_dump())
            assert response.status_code == 504
            assert backend.active == 0
            assert backend.cancelled == 1
            backend.release.set()
            assert (await c.post("/v1/systemone", json=request().model_dump())).status_code == 200


async def test_admission_overload_and_slot_recovery():
    backend = BlockingBackend()
    app = create_app(Settings(_env_file=None, max_requests=1), backend)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            first = asyncio.create_task(c.post("/v1/systemone", json=request(1).model_dump()))
            await backend.started.wait()
            rejected = await c.post("/v1/systemone", json=request(1).model_dump())
            assert rejected.status_code == 529
            backend.release.set()
            assert (await first).status_code == 200
            assert (await c.post("/v1/systemone", json=request(1).model_dump())).status_code == 200


async def test_one_failed_question_cancels_siblings():
    backend = BlockingBackend()
    original = backend.complete

    async def fail_or_block(messages):
        if "fail me" in messages[-1]["content"]:
            await backend.started.wait()
            raise ServiceError(502, "bad", "Failure")
        return await original(messages)

    backend.complete = fail_or_block
    service = DecisionService(Settings(_env_file=None), backend)
    payload = request(2)
    payload.questions["0"].instructions = "fail me"
    with pytest.raises(ServiceError):
        await service.evaluate(payload)
    assert backend.active == 0
    assert backend.cancelled == 1
