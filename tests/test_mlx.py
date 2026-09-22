import asyncio
import threading
from types import SimpleNamespace

import pytest

from vllm_verifier.backend import Completion
from vllm_verifier.config import Settings
from vllm_verifier.engine.mlx import MLXRuntime
from vllm_verifier.engine.mlx_backend import MLXBackend
from vllm_verifier.errors import ServiceError
from vllm_verifier.models import Usage


def test_mlx_rejects_non_mac_before_import(monkeypatch):
    monkeypatch.setattr("vllm_verifier.engine.mlx.platform.system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        MLXRuntime()


def fake_runtime(monkeypatch):
    monkeypatch.setattr("vllm_verifier.engine.mlx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("vllm_verifier.engine.mlx.platform.machine", lambda: "arm64")
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2])
    model = SimpleNamespace(config=SimpleNamespace(model_type="diffusion_gemma"))
    values = dict(
        cached_tokens=0,
        diffusion_canvas_tokens=64,
        diffusion_denoising_steps=5,
        diffusion_work_tokens=320,
        peak_memory=1,
        prompt_tps=10,
        generation_tps=20,
    )

    def stream(*args, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(is_draft=True, text="DRAFT", finish_reason=None)
        yield SimpleNamespace(is_draft=False, text='{"noul":', finish_reason=None)
        yield SimpleNamespace(
            is_draft=False,
            text="0.5}",
            finish_reason="stop",
            prompt_tokens=2,
            generation_tokens=6,
            **values,
        )

    modules = {
        "mlx.core": SimpleNamespace(
            metal=SimpleNamespace(is_available=lambda: True),
            reset_peak_memory=lambda: None,
            set_cache_limit=lambda n: None,
            set_memory_limit=lambda n: None,
            device_info=lambda: {"max_recommended_working_set_size": 100},
            array=lambda value, **kw: value,
            int32="int32",
        ),
        "mlx_vlm": SimpleNamespace(load=lambda *a, **kw: (model, tokenizer)),
        "mlx_vlm.generate.diffusion": SimpleNamespace(stream_diffusion_generate=stream),
    }
    monkeypatch.setattr("vllm_verifier.engine.mlx.importlib.import_module", modules.__getitem__)
    return MLXRuntime(canvas_tokens=64), calls


def test_mlx_uses_terminal_usage_and_discards_drafts(monkeypatch):
    runtime, calls = fake_runtime(monkeypatch)
    output = runtime.complete_tokens([1, 2], 256)
    assert output.text == '{"noul":0.5}'
    assert output.finish_reason == "stop"
    assert output.usage == Usage(input_tokens=2, output_tokens=6)
    assert calls[0]["diffusion_max_canvas_length"] == 64
    assert calls[0]["apc_manager"] is None
    assert runtime.stats[0]["diffusion_work_tokens"] == 320
    with pytest.raises(ValueError, match="context"):
        runtime.complete_tokens([1] * 4096, 256)
    with pytest.raises(ValueError, match="text"):
        runtime.encode([{"content": []}])


@pytest.mark.asyncio
async def test_backend_owns_one_thread_and_closes(monkeypatch):
    threads = []

    class Local:
        def __init__(self, *args, **kwargs):
            threads.append(threading.get_ident())
            self.stats = []

        def encode(self, messages):
            threads.append(threading.get_ident())
            return [1]

        def complete_tokens(self, tokens, output_tokens):
            threads.append(threading.get_ident())
            return Completion('{"noul":0.5}', Usage(), "stop")

    monkeypatch.setattr("vllm_verifier.engine.mlx_backend.MLXRuntime", Local)
    backend = MLXBackend(Settings(_env_file=None))
    assert not await backend.ready()
    await backend.start()
    try:
        assert await backend.ready()
        await asyncio.gather(*(backend.complete([{"content": "text"}]) for _ in range(3)))
        with pytest.raises(ServiceError, match="text"):
            await backend.complete([{"content": []}])
        assert len(set(threads)) == 1
        assert threads[0] != threading.get_ident()
    finally:
        await backend.close()
    assert not await backend.ready()
    with pytest.raises(ServiceError, match="not ready"):
        await backend.complete([{"content": "text"}])
    await backend.close()


@pytest.mark.asyncio
async def test_backend_releases_executor_on_failed_start(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("load failed")

    monkeypatch.setattr("vllm_verifier.engine.mlx_backend.MLXRuntime", fail)
    backend = MLXBackend(Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="load failed"):
        await backend.start()
    assert backend.closed


def test_api_selects_mlx_lifecycle(monkeypatch):
    from fastapi.testclient import TestClient

    from vllm_verifier.app import create_app

    events = []

    class LocalBackend:
        def __init__(self, settings):
            pass

        async def start(self):
            events.append("start")

        async def ready(self):
            return True

        async def complete(self, messages):
            return Completion('{"noul":0.8}', Usage(input_tokens=2, output_tokens=4), "stop")

        async def close(self):
            events.append("close")

    monkeypatch.setattr("vllm_verifier.engine.mlx_backend.MLXBackend", LocalBackend)
    settings = Settings(_env_file=None, runtime="mlx")
    with TestClient(create_app(settings)) as client:
        assert client.get("/readyz").status_code == 200
        response = client.post(
            "/v1/systemone",
            json={
                "model": "jev-latest",
                "state": "refund",
                "questions": {"q": {"type": "noul"}},
            },
        )
        assert response.status_code == 200
        assert response.json()["answers"]["q"]["noul"] == 0.8
    assert events == ["start", "close"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ('<|channel>thought\n<channel|>{"noul":1}', '{"noul":1}'),
        ('<|channel>thought\nReasoning text.<channel|>{"noul":1}', '{"noul":1}'),
        ('{"noul":0.5}', '{"noul":0.5}'),
        ('prose {"noul":0.5}', 'prose {"noul":0.5}'),
    ],
)
def test_gemma_protocol_channel(text, expected):
    from vllm_verifier.engine.gemma import final_content

    assert final_content(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "<|channel>thought\nno closing marker",
        '<|channel>unknown\n<channel|>{"noul":1}',
        '<|channel>thought\n<channel|><channel|>{"noul":1}',
    ],
)
def test_gemma_malformed_channel_fails(text):
    from vllm_verifier.engine.gemma import final_content

    with pytest.raises(ValueError):
        final_content(text)


def test_malformed_model_channel_reaches_strict_verifier(monkeypatch):
    from vllm_verifier.engine.gemma import final_content
    from vllm_verifier.errors import InvalidOutput
    from vllm_verifier.models import Noul
    from vllm_verifier.verification import verify

    runtime, _ = fake_runtime(monkeypatch)
    original = runtime.stream

    def malformed(*args, **kwargs):
        for item in original(*args, **kwargs):
            if not item.is_draft and item.finish_reason is None:
                item.text = "<|channel>thought\n" + item.text
            yield item

    runtime.stream = malformed
    completion = runtime.complete_tokens([1, 2], 256)
    assert completion.finish_reason == "stop"
    with pytest.raises(ValueError):
        final_content(completion.text)
    with pytest.raises(InvalidOutput):
        verify(completion.text, Noul(type="noul"))
