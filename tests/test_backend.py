import json

import httpx
import pytest

from vllm_verifier.backend import VLLMBackend
from vllm_verifier.config import Settings
from vllm_verifier.errors import ServiceError


async def test_upstream_protocol_and_readiness():
    async def handler(request):
        assert request.headers["authorization"] == "Bearer upstream-secret"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "response_format" not in body
        assert "structured_outputs" not in body
        assert "logprobs" not in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"noul":0.8}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            },
        )

    backend = VLLMBackend(
        Settings(_env_file=None, model="test-model", upstream_api_key="upstream-secret"),
        httpx.MockTransport(handler),
    )
    try:
        assert await backend.ready()
        result = await backend.complete([])
        assert result.usage.input_tokens == 12
        assert result.text == '{"noul":0.8}'
    finally:
        await backend.close()


@pytest.mark.parametrize("status,expected", [(400, 502), (401, 502), (429, 529), (503, 529)])
async def test_upstream_http_errors(status, expected):
    backend = VLLMBackend(
        Settings(_env_file=None),
        httpx.MockTransport(lambda _: httpx.Response(status, text="private upstream contents")),
    )
    try:
        with pytest.raises(ServiceError) as caught:
            await backend.complete([])
        assert caught.value.status == expected
        assert "private" not in str(caught.value)
        assert not await backend.ready()
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
    ],
)
async def test_malformed_upstream(response):
    backend = VLLMBackend(
        Settings(_env_file=None), httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    )
    try:
        with pytest.raises(ServiceError, match="Invalid inference"):
            await backend.complete([])
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "error,status", [(httpx.ReadTimeout("timeout"), 504), (httpx.ConnectError("unreachable"), 502)]
)
async def test_transport_errors(error, status):
    def handler(_):
        raise error

    backend = VLLMBackend(Settings(_env_file=None), httpx.MockTransport(handler))
    try:
        with pytest.raises(ServiceError) as caught:
            await backend.complete([])
        assert caught.value.status == status
    finally:
        await backend.close()


async def test_readiness_response_is_bounded():
    backend = VLLMBackend(
        Settings(_env_file=None),
        httpx.MockTransport(lambda _: httpx.Response(200, content=b" " * (2 * 1024 * 1024 + 1))),
    )
    try:
        assert not await backend.ready()
        with pytest.raises(ServiceError) as caught:
            await backend.complete([])
        assert caught.value.code == "upstream_response_too_large"
    finally:
        await backend.close()


async def test_redirect_is_not_followed():
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(307, headers={"Location": "http://other-server/private"})

    backend = VLLMBackend(Settings(_env_file=None), httpx.MockTransport(handler))
    try:
        with pytest.raises(ServiceError) as caught:
            await backend.complete([])
        assert caught.value.code == "upstream_error"
        assert len(calls) == 1
    finally:
        await backend.close()
