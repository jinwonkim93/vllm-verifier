import asyncio
import base64
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from vllm_verifier.app import create_app
from vllm_verifier.cli import main
from vllm_verifier.config import Settings
from vllm_verifier.errors import ServiceError
from vllm_verifier.models import SystemOneRequest
from vllm_verifier.vision import validate_images

from .conftest import ScriptedBackend
from .support import uniform_app


def test_nested_nonfinite_request_rejected(payload):
    with TestClient(uniform_app()) as client:
        for value in ["NaN", "Infinity", "1e999"]:
            raw = (
                '{"model":"jev-latest","state":{"nested":[' + value + "]},"
                '"questions":{"q":{"type":"noul","instructions":"Yes?"}}}'
            )
            response = client.post(
                "/v1/systemone", content=raw, headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 422


def test_maximum_criteria_and_oversized_criteria(payload):
    payload["questions"] = {"q": {"type": "choice", "criteria": {str(i): None for i in range(255)}}}
    SystemOneRequest.model_validate(payload)
    payload["questions"]["q"]["criteria"]["overflow"] = None
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://user:pass@host/v1", "http://host?key=x"]
)
def test_invalid_upstream_urls(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, base_url=url)


def test_empty_keys_are_disabled():
    assert Settings(_env_file=None, api_key="", upstream_api_key="").api_key is None


@pytest.mark.parametrize("port", ["0", "-1", "65536"])
def test_cli_rejects_invalid_port(monkeypatch, port):
    monkeypatch.setattr("sys.argv", ["vllm-verifier", "--port", port])
    with patch("vllm_verifier.cli.uvicorn.run") as serve, pytest.raises(SystemExit):
        main()
    serve.assert_not_called()


def test_cli_port_and_non_loopback_guard(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vllm-verifier", "--port", "9090"])
    with (
        patch("vllm_verifier.cli.Settings", return_value=Settings(_env_file=None)),
        patch("vllm_verifier.cli.uvicorn.run") as serve,
    ):
        main()
        assert serve.call_args.kwargs["port"] == 9090
    monkeypatch.setattr("sys.argv", ["vllm-verifier", "--host", "0.0.0.0"])
    with (
        patch("vllm_verifier.cli.Settings", return_value=Settings(_env_file=None)),
        pytest.raises(SystemExit),
    ):
        main()


def test_vision_mime_mismatch():
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    with pytest.raises(ServiceError):
        validate_images([url])


def test_invalid_observation_does_not_proceed(payload):
    from .test_api import image_url

    payload["images"] = [image_url()]
    backend = ScriptedBackend([""], finish="stop")
    with TestClient(create_app(Settings(_env_file=None), backend)) as client:
        response = client.post("/v1/vision/systemone", json=payload)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_observation"
        assert len(backend.messages) == 1


async def test_chunked_body_size_and_slow_body_timeout():
    app = create_app(Settings(_env_file=None, max_body_bytes=1024, request_timeout=0.01))
    async with app.router.lifespan_context(app):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/systemone",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1),
            "http_version": "1.1",
        }
        chunks = iter([b"x" * 600, b"x" * 600])
        messages = []

        async def receive():
            return {"type": "http.request", "body": next(chunks), "more_body": True}

        async def send(message):
            messages.append(message)

        await app(scope.copy(), receive, send)
        assert messages[0]["status"] == 413

        async def slow_receive():
            await asyncio.sleep(1)

        messages.clear()
        await app(scope.copy(), slow_receive, send)
        assert messages[0]["status"] == 408


def test_openapi_contract():
    schema = create_app(Settings(_env_file=None)).openapi()
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    assert schema["paths"]["/v1/systemone"]["post"]["security"] == [{"HTTPBearer": []}]
    assert schema["paths"]["/healthz"]["get"]["security"] == []
