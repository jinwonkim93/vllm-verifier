import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vllm_verifier.app import create_app
from vllm_verifier.config import Settings
from vllm_verifier.errors import ServiceError

from .conftest import ScriptedBackend
from .support import uniform_app


def config(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_mixed_contract_usage_isolation_and_close(payload):
    backend = ScriptedBackend(
        [
            '{"probabilities":[0.8,0.15,0.05]}',
            '{"probabilities":[0.1,0.2,0.7]}',
            '{"noul":0.95}',
        ]
    )
    with TestClient(create_app(config(), backend)) as client:
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert set(result) == {"model", "answers", "usage"}
        assert result["model"] == config().model
        assert result["answers"]["department"]["choice"] == "billing"
        assert result["answers"]["urgency"]["score"] == pytest.approx(1.6)
        assert result["answers"]["refund"] == {"type": "noul", "noul": 0.95}
        assert result["usage"] == {"input_tokens": 30, "output_tokens": 15}
        assert response.headers["x-request-id"]
        for messages, question in zip(backend.messages, payload["questions"].values(), strict=True):
            prompt = json.loads(messages[-1]["content"])
            assert prompt == {"state": payload["state"], "question": question}
    assert backend.closed


def test_retry_counts_usage_and_exhaustion(payload):
    payload["questions"] = {"q": {"type": "noul", "instructions": "Refund?"}}
    backend = ScriptedBackend(['{"noul":2}', '{"noul":0.8}'])
    with TestClient(create_app(config(), backend)) as client:
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200
        assert response.json()["usage"] == {"input_tokens": 20, "output_tokens": 10}
        assert "failed validation" in backend.messages[-1][0]["content"]
    for finish, text in [("stop", "bad"), ("length", '{"noul":0.8}')]:
        backend = ScriptedBackend([text, text], finish=finish)
        with TestClient(create_app(config(), backend)) as client:
            response = client.post("/v1/systemone", json=payload)
            assert response.status_code == 502
            assert response.json()["error"]["code"] == "invalid_model_output"


def test_auth_before_body_and_public_probes(payload):
    with TestClient(uniform_app(config(api_key="test-key"))) as client:
        assert client.post("/v1/systemone", content="not json").status_code == 401
        assert client.get("/metrics").status_code == 401
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        headers = {"Authorization": "Bearer test-key"}
        response = client.post("/v1/systemone", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["model"] == "test-uniform"
        assert "verifier_requests_total" in client.get("/metrics", headers=headers).text


def test_unknown_model_does_not_call_backend(payload):
    backend = ScriptedBackend([])
    payload["model"] = "unrecognized"
    with TestClient(create_app(config(), backend)) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 422
        assert backend.messages == []


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"q": {"type": "unknown", "instructions": "x"}},
        {"q": {"type": "choice", "instructions": "x", "criteria": {}}},
        {"q": {"type": "score", "instructions": "x", "criteria": []}},
        {"q": {"type": "score", "instructions": "x", "criteria": ["x"] * 11}},
        {"q": {"type": "noul", "instructions": "x", "criteria": {"typo": "x"}}},
        {str(i): {"type": "noul", "instructions": "x"} for i in range(65)},
    ],
)
def test_invalid_request(payload, questions):
    payload["questions"] = questions
    with TestClient(uniform_app(config())) as client:
        result = client.post("/v1/systemone", json=payload)
        assert result.status_code == 422
        assert "I was charged" not in result.text


def test_body_limit_and_malformed_json():
    with TestClient(uniform_app(config(max_body_bytes=1024))) as client:
        assert client.post("/v1/systemone", content="x" * 1025).status_code == 413
        assert client.post("/v1/systemone", content='{"state":NaN}').status_code == 422


def image_url():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_vision_caption_once_and_usage(payload):
    payload["images"] = [image_url()]
    backend = ScriptedBackend(
        [
            "A red square.",
            '{"probabilities":[1,0,0]}',
            '{"probabilities":[0,0,1]}',
            '{"noul":0.9}',
        ]
    )
    with TestClient(create_app(config(), backend)) as client:
        response = client.post("/v1/vision/systemone", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["observation"] == "A red square."
        assert response.json()["usage"] == {"input_tokens": 40, "output_tokens": 20}
        assert len(backend.messages) == 4
        for messages in backend.messages[1:]:
            state = json.loads(messages[-1]["content"])["state"]
            assert state == {"context": payload["state"], "image_observation": "A red square."}


@pytest.mark.parametrize(
    "image",
    [
        "https://example.com/image.png",
        "http://169.254.169.254/metadata",
        "file:///etc/passwd",
        "data:image/png;base64,!!",
        "data:image/png;base64,YWJj",
    ],
)
def test_vision_rejects_urls_and_invalid_images(payload, image):
    payload["images"] = [image]
    backend = ScriptedBackend([])
    with TestClient(create_app(config(), backend)) as client:
        assert client.post("/v1/vision/systemone", json=payload).status_code == 422
        assert backend.messages == []


def test_backend_error_redacted(payload):
    backend = ScriptedBackend([ServiceError(529, "upstream_overloaded", "Overloaded")])
    with TestClient(create_app(config(), backend)) as client:
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 529
        assert response.headers["retry-after"] == "1"
