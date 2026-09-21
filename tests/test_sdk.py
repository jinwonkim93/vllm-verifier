"""Exercise the installed official SDK's serializer, routes and response parser."""

import httpx2
from fastapi.testclient import TestClient
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

from vllm_verifier.config import Settings

from .support import uniform_app


def test_official_python_sdk():
    app = uniform_app(Settings(_env_file=None, api_key="local-key"))
    with TestClient(app) as gateway:

        def handler(request):
            response = gateway.request(
                request.method,
                request.url.raw_path.decode(),
                headers=dict(request.headers),
                content=request.content,
            )
            return httpx2.Response(
                response.status_code, headers=dict(response.headers), content=response.content
            )

        with TypeSafeClient(
            api_key="local-key",
            base_url="http://gateway",
            transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            result = client.system_one(
                state={"ticket": "Please refund the duplicate charge."},
                questions={
                    "team": Choice(instructions="Route", criteria={"billing": None, "other": None}),
                    "severity": Score(instructions="Rate", criteria=["low", {"high": "urgent"}]),
                    "refund": Noul(instructions="Refund requested?"),
                },
            )
            assert result.choices["team"].choice == "billing"
            assert result.scores["severity"].probabilities == {0: 0.5, 1: 0.5}
            assert result.scores["severity"].legend[1] == {"high": "urgent"}
            assert result.nouls["refund"].noul == 0.5
            assert result.model == "test-uniform"
            assert result.usage.input_tokens == 0
            assert "jev-latest" in {model.name for model in client.models.list().models}


def test_sdk_optional_instructions_and_single_score():
    app = uniform_app()
    with TestClient(app) as gateway:
        response = gateway.post(
            "/v1/systemone",
            json={
                "model": "jev-latest",
                "state": "test",
                "questions": {
                    "a": {"type": "choice", "criteria": {"only": None}},
                    "b": {"type": "score", "criteria": ["single"]},
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["answers"]["b"]["score"] == 0
