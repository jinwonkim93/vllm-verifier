"""Deterministic test doubles. Not included in the application wheel or image."""

import json
from typing import Any

from vllm_verifier.app import create_app
from vllm_verifier.backend import Completion
from vllm_verifier.config import Settings
from vllm_verifier.models import Usage


class UniformBackend:
    """Explicit, uniform fixtures for CPU-only wiring checks. Never an inference fallback."""

    async def complete(self, messages: list[dict[str, Any]]) -> Completion:
        if isinstance(messages[-1]["content"], list):
            return Completion("FIXTURE: image contents were not evaluated.", Usage(), "stop")
        question = json.loads(messages[-1]["content"])["question"]
        answer: dict[str, Any]
        if question["type"] == "noul":
            answer = {"noul": 0.5}
        else:
            count = len(question["criteria"])
            answer = {"probabilities": [1 / count] * count}
        return Completion(json.dumps(answer), Usage(), "stop")

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def uniform_app(settings=None):
    settings = settings or Settings(_env_file=None)
    settings.model = "test-uniform"
    return create_app(settings, UniformBackend())
