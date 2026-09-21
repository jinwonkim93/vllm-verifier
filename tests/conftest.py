import json
from pathlib import Path

import pytest

from vllm_verifier.backend import Completion
from vllm_verifier.models import Usage


@pytest.fixture
def payload():
    return json.loads(Path("examples/triage.json").read_text())


class ScriptedBackend:
    def __init__(self, texts, finish="stop"):
        self.texts = iter(texts)
        self.messages = []
        self.finish = finish
        self.closed = False

    async def complete(self, messages):
        self.messages.append(messages)
        item = next(self.texts)
        if isinstance(item, Exception):
            raise item
        return Completion(item, Usage(input_tokens=10, output_tokens=5), self.finish)

    async def ready(self):
        return True

    async def close(self):
        self.closed = True
