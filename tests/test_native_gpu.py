"""Explicit GPU gate; never substitute synthetic responses when enabled."""

import os

import pytest

from vllm_verifier.engine import DecisionEngine, EngineConfig
from vllm_verifier.engine.native import VLLMRuntime
from vllm_verifier.models import SystemOneRequest


@pytest.mark.skipif(os.getenv("VERIFIER_TEST_GPU") != "1", reason="requires explicit GPU opt-in")
def test_native_diffusion_decisions():
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = os.getenv("VERIFIER_TEST_MODEL", "google/diffusiongemma-26B-A4B-it")
    runtime = VLLMRuntime(model, enable_prefix_caching=True, max_model_len=4096)
    engine = DecisionEngine(runtime, model, EngineConfig(max_model_len=4096))
    request = SystemOneRequest.model_validate(
        {
            "model": model,
            "state": "Please refund the duplicate payment.",
            "questions": {
                "team": {"type": "choice", "criteria": {"billing": "Payments", "tech": "Bugs"}},
                "refund": {"type": "noul", "instructions": "Is a refund requested?"},
            },
        }
    )
    result = engine.evaluate([request])
    assert set(result.responses[0].answers) == {"team", "refund"}
    assert result.responses[0].usage.input_tokens > 0
    assert result.responses[0].usage.output_tokens > 0
    assert result.stats.shared_prefix_tokens[0] > 0
