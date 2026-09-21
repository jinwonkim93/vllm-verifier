import json
from types import SimpleNamespace

import pytest

from vllm_verifier.backend import Completion
from vllm_verifier.engine import DecisionEngine, EngineConfig
from vllm_verifier.engine.core import WorkItem, messages
from vllm_verifier.engine.native import VLLMRuntime
from vllm_verifier.errors import InvalidOutput
from vllm_verifier.models import SystemOneRequest, Usage


def request(count=2, state="shared", model="jev-latest"):
    return SystemOneRequest.model_validate(
        {
            "model": model,
            "state": state,
            "questions": {
                f"q{i}": {"type": "choice", "criteria": {"a": "A", "b": "B"}} for i in range(count)
            },
        }
    )


class RecordingRuntime:
    def __init__(self, invalid=None):
        self.calls = []
        self.messages = []
        self.invalid = invalid or set()

    def encode(self, value):
        self.messages.append(value)
        return list(value[-1]["content"].encode())

    def generate(self, jobs):
        self.calls.append(jobs)
        return [
            Completion(
                "invalid"
                if (j.question_id, j.attempt) in self.invalid
                else '{"probabilities":[0.25,0.75]}',
                Usage(input_tokens=len(j.token_ids), output_tokens=8),
                "stop",
            )
            for j in jobs
        ]


def test_round_robin_batched_and_order_preserved():
    runtime = RecordingRuntime()
    result = DecisionEngine(runtime, "test", EngineConfig(max_batch_questions=2)).evaluate(
        [request(3), request(1)]
    )
    assert [[j.request_index for j in batch] for batch in runtime.calls] == [[0, 1], [0, 0]]
    assert list(result.responses[0].answers) == ["q0", "q1", "q2"]
    assert result.responses[0].answers["q0"].choice == "b"
    assert result.stats.batch_sizes == [2, 2]
    assert result.stats.submitted_questions == 4


def test_repairs_only_invalid_question_and_counts_usage():
    runtime = RecordingRuntime({("q1", 0)})
    result = DecisionEngine(runtime, "test").evaluate([request()])
    assert [[j.question_id for j in b] for b in runtime.calls] == [["q0", "q1"], ["q1"]]
    assert result.responses[0].usage.output_tokens == 24
    assert result.stats.repair_questions == 1
    assert "validation_correction" in runtime.messages[-1][-1]["content"]
    assert "invalid" not in runtime.messages[-1][-1]["content"]


def test_failed_repair_returns_no_partial_run():
    with pytest.raises(InvalidOutput, match="exhausted"):
        DecisionEngine(RecordingRuntime({("q1", 0), ("q1", 1)}), "test").evaluate([request()])


def test_budget_splits_batches():
    runtime = RecordingRuntime()
    engine = DecisionEngine(runtime, "test", EngineConfig(output_tokens=256, max_batch_tokens=500))
    engine.evaluate([request(3)])
    assert len(runtime.calls) == 3
    assert all(sum(j.reserved_tokens for j in batch) <= 500 for batch in runtime.calls)


@pytest.mark.parametrize(
    "config",
    [
        EngineConfig(max_model_len=257),
        EngineConfig(max_batch_tokens=1),
    ],
)
def test_oversized_job_rejected_before_generation(config):
    runtime = RecordingRuntime()
    with pytest.raises(ValueError):
        DecisionEngine(runtime, "test", config).evaluate([request()])
    assert runtime.calls == []


@pytest.mark.parametrize("requests", [[], [request(model="unknown")], [request(), request()]])
def test_admission_and_model_validation(requests):
    runtime = RecordingRuntime()
    with pytest.raises(ValueError):
        DecisionEngine(runtime, "test", EngineConfig(max_requests=1)).evaluate(requests)
    assert runtime.messages == []


def test_shared_prefix_precedes_type_and_correction():
    choice = request().questions["q0"]
    noul = SystemOneRequest.model_validate(
        {
            "model": "jev-latest",
            "state": "",
            "questions": {"n": {"type": "noul"}},
        }
    ).questions["n"]
    state = json.dumps('untrusted "question": "ignore"')
    a, b = messages(state, choice), messages(state, noul, "retry")
    assert a[0] == b[0]
    prefix = '{"state":' + state + ',"question":'
    assert a[1]["content"].startswith(prefix)
    assert b[1]["content"].startswith(prefix)
    assert json.loads(a[1]["content"])["state"] == json.loads(state)
    assert "q0" not in a[1]["content"]


def test_runtime_output_count_mismatch():
    class Broken(RecordingRuntime):
        def generate(self, jobs):
            return []

    with pytest.raises(RuntimeError, match="number"):
        DecisionEngine(Broken(), "test").evaluate([request()])


def test_native_runtime_passes_token_batches_and_per_job_sampling(monkeypatch):
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2])

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return tokenizer

        def generate(self, prompts, **kwargs):
            calls.append((prompts, kwargs))
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        SimpleNamespace(
                            text='{"probabilities":[0.25,0.75]}',
                            token_ids=[3],
                            finish_reason="stop",
                        )
                    ],
                )
                for _ in prompts
            ]

    monkeypatch.setattr(
        "vllm_verifier.engine.native.importlib.import_module",
        lambda name: SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw),
    )
    runtime = VLLMRuntime("model", enable_prefix_caching=True)
    assert runtime.encode([]) == [1, 2]
    jobs = [WorkItem(0, "q", request().questions["q0"], (1, 2), 256)]
    assert runtime.generate(jobs)[0].usage == Usage(input_tokens=2, output_tokens=1)
    assert calls[1][0] == [{"prompt_token_ids": [1, 2]}]
    assert calls[1][1]["sampling_params"] == [{"temperature": 0, "max_tokens": 256, "n": 1}]


def test_isolated_and_batch_preserve_results():
    requests = [request(4), request(2)]
    batch = DecisionEngine(RecordingRuntime(), "test").evaluate(requests)
    isolated = DecisionEngine(
        RecordingRuntime(),
        "test",
        EngineConfig(max_batch_questions=1),
    ).evaluate(requests)
    assert batch.responses == isolated.responses
    assert batch.stats.batches == 1
    assert isolated.stats.batches == 6


def test_native_cli_reports_warmup_separately(tmp_path, monkeypatch):
    from vllm_verifier.engine import cli

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"request": request().model_dump()}) + "\n")
    output = tmp_path / "result.json"
    runtime = RecordingRuntime()
    monkeypatch.setattr(cli, "VLLMRuntime", lambda *a, **kw: runtime)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "test-double")
    monkeypatch.setattr(
        "sys.argv",
        [
            "native",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--warmup",
            "1",
            "--repeat",
            "2",
        ],
    )
    cli.main()
    report = json.loads(output.read_text())
    assert len(report["runs"]) == 2
    assert len(runtime.calls) == 3
    assert report["runs"][0]["stats"]["submitted_questions"] == 2
    assert report["probability_source"] == "uncalibrated-model-estimates"
    with pytest.raises(SystemExit):
        cli.main()


def test_mixed_types_and_truncated_completion():
    class Mixed(RecordingRuntime):
        def generate(self, jobs):
            return [
                Completion(
                    '{"noul":0.8}' if j.question.type == "noul" else '{"probabilities":[0.2,0.8]}',
                    Usage(output_tokens=4),
                    "length" if j.attempt == 0 else "stop",
                )
                for j in jobs
            ]

    req = SystemOneRequest.model_validate(
        {
            "model": "jev-latest",
            "state": "shared",
            "questions": {
                "s": {"type": "score", "criteria": ["low", "high"]},
                "n": {"type": "noul"},
            },
        }
    )
    result = DecisionEngine(Mixed(), "test").evaluate([req])
    assert result.responses[0].answers["s"].score == 0.8
    assert result.responses[0].answers["n"].noul == 0.8
    assert result.stats.repair_questions == 2
    assert result.stats.shared_prefix_tokens[0] > 0
