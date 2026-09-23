import asyncio
import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vllm_verifier.app import create_app
from vllm_verifier.config import Settings
from vllm_verifier.decision import artifact
from vllm_verifier.decision.runtime import KaiRuntime, content, records
from vllm_verifier.decision.service import DirectDecisionService
from vllm_verifier.errors import ServiceError
from vllm_verifier.models import NoulAnswer, SystemOneRequest, SystemOneResponse, Usage


def request(state="Evidence", questions=None):
    return SystemOneRequest.model_validate(
        {
            "model": "jev-latest",
            "state": state,
            "questions": questions or {"q": {"type": "noul", "instructions": "Is it evidence?"}},
        }
    )


def settings(**kwargs):
    return Settings(_env_file=None, runtime="decision", model=artifact.MODEL, **kwargs)


class FakeRuntime:
    def __init__(self, **kwargs):
        self.calls = []
        self.owner = threading.get_ident()

    def evaluate_batch(self, requests):
        assert threading.get_ident() == self.owner
        self.calls.append(requests)
        return [
            SystemOneResponse(
                model=artifact.MODEL,
                answers={qid: NoulAnswer(noul=0.75) for qid in r.questions},
                usage=Usage(),
            )
            for r in requests
        ]


def test_records_preserve_semantics_and_optional_fields():
    q = {
        "choice": {"type": "choice", "criteria": {"a": None, "b": {"value": 2}}},
        "score": {"type": "score", "instructions": ["urgency"], "criteria": ["low", "high"]},
        "noul": {"type": "noul", "criteria": {"true": "supported", "false": None}},
    }
    rows = records(request({"z": 1, "a": 2}, q))
    assert rows[0]["state_text"] == '{"a":2,"z":1}'
    assert rows[0]["question"]["options"] == [
        {"id": "a", "text": "a"},
        {"id": "b", "text": 'b: {"value":2}'},
    ]
    assert rows[1]["question"]["levels"][1] == {"id": "1", "value": 1, "text": "high"}
    assert rows[2]["question"]["true_criterion"] == "supported"
    assert "false_criterion" not in rows[2]["question"]


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_content_rejected(value):
    with pytest.raises(ValueError, match="empty"):
        content(value)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "criteria": {"a": "only"}},
        {"type": "score", "criteria": ["only"]},
    ],
)
def test_single_candidate_rejected(question):
    with pytest.raises(ValueError, match="at least two"):
        records(request(questions={"q": question}))


def test_artifact_verified_before_import(tmp_path, monkeypatch):
    root = tmp_path / "native"
    root.mkdir()
    source = root / "__init__.py"
    source.write_text("raise RuntimeError('must not import')")
    manifest = root / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "__init__.py": {
                        "bytes": source.stat().st_size,
                        "sha256": artifact.digest(source),
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="reviewed"):
        artifact.load_artifact(tmp_path)
    monkeypatch.setattr(artifact, "MANIFEST_SHA256", artifact.digest(manifest))
    artifact.verify_artifact(root)
    source.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        artifact.load_artifact(tmp_path)


def test_physical_batches_and_invalid_request_isolation():
    runtime = KaiRuntime.__new__(KaiRuntime)
    runtime.batch_size, runtime.batch_tokens = 2, 1024
    runtime.torch = SimpleNamespace(inference_mode=nullcontext)

    def encode(row):
        if row["state_text"] == "too long":
            raise ValueError("exceeds 1024 tokens")
        return {"input_tokens": int(row["state_text"])}

    runtime.collator = SimpleNamespace(encode=encode)
    batches = []

    def run(entries, requests, outcomes):
        batches.append(entries)
        for ri, qid, _, length in entries:
            outcomes[ri].answers[qid] = NoulAnswer(noul=0.5)
            outcomes[ri].usage.input_tokens += length

    runtime._run = run
    result = runtime.evaluate_batch(
        [request("700"), request("too long"), request("200"), request("250"), request("600")]
    )
    assert isinstance(result[1], ServiceError) and result[1].status == 422
    assert [[e[3] for e in b] for b in batches] == [[200, 250], [600], [700]]
    assert result[0].usage.input_tokens == 700
    assert result[2].usage.output_tokens == 0

    def fail(*args):
        raise RuntimeError("GPU failure")

    runtime._run = fail
    result = runtime.evaluate_batch([request("10"), request("too long")])
    assert [r.status for r in result] == [502, 422]


async def test_scheduler_coalesces_requests_and_preserves_order(monkeypatch):
    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", FakeRuntime)
    service = DirectDecisionService(settings(decision_batch_wait_ms=10))
    await service.start()
    try:
        results = await asyncio.gather(*(service.evaluate(request(str(i))) for i in range(9)))
        assert len(results) == 9
        assert [len(call) for call in service.runtime.calls] == [8, 1]
        assert [r.state for call in service.runtime.calls for r in call] == list(map(str, range(9)))
    finally:
        await service.close()
    assert not await service.ready()


async def test_cancelled_queued_request_never_runs(monkeypatch):
    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", FakeRuntime)
    service = DirectDecisionService(settings(decision_batch_wait_ms=30))
    await service.start()
    try:
        cancelled = asyncio.create_task(service.evaluate(request("cancelled")))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        await service.evaluate(request("live"))
        assert [r.state for call in service.runtime.calls for r in call] == ["live"]
    finally:
        await service.close()


async def test_queue_overload_and_shutdown(monkeypatch):
    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", FakeRuntime)
    service = DirectDecisionService(settings(max_requests=1, decision_batch_wait_ms=100))
    await service.start()
    first = asyncio.create_task(service.evaluate(request()))
    await asyncio.sleep(0)
    # Without yielding again the single queue slot is still occupied.
    with pytest.raises(ServiceError) as caught:
        await service.evaluate(request())
    assert caught.value.status == 529
    await service.close()
    result = await asyncio.gather(first, return_exceptions=True)
    assert isinstance(result[0], ServiceError) and result[0].status == 503


async def test_start_failure_cleans_executor(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", fail)
    service = DirectDecisionService(settings())
    with pytest.raises(RuntimeError, match="load failed"):
        await service.start()
    assert service.closed


async def test_batch_failure_recovers(monkeypatch):
    class FailsOnce(FakeRuntime):
        def evaluate_batch(self, requests):
            if not self.calls:
                self.calls.append(requests)
                return []
            return super().evaluate_batch(requests)

    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", FailsOnce)
    service = DirectDecisionService(settings(decision_batch_wait_ms=0))
    await service.start()
    try:
        with pytest.raises(ServiceError) as caught:
            await service.evaluate(request())
        assert caught.value.status == 502
        assert (await service.evaluate(request())).answers["q"].noul == 0.75
    finally:
        await service.close()


def test_api_and_official_sdk(monkeypatch):
    from typesafe_sdk import Noul, TypeSafeClient

    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", FakeRuntime)
    with TestClient(create_app(settings())) as http:
        assert http.get("/readyz").status_code == 200
        assert http.post("/v1/systemone", json=request().model_dump()).status_code == 200
        unknown = request().model_dump()
        unknown["model"] = "not-loaded"
        assert http.post("/v1/systemone", json=unknown).status_code == 422
        body = request().model_dump() | {"images": ["data:image/png;base64,AAAA"]}
        assert http.post("/v1/vision/systemone", json=body).status_code == 422
        with TypeSafeClient(api_key="test", base_url="http://testserver", http_client=http) as sdk:
            result = sdk.system_one(state="Evidence", questions={"q": Noul(instructions="Is it?")})
            assert result.nouls["q"].noul == 0.75


def test_candidate_readout_preserves_labels_and_structured_legends():
    from vllm_verifier.decision.readout import candidate_answer
    from vllm_verifier.models import Choice, Noul, Score

    choice = Choice(type="choice", criteria={"z": "last", "a": "first"})
    answer = candidate_answer(choice, [0.5, 0.5])
    assert answer.choice == "z" and answer.confidence == 0.5
    assert candidate_answer(Noul(type="noul"), [0.2, 0.8]).noul == 0.8
    score = Score(type="score", criteria=[{"level": "low"}, ["middle"], "high"])
    answer = candidate_answer(score, [0.1, 0.2, 0.7])
    assert answer.score == pytest.approx(1.6)
    assert answer.legend == {"0": {"level": "low"}, "1": ["middle"], "2": "high"}


@pytest.mark.parametrize("p", [[0.5], [float("nan"), 0.5], [-0.1, 1.1], [0.2, 0.4]])
def test_candidate_readout_rejects_invalid_distributions(p):
    from vllm_verifier.decision.readout import candidate_answer
    from vllm_verifier.models import Noul

    with pytest.raises(ValueError):
        candidate_answer(Noul(type="noul"), p)


async def test_cancelled_running_request_does_not_overlap_next_batch(monkeypatch):
    started, release = threading.Event(), threading.Event()

    class BlockingRuntime(FakeRuntime):
        def evaluate_batch(self, requests):
            if not self.calls:
                started.set()
                assert release.wait(timeout=5)
            return super().evaluate_batch(requests)

    monkeypatch.setattr("vllm_verifier.decision.service.KaiRuntime", BlockingRuntime)
    service = DirectDecisionService(settings(decision_batch_wait_ms=0))
    await service.start()
    try:
        first = asyncio.create_task(service.evaluate(request("running")))
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        second = asyncio.create_task(service.evaluate(request("next")))
        await asyncio.sleep(0)
        assert not service.runtime.calls and not second.done()
        release.set()
        await asyncio.wait_for(second, 5)
        assert [r.state for call in service.runtime.calls for r in call] == ["running", "next"]
    finally:
        release.set()
        await service.close()


def test_complete_cached_artifact_needs_no_network(tmp_path, monkeypatch):
    import sys

    directory = tmp_path / "decision" / artifact.REVISION / "native"
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("")
    manifest = directory / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": {"__init__.py": {}}}))
    monkeypatch.setattr(artifact, "MANIFEST_SHA256", artifact.digest(manifest))

    def fail_download(*args, **kwargs):
        raise AssertionError("Cached startup must not contact the network")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fail_download)
    )
    assert artifact.download_artifact(tmp_path) == directory.parent
