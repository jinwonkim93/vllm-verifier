"""FP32 candidate scoring with the unchanged, pinned Kai model and tokenizer."""

import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..errors import ServiceError
from ..models import (
    Choice,
    Content,
    Noul,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from .artifact import MODEL, download_artifact, load_artifact
from .readout import candidate_answer


def content(value: Content | None, default: str = "") -> str:
    if value is None:
        result = default
    elif isinstance(value, str):
        result = value
    else:
        result = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not result.strip():
        raise ValueError("Decision input text must not be empty")
    return result


def records(request: SystemOneRequest) -> list[dict[str, Any]]:
    state = content(request.state)
    rows = []
    for qid, question in request.questions.items():
        q: dict[str, Any] = {
            "id": qid,
            "type": question.type.capitalize(),
            "text": content(question.instructions, "Evaluate the candidates against the state."),
        }
        if isinstance(question, Noul):
            if question.criteria:
                for key in ("false", "true"):
                    value = getattr(question.criteria, key)
                    if value is not None:
                        q[key + "_criterion"] = content(value)
        else:
            if len(question.criteria) < 2:
                raise ValueError("Kai requires at least two Choice candidates or Score levels")
            if isinstance(question, Choice):
                q["options"] = [
                    {"id": key, "text": key if value is None else key + ": " + content(value)}
                    for key, value in question.criteria.items()
                ]
            else:
                q["levels"] = [
                    {"id": str(i), "value": i, "text": content(value)}
                    for i, value in enumerate(question.criteria)
                ]
        rows.append({"id": qid, "state_text": state, "question": q})
    return rows


class KaiRuntime:
    """Single-owner runtime; no cross-request token or result cache."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        device: str = "mps",
        batch_size: int = 8,
        batch_tokens: int = 4096,
        artifact_dir: Path | None = None,
    ):
        torch = importlib.import_module("torch")

        if device not in {"mps", "cpu"}:
            raise ValueError("Decision runtime supports mps or cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Metal is unavailable; select decision_device=cpu explicitly")
        if batch_size < 1 or batch_tokens < 1024:
            raise ValueError("Invalid physical batch limits")
        self.device, self.batch_size, self.batch_tokens = device, batch_size, batch_tokens
        self.torch = torch
        # Match the reference's explicit attention path, including padding masks.
        torch.backends.mha.set_fastpath_enabled(False)
        directory = artifact_dir or download_artifact(cache_dir)
        artifact = load_artifact(directory)
        self.model, collator, self.config = artifact.load_export(
            directory / "native",
            device=device,
        )
        self.collator = type(collator)(
            collator.tokenizer, max_length=1024, state_truncation="error"
        )

    def evaluate_batch(
        self,
        requests: list[SystemOneRequest],
    ) -> list[SystemOneResponse | ServiceError]:
        if not 1 <= len(requests) <= 128 or sum(len(r.questions) for r in requests) > 512:
            raise ValueError("Batch admission exceeds 128 requests or 512 decisions")
        outcomes: list[SystemOneResponse | ServiceError] = []
        pending: dict[str, list[tuple[int, str, dict[str, Any], int]]] = defaultdict(list)
        for index, request in enumerate(requests):
            try:
                rows = records(request)
                # Admit the entire request before scheduling any of its questions.
                lengths = [self.collator.encode(row)["input_tokens"] for row in rows]
            except ValueError as exc:
                outcomes.append(ServiceError(422, "engine_input_error", str(exc)))
                continue
            outcomes.append(SystemOneResponse(model=MODEL, answers={}, usage=Usage()))
            for qid, row, length in zip(request.questions, rows, lengths, strict=True):
                pending[row["question"]["type"]].append((index, qid, row, length))
        try:
            with self.torch.inference_mode():
                for entries in pending.values():
                    entries.sort(key=lambda entry: entry[3])
                    start = 0
                    while start < len(entries):
                        end = start + 1
                        while end < len(entries) and end - start < self.batch_size:
                            if (end - start + 1) * entries[end][3] > self.batch_tokens:
                                break
                            end += 1
                        self._run(entries[start:end], requests, outcomes)
                        start = end
        except (RuntimeError, MemoryError, FloatingPointError, ValueError) as exc:
            for index, outcome in enumerate(outcomes):
                if isinstance(outcome, SystemOneResponse):
                    outcomes[index] = ServiceError(502, "engine_error", "Candidate scoring failed")
            # No partially evaluated response escapes a failed physical batch.
            self.last_error = type(exc).__name__
        for request, outcome in zip(requests, outcomes, strict=True):
            if isinstance(outcome, SystemOneResponse):
                outcome.answers = {key: outcome.answers[key] for key in request.questions}
        return outcomes

    def _run(
        self,
        entries: list[tuple[int, str, dict[str, Any], int]],
        requests: list[SystemOneRequest],
        outcomes: list[SystemOneResponse | ServiceError],
    ) -> None:
        batch, encoded = self.collator([entry[2] for entry in entries], device=self.device)
        logits = self.model(batch)
        if not self.torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite candidate logits")
        # Transfer one probability matrix per batch, not one GPU synchronization per question.
        probabilities = logits.softmax(-1).cpu().tolist()
        for i, ((ri, qid, _, _), row) in enumerate(zip(entries, encoded, strict=True)):
            p = probabilities[i][: len(row["positions"])]
            question = requests[ri].questions[qid]
            outcome = outcomes[ri]
            assert isinstance(outcome, SystemOneResponse)
            outcome.answers[qid] = candidate_answer(question, p)
            outcome.usage.input_tokens += row["input_tokens"]
