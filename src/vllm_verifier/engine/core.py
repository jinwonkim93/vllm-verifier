"""Compile typed decisions and admit bounded batches to a native model runtime."""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import Field

from ..backend import Completion
from ..errors import InvalidOutput
from ..models import Answer, Contract, Question, SystemOneRequest, SystemOneResponse, Usage
from ..prompts import SYSTEM
from ..verification import verify


class EngineConfig(Contract):
    max_batch_questions: int = Field(default=32, ge=1, le=4096)
    max_batch_tokens: int = Field(default=131072, ge=1)
    max_model_len: int = Field(default=16384, ge=257)
    output_tokens: int = Field(default=2048, ge=256, le=16384)
    validation_retries: int = Field(default=1, ge=0, le=3)
    max_requests: int = Field(default=128, ge=1, le=4096)


@dataclass(frozen=True)
class WorkItem:
    request_index: int
    question_id: str
    question: Question
    token_ids: tuple[int, ...]
    output_tokens: int
    attempt: int = 0

    @property
    def reserved_tokens(self) -> int:
        return len(self.token_ids) + self.output_tokens


class Runtime(Protocol):
    def encode(self, messages: list[dict[str, Any]]) -> list[int]: ...

    def generate(self, jobs: list[WorkItem]) -> list[Completion]: ...


@dataclass
class RunStats:
    submitted_questions: int = 0
    repair_questions: int = 0
    batches: int = 0
    generation_seconds: float = 0
    elapsed_seconds: float = 0
    batch_sizes: list[int] = field(default_factory=list)
    shared_prefix_tokens: list[int] = field(default_factory=list)


@dataclass
class DecisionRun:
    responses: list[SystemOneResponse]
    stats: RunStats


# All types use the same system prefix; type-specific material follows the shared state.
CONTRACT = """For choice and score return only {"probabilities": [numbers]}, one per criterion
in order, each in [0,1], summing to one. For noul return only {"noul": number} in [0,1]."""


def messages(
    state_json: str, question: Question, correction: str | None = None
) -> list[dict[str, Any]]:
    suffix: dict[str, Any] = {"question": question.model_dump(exclude_none=True)}
    if correction:
        suffix["validation_correction"] = correction
    # State remains JSON-escaped evidence. IDs and other questions never enter the prompt.
    content = (
        '{"state":'
        + state_json
        + ","
        + json.dumps(suffix, ensure_ascii=False, allow_nan=False, separators=(",", ":"))[1:]
    )
    return [
        {"role": "system", "content": SYSTEM + "\n" + CONTRACT},
        {"role": "user", "content": content},
    ]


class DecisionEngine:
    """Synchronous offline execution, one owner per runtime; not an online scheduler."""

    def __init__(self, runtime: Runtime, model: str, config: EngineConfig | None = None):
        self.runtime = runtime
        self.model = model
        self.config = config or EngineConfig()

    def _job(
        self,
        index: int,
        key: str,
        state: str,
        question: Question,
        attempt: int = 0,
        correction: str | None = None,
    ) -> WorkItem:
        tokens = tuple(self.runtime.encode(messages(state, question, correction)))
        job = WorkItem(index, key, question, tokens, self.config.output_tokens, attempt)
        if job.reserved_tokens > self.config.max_model_len:
            raise ValueError("Decision prompt plus output budget exceeds max_model_len")
        if job.reserved_tokens > self.config.max_batch_tokens:
            raise ValueError("A decision exceeds the batch token budget")
        return job

    def evaluate(self, requests: list[SystemOneRequest]) -> DecisionRun:
        started = time.perf_counter()
        if not requests or len(requests) > self.config.max_requests:
            raise ValueError("Request count is outside the engine admission limit")
        if any(r.model not in {self.model, "jev-latest", "diffusion-jev"} for r in requests):
            raise ValueError("Unknown model")
        # Serialize shared state once, then tokenize each complete chat with the real template.
        states = [json.dumps(r.state, ensure_ascii=False, allow_nan=False) for r in requests]
        queues = [
            deque(
                self._job(i, key, states[i], question)
                for key, question in request.questions.items()
            )
            for i, request in enumerate(requests)
        ]
        # Count token-identical prefixes, not hypothetical GPU cache hits or saved FLOPs.
        shared_prefix_tokens = []
        for queue in queues:
            if len(queue) < 2:
                shared_prefix_tokens.append(0)
                continue
            prefix = 0
            for column in zip(*(job.token_ids for job in queue), strict=False):
                if len(set(column)) != 1:
                    break
                prefix += 1
            shared_prefix_tokens.append(prefix)
        # Round-robin across requests before batching; no request owns the whole initial queue.
        pending: deque[WorkItem] = deque()
        while any(queues):
            for queue in queues:
                if queue:
                    pending.append(queue.popleft())
        answers: list[dict[str, Answer]] = [{} for _ in requests]
        usage = [Usage() for _ in requests]
        stats = RunStats(shared_prefix_tokens=shared_prefix_tokens)
        while pending:
            batch: list[WorkItem] = []
            reserved = 0
            while pending and len(batch) < self.config.max_batch_questions:
                if reserved + pending[0].reserved_tokens > self.config.max_batch_tokens:
                    break
                job = pending.popleft()
                batch.append(job)
                reserved += job.reserved_tokens
            tick = time.perf_counter()
            completions = self.runtime.generate(batch)
            stats.generation_seconds += time.perf_counter() - tick
            if len(completions) != len(batch):
                raise RuntimeError("Runtime returned a different number of completions")
            stats.batches += 1
            stats.batch_sizes.append(len(batch))
            stats.submitted_questions += len(batch)
            for job, completion in zip(batch, completions, strict=True):
                usage[job.request_index].add(completion.usage)
                try:
                    if completion.finish_reason != "stop":
                        raise InvalidOutput("Completion did not finish normally")
                    answer = verify(completion.text, job.question)
                except InvalidOutput as exc:
                    if job.attempt >= self.config.validation_retries:
                        raise InvalidOutput("Decision exhausted validation retries") from exc
                    pending.append(
                        self._job(
                            job.request_index,
                            job.question_id,
                            states[job.request_index],
                            job.question,
                            job.attempt + 1,
                            str(exc),
                        )
                    )
                    stats.repair_questions += 1
                else:
                    answers[job.request_index][job.question_id] = answer
        stats.elapsed_seconds = time.perf_counter() - started
        return DecisionRun(
            [
                SystemOneResponse(
                    model=self.model,
                    answers={key: answers[i][key] for key in request.questions},
                    usage=usage[i],
                )
                for i, request in enumerate(requests)
            ],
            stats,
        )
