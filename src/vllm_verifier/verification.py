"""Local structural verification; no claims of calibrated or semantic correctness."""

import json
import math
from typing import Any

from .errors import InvalidOutput
from .models import Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question, ScoreAnswer


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidOutput("duplicate JSON keys")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    text = text.strip()
    # Accept a single enclosing Markdown fence, never fish JSON out of prose/reasoning.
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise InvalidOutput("expected one JSON object") from exc


def _probability(value: Any) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise InvalidOutput("probabilities must be finite numbers between zero and one")
    return float(value)


def verify(text: str, question: Question) -> Answer:
    raw = strict_json(text)
    if isinstance(question, Noul):
        if not isinstance(raw, dict) or set(raw) != {"noul"}:
            raise InvalidOutput("expected only noul")
        return NoulAnswer(noul=_probability(raw["noul"]))
    if not isinstance(raw, dict) or set(raw) != {"probabilities"}:
        raise InvalidOutput("expected only probabilities")
    values = raw["probabilities"]
    size = len(question.criteria)
    if not isinstance(values, list) or len(values) != size:
        raise InvalidOutput("one probability is required per criterion, in order")
    probabilities = [_probability(value) for value in values]
    total = math.fsum(probabilities)
    if abs(total - 1.0) > 0.01 or total == 0:
        raise InvalidOutput("probabilities must sum to one (rounding tolerance 0.01)")
    probabilities = [value / total for value in probabilities]
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    confidence = 1.0 if size == 1 else max(0.0, min(1.0, 1.0 - entropy / math.log(size)))
    if isinstance(question, Choice):
        keys = list(question.criteria)
        distribution = dict(zip(keys, probabilities, strict=True))
        # Stable tie-breaking follows input order.
        chosen = keys[max(range(size), key=lambda index: probabilities[index])]
        return ChoiceAnswer(choice=chosen, probabilities=distribution, confidence=confidence)
    return ScoreAnswer(
        score=math.fsum(i * p for i, p in enumerate(probabilities)),
        probabilities={str(i): p for i, p in enumerate(probabilities)},
        confidence=confidence,
        legend={str(i): description for i, description in enumerate(question.criteria)},
    )
