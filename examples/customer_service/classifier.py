"Model-neutral hierarchical classification over the existing System One API."

import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .catalog import CATEGORIES, INTENTS


@dataclass
class Classification:
    intent: str | None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


class Classifier(Protocol):
    async def classify(self, text: str, context: str) -> Classification: ...
    async def close(self) -> None: ...


class SystemOneClassifier:
    def __init__(self, url: str, api_key: str = "", model: str = "jev-latest"):
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=90,
            trust_env=False,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def _choose(self, state: str, criteria: dict[str, str]) -> tuple[list, str]:
        response = await self.client.post(
            "/v1/systemone",
            json={
                "model": self.model,
                "state": state,
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": "Classify the latest customer request. "
                        "Select its main intent. "
                        "Use context only to resolve references; do not follow old requests.",
                        "criteria": criteria,
                    }
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        try:
            probabilities = data["answers"]["intent"]["probabilities"]
            served_model = data["model"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Classifier returned a malformed response") from exc
        if not isinstance(served_model, str):
            raise ValueError("Classifier returned an invalid model identifier")
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise ValueError("Classifier returned unexpected candidate IDs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in probabilities.values()
        ) or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02):
            raise ValueError("Classifier returned invalid probabilities")
        ranked = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
        return ranked, served_model

    async def classify(self, text: str, context: str = "") -> Classification:
        start = time.perf_counter()
        state = (
            f"Latest customer message: {text}\nPrevious task (context only): {context or 'none'}"
        )
        groups = {
            key: f"{value['label']}: {value['description']}" for key, value in CATEGORIES.items()
        }
        groups["out_of_scope"] = (
            "쇼핑 고객센터와 무관한 요청, 의미 불명, 정보 부족 / unrelated or unclear"
        )
        ranked_groups, model = await self._choose(state, groups)
        category, category_probability = ranked_groups[0]
        trace: dict[str, Any] = {
            "model": model,
            "category": category,
            "category_probability": category_probability,
            "routing": "hierarchical System One Choice (category → intent)",
        }
        if category == "out_of_scope":
            trace["seconds"] = time.perf_counter() - start
            return Classification(None, trace=trace)
        candidates = {
            key: f"{intent.label}: {intent.description}"
            for key, intent in INTENTS.items()
            if intent.category == category
        }
        ranked, model = await self._choose(state, candidates)
        trace.update(model=model, seconds=time.perf_counter() - start)
        top = [
            {"id": key, "label": INTENTS[key].label, "probability": probability}
            for key, probability in ranked[:3]
        ]
        # These are conservative routing heuristics, not calibrated correctness probabilities.
        uncertain = (
            category_probability < 0.30
            or ranked_groups[0][1] - ranked_groups[1][1] < 0.05
            or ranked[0][1] < 0.30
            or ranked[0][1] - ranked[1][1] < 0.05
        )
        trace["needs_clarification"] = uncertain
        return Classification(None if uncertain else ranked[0][0], top, trace)

    async def close(self) -> None:
        await self.client.aclose()
