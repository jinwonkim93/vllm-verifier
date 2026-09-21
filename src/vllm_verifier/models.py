import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Content = str | dict[str, JsonValue] | list[JsonValue]
Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def finite_json(cls, value: Any) -> Any:
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("non-finite JSON numbers are not supported")
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return value


class Choice(Contract):
    type: Literal["choice"]
    instructions: Content | None = None
    criteria: Annotated[dict[Identifier, Content | None], Field(min_length=1, max_length=255)]


class Score(Contract):
    type: Literal["score"]
    instructions: Content | None = None
    criteria: Annotated[list[Content], Field(min_length=1, max_length=10)]


class NoulCriteria(Contract):
    true: Content | None = None
    false: Content | None = None


class Noul(Contract):
    type: Literal["noul"]
    instructions: Content | None = None
    criteria: NoulCriteria | None = None


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class SystemOneRequest(Contract):
    model: Annotated[str, Field(min_length=1, max_length=256)]
    state: Content
    questions: Annotated[dict[Identifier, Question], Field(min_length=1, max_length=64)]


class ChoiceAnswer(Contract):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(Contract):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, Probability]
    confidence: Probability
    legend: dict[str, Content]


class NoulAnswer(Contract):
    type: Literal["noul"] = "noul"
    noul: Probability


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(Contract):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class SystemOneResponse(Contract):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class VisionRequest(SystemOneRequest):
    images: Annotated[list[str], Field(min_length=1, max_length=4)]


class VisionResponse(SystemOneResponse):
    observation: str
