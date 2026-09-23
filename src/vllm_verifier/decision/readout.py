"""Typed answers from candidate-head probabilities; independent of model execution."""

import math

from ..models import Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question, ScoreAnswer


def candidate_answer(question: Question, probabilities: list[float]) -> Answer:
    size = 2 if isinstance(question, Noul) else len(question.criteria)
    if len(probabilities) != size or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError("Invalid candidate probability vector")
    if abs(math.fsum(probabilities) - 1) > 2e-5:
        raise ValueError("Candidate probabilities do not sum to one")
    if isinstance(question, Noul):
        return NoulAnswer(noul=probabilities[1])
    if isinstance(question, Choice):
        keys = list(question.criteria)
        return ChoiceAnswer(
            choice=keys[max(range(size), key=probabilities.__getitem__)],
            probabilities=dict(zip(keys, probabilities, strict=True)),
            confidence=max(probabilities),
        )
    return ScoreAnswer(
        score=math.fsum(i * p for i, p in enumerate(probabilities)),
        probabilities={str(i): p for i, p in enumerate(probabilities)},
        confidence=max(probabilities),
        legend={str(i): value for i, value in enumerate(question.criteria)},
    )
