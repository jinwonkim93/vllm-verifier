import json

import pytest
from pydantic import ValidationError

from vllm_verifier.errors import InvalidOutput
from vllm_verifier.models import Choice, Noul, Score, SystemOneRequest
from vllm_verifier.verification import verify


def choice():
    return Choice(type="choice", instructions="Select", criteria={"a": None, "b": "B"})


def test_choice_and_entropy():
    answer = verify('{"probabilities":[0.9,0.1]}', choice())
    assert answer.choice == "a"
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    assert answer.confidence == pytest.approx(0.5310044064)


def test_uniform_tie_and_singleton():
    assert verify('{"probabilities":[0.5,0.5]}', choice()).choice == "a"
    assert verify('{"probabilities":[0.5,0.5]}', choice()).confidence == 0
    singleton = Choice(type="choice", instructions="Select", criteria={"only": None})
    assert verify('{"probabilities":[1]}', singleton).confidence == 1


def test_score_and_structured_legend():
    question = Score(type="score", instructions={"q": "Rate"}, criteria=["low", {"level": "high"}])
    answer = verify('{"probabilities":[0.25,0.75]}', question)
    assert answer.score == 0.75
    assert answer.legend == {"0": "low", "1": {"level": "high"}}


@pytest.mark.parametrize(
    "text",
    [
        '{"probabilities":[0.9]}',
        '{"probabilities":[0.5,0.5,0]}',
        '{"probabilities":[-0.1,1.1]}',
        '{"probabilities":[0,0]}',
        '{"probabilities":[0.3,0.3]}',
        '{"probabilities":[true,0]}',
        '{"probabilities":["0.5",0.5]}',
        '{"probabilities":[NaN,0]}',
        '{"probabilities":[Infinity,0]}',
        '{"probabilities":{"a":1,"b":0}}',
        '{"probabilities":[1,0],"choice":"a"}',
        '{"probabilities":[1,0],"probabilities":[0,1]}',
        'Here you go: {"probabilities":[1,0]}',
        "{}",
        "null",
        "[]",
        '{"probabilities":[1,0]} trailing',
    ],
)
def test_reject_invalid_distributions(text):
    with pytest.raises(InvalidOutput):
        verify(text, choice())


def test_rounding_and_fenced_json():
    answer = verify('```json\n{"probabilities":[0.499,0.499]}\n```', choice())
    assert answer.probabilities == {"a": 0.5, "b": 0.5}


@pytest.mark.parametrize("value", [0, 0.4, 1])
def test_noul(value):
    answer = verify(json.dumps({"noul": value}), Noul(type="noul", instructions="Yes?"))
    assert answer.model_dump() == {"type": "noul", "noul": value}


@pytest.mark.parametrize("raw", ['{"noul":true}', '{"noul":1.1}', '{"noul":NaN}', "{}"])
def test_invalid_noul(raw):
    with pytest.raises(InvalidOutput):
        verify(raw, Noul(type="noul", instructions="Yes?"))


@pytest.mark.parametrize("state", [None, 1, True])
def test_reject_scalar_state(state):
    with pytest.raises(ValidationError):
        SystemOneRequest(
            model="jev-latest", state=state, questions={"a": Noul(type="noul", instructions="Yes?")}
        )
