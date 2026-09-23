"""Dialogue and HTTP tests use explicit test doubles, never fake model benchmarks."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from examples.customer_service.actions import DemoStore, execute
from examples.customer_service.app import create_app
from examples.customer_service.catalog import CATEGORIES, INTENTS, SLOTS
from examples.customer_service.classifier import Classification, SystemOneClassifier
from examples.customer_service.dialogue import Conversation, Dialogue
from examples.customer_service.slots import extract


class ScriptedClassifier:
    def __init__(self, *intents):
        self.intents = iter(intents)
        self.calls = []

    async def classify(self, text, context):
        self.calls.append((text, context))
        value = next(self.intents)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, Classification) else Classification(value)

    async def close(self):
        pass


SAMPLES = {
    "order_id": "ORD-1002",
    "product": "에어버즈",
    "other_product": "미니 스피커",
    "query": "이어폰",
    "reason": "단순 변심",
    "quantity": "2",
    "option": "블랙",
    "address": "서울시 예시로 12",
    "instructions": "문 앞",
    "region": "서울",
    "date": "2026-10-01",
    "device": "아이폰",
    "event": "가을맞이",
    "coupon": "FALL20",
    "payment_method": "카드",
    "email": "demo@example.com",
    "phone": "010-0000-0000",
    "name": "데모",
    "notification": "끄기",
    "channel": "채팅",
    "message": "축하해",
}


def test_catalog_has_exactly_100_complete_intents():
    assert len(INTENTS) == 100 and len(CATEGORIES) == 10
    assert all(sum(i.category == c for i in INTENTS.values()) == 10 for c in CATEGORIES)
    assert len({i.example for i in INTENTS.values()}) == 100
    assert set(SAMPLES) == set(SLOTS)


@pytest.mark.parametrize("intent", INTENTS.values(), ids=INTENTS.keys())
def test_every_intent_has_a_working_action(intent):
    message, cards = execute(intent, {s: SAMPLES[s] for s in intent.required_slots}, DemoStore())
    assert isinstance(message, str) and message
    assert isinstance(cards, list)


async def test_eta_slot_filling_and_order_reference():
    model = ScriptedClassifier("delivery_eta", "delivery_tracking")
    bot, state = Dialogue(model), Conversation()
    first = await bot.turn(state, "내 상품 언제 도착해?")
    assert first["state"]["active"]["missing_slots"] == ["order_id"]
    second = await bot.turn(state, "ORD-1001")
    assert "2026-09-24" in second["reply"]["text"]
    assert state.active is None
    third = await bot.turn(state, "그 주문 현재 어디야?")
    assert "DEMO-71001" in third["reply"]["text"]
    assert len(model.calls) == 2
    assert "ORD-1001" in model.calls[-1][1]


async def test_change_requires_confirmation_and_correction_reconfirms():
    bot, state = Dialogue(ScriptedClassifier("order_quantity")), Conversation()
    await bot.turn(state, "주문 수량을 변경하고 싶어")
    await bot.turn(state, "ORD-1002")
    await bot.turn(state, "2개")
    assert state.active.phase == "confirming"
    assert state.store.orders["ORD-1002"]["quantity"] == 1
    await bot.turn(state, "아니 3개")
    assert state.active.slots["quantity"] == "3"
    assert state.active.phase == "confirming"
    await bot.turn(state, "네")
    assert state.store.orders["ORD-1002"]["quantity"] == 3
    assert len(state.store.tickets) == 1


async def test_interrupt_handoff_resume_preserves_original_frame():
    bot, state = Dialogue(ScriptedClassifier("return_request", "agent_handoff")), Conversation()
    await bot.turn(state, "반품할래")
    await bot.turn(state, "ORD-1003")
    await bot.turn(state, "상담사 연결해줘")
    assert state.suspended[0].slots == {"order_id": "ORD-1003"}
    await bot.turn(state, "채팅")
    assert not state.store.tickets
    await bot.turn(state, "네")
    assert len(state.store.tickets) == 1
    result = await bot.turn(state, "이어서")
    assert result["state"]["active"]["intent"] == "return_request"
    assert state.active.missing == ["reason"]
    await bot.turn(state, "단순 변심")
    assert state.active.phase == "confirming"
    await bot.turn(state, "취소")
    assert state.active is None and len(state.store.tickets) == 1


async def test_product_search_selection_and_followup():
    bot, state = Dialogue(ScriptedClassifier("product_search", "product_stock")), Conversation()
    await bot.turn(state, "상품 검색하고 싶어")
    assert state.active.missing == ["query"]
    result = await bot.turn(state, "무선 이어폰")
    assert result["reply"]["cards"][0]["name"] == "에어버즈"
    await bot.turn(state, "에어버즈")
    assert state.last_completed.intent == "product_detail"
    result = await bot.turn(state, "그 상품 재고 있어?")
    assert "24개" in result["reply"]["text"]


async def test_unknown_order_retains_frame_and_asks_again():
    bot, state = Dialogue(ScriptedClassifier("delivery_eta")), Conversation()
    result = await bot.turn(state, "ORD-9999 언제 도착해?")
    assert state.active.missing == ["order_id"]
    assert "없어요" in result["reply"]["text"]
    await bot.turn(state, "1001")
    assert state.last_completed.slots["order_id"] == "ORD-1001"


async def test_uncertain_classification_requires_selection():
    classified = Classification(
        None, [{"id": "delivery_eta", "label": "도착 예정 시간", "probability": 0.2}]
    )
    bot, state = Dialogue(ScriptedClassifier(classified)), Conversation()
    result = await bot.turn(state, "주문 문의")
    assert state.active is None and result["state"]["clarification_candidates"]
    await bot.turn(state, "1")
    assert state.active.intent == "delivery_eta"


@pytest.mark.parametrize(
    "text,expected,slot",
    [
        ("2026-02-30", "date", "date"),
        ("0개", "quantity", "quantity"),
        ("not-email", "email", "email"),
        ("1000개", "quantity", "quantity"),
    ],
)
def test_bad_slot_values_are_not_extracted(text, expected, slot):
    assert slot not in extract(text, expected)


def test_second_product_does_not_overwrite_first():
    assert extract("미니 스피커", "other_product").get("product") is None
    assert extract("미니 스피커", "other_product")["other_product"] == "미니 스피커"


def test_sessions_are_isolated_and_errors_are_transactional():
    router = ScriptedClassifier("delivery_eta", httpx.ConnectError("offline"))
    app = create_app(router)
    with TestClient(app) as a, TestClient(app) as b:
        assert a.get("/").status_code == 200
        assert len(a.get("/api/catalog").json()["intents"]) == 100
        a.post("/api/chat", json={"text": "언제 도착해?"})
        b.get("/api/session")
        assert b.get("/api/session").json()["state"]["active"] is None
        before = a.get("/api/session").json()
        assert a.post("/api/chat", json={"text": "상담사 연결해줘"}).status_code == 503
        assert a.get("/api/session").json() == before
        assert a.post("/api/chat", json={"text": "ORD-1001"}).status_code == 200
        assert a.post("/api/reset", json={}).json()["state"]["turn"] == 0
        assert a.post("/api/chat", json={"text": "x" * 401}).status_code == 422
        assert (
            a.post(
                "/api/chat", json={"text": "x"}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )


async def test_hierarchical_classifier_works_with_both_model_names():
    for model in (
        "llm-semantic-router/Decision-1.0-Kai-0.6B",
        "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    ):
        import json

        calls = []

        def handler(request, calls=calls, model=model):
            payload = json.loads(request.content)
            options = payload["questions"]["intent"]["criteria"]
            calls.append(options)
            chosen = "delivery" if "delivery" in options else "delivery_eta"
            p = {k: 1.0 if k == chosen else 0.0 for k in options}
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "answers": {
                        "intent": {
                            "type": "choice",
                            "choice": chosen,
                            "probabilities": p,
                            "confidence": 1.0,
                        }
                    },
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            )

        router = SystemOneClassifier("http://engine")
        await router.client.aclose()
        router.client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(handler)
        )
        result = await router.classify("언제 도착해?", "")
        assert result.intent == "delivery_eta" and result.trace["model"] == model
        assert list(map(len, calls)) == [11, 10]
        await router.close()


async def test_per_session_lock_serializes_turns():
    class SlowClassifier(ScriptedClassifier):
        async def classify(self, text, context):
            await asyncio.sleep(0.01)
            return Classification("delivery_eta")

    app = create_app(SlowClassifier())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as c:
        await c.get("/api/session")
        await asyncio.gather(
            c.post("/api/chat", json={"text": "언제 와?"}),
            c.post("/api/chat", json={"text": "ORD-1001"}),
        )
        state = (await c.get("/api/session")).json()["state"]
        assert state["turn"] == 2 and state["last_completed"]["intent"] == "delivery_eta"


async def test_thanks_does_not_fill_pending_reason():
    bot, state = Dialogue(ScriptedClassifier("return_request")), Conversation()
    await bot.turn(state, "ORD-1001 반품하고 싶어")
    await bot.turn(state, "고마워")
    assert state.active.missing == ["reason"]
    assert state.active.slots == {"order_id": "ORD-1001"}


async def test_classifier_rejects_invalid_probabilities():
    router = SystemOneClassifier("http://engine")
    await router.client.aclose()
    router.client = httpx.AsyncClient(
        base_url="http://engine",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"model": "bad", "answers": {"intent": {"probabilities": {"a": 2, "b": -1}}}},
            )
        ),
    )
    with pytest.raises(ValueError, match="invalid probabilities"):
        await router._choose("hello", {"a": "A", "b": "B"})
    await router.close()


async def test_clarification_keeps_entities_from_original_request():
    model = ScriptedClassifier(
        Classification(
            None, [{"id": "delivery_eta", "label": "도착 예정 시간", "probability": 0.25}]
        )
    )
    bot, state = Dialogue(model), Conversation()
    await bot.turn(state, "ORD-1001 도착 언제야?")
    result = await bot.turn(state, "1")
    assert "2026-09-24" in result["reply"]["text"]
    assert state.last_completed.slots["order_id"] == "ORD-1001"


async def test_independent_new_request_does_not_inherit_old_model_context():
    model = ScriptedClassifier("delivery_eta", "order_quantity")
    bot, state = Dialogue(model), Conversation()
    await bot.turn(state, "ORD-1001 언제 도착해?")
    await bot.turn(state, "주문 수량 변경해줘")
    assert model.calls[-1][1] == ""
    assert state.active.intent == "order_quantity"
