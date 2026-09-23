"Explicit multi-turn frames, slot filling, corrections and confirmation gates."

from dataclasses import asdict, dataclass, field
from typing import Any

from .actions import DemoStore, execute
from .catalog import INTENTS, SLOTS
from .classifier import Classifier
from .slots import TASK_CUE, extract, is_slot_reply, normalize

YES = {"네", "예", "응", "좋아", "확인", "진행", "진행해", "진행해줘", "동의", "yes", "확인했어"}
NO = {"아니", "아니요", "아냐", "안할래", "취소", "그만", "cancel"}
RESUME = {"이어서", "계속", "아까거", "이전문의", "이전문의계속", "resume"}
REFER = ("그 주문", "그 상품", "그거", "아까", "같은 주문", "같은 상품", "이 상품")


@dataclass
class Frame:
    intent: str
    slots: dict[str, str] = field(default_factory=dict)
    phase: str = "collecting"

    @property
    def missing(self) -> list[str]:
        return [key for key in INTENTS[self.intent].required_slots if key not in self.slots]

    def view(self) -> dict[str, Any]:
        return {**asdict(self), "label": INTENTS[self.intent].label, "missing_slots": self.missing}


@dataclass
class Conversation:
    active: Frame | None = None
    suspended: list[Frame] = field(default_factory=list)
    last_completed: Frame | None = None
    choices: list[str] = field(default_factory=list)
    clarification_text: str = ""
    messages: list[dict] = field(default_factory=list)
    store: DemoStore = field(default_factory=DemoStore)
    trace: dict = field(default_factory=dict)
    turn: int = 0

    def snapshot(self) -> dict:
        return {
            "turn": self.turn,
            "active": self.active.view() if self.active else None,
            "suspended": [frame.view() for frame in self.suspended],
            "last_completed": self.last_completed.view() if self.last_completed else None,
            "clarification_candidates": self.choices,
            "demo_actions": len(self.store.tickets),
        }


class Dialogue:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    async def turn(
        self, state: Conversation, text: str, selected_intent: str | None = None
    ) -> dict:
        state.turn += 1
        state.messages.append({"role": "user", "text": text})
        state.trace = {"routing": "dialogue state", "model": None}
        command = normalize(text)
        if command in RESUME:
            if not state.suspended:
                return self.reply(state, "중단한 문의가 없어요. 어떤 도움이 필요하세요?")
            previous = state.suspended.pop()
            if state.active:
                state.suspended.append(state.active)
            state.active = previous
            state.choices = []
            return self.advance(state, prefix="이전 문의를 이어갈게요. ")
        if command in NO and (state.active or state.choices):
            state.active = None
            state.choices = []
            return self.reply(state, "진행 중인 문의를 취소했어요. 접수나 변경은 하지 않았어요.")
        if selected_intent:
            if selected_intent not in INTENTS:
                raise ValueError("Unknown intent")
            state.trace["routing"] = "explicit user selection"
            return self.activate(state, selected_intent, text)
        if state.choices:
            chosen = next(
                (key for key in state.choices if command in {normalize(INTENTS[key].label), key}),
                None,
            )
            if command in {"1", "2", "3"} and int(command) <= len(state.choices):
                chosen = state.choices[int(command) - 1]
            if chosen:
                return self.activate(state, chosen, state.clarification_text)
        if state.active and command in {"고마워", "감사합니다", "고마워요", "감사", "thanks"}:
            return self.reply(state, "천만에요. 진행 중인 문의는 그대로 유지할게요.")
        if state.active:
            frame = state.active
            if frame.phase == "confirming" and command in YES:
                intent = INTENTS[frame.intent]
                message, cards = execute(intent, frame.slots, state.store)
                return self.finish(state, message, cards)
            expected = frame.missing[0] if frame.missing else None
            values = extract(text, expected)
            useful = {k: v for k, v in values.items() if k in INTENTS[frame.intent].required_slots}
            correction = command.startswith(("아니", "정정", "수정")) and bool(useful)
            bare = bool(useful) and not TASK_CUE.search(text)
            if correction or bare or (expected and is_slot_reply(text, expected, values)):
                frame.slots.update(useful)
                frame.phase = "collecting"
                state.choices = []
                return self.advance(state, prefix="수정했어요. " if correction else "")
            if command in YES:
                return self.advance(state)
            if expected and self.invalid_slot(text, expected):
                return self.reply(
                    state,
                    "입력 형식을 확인해주세요. " + SLOTS[expected][1],
                    suggestions=[SLOTS[expected][2]],
                )
        # A bare product after search is an explicit result selection, not a fresh intent guess.
        values = extract(text)
        if not state.active and not TASK_CUE.search(text) and state.last_completed:
            last = state.last_completed.intent
            if "product" in values and INTENTS[last].category == "products":
                target = last if INTENTS[last].handler == "product" else "product_detail"
                state.trace["routing"] = "product follow-up"
                return self.activate(state, target, text)
            if "order_id" in values and "order_id" in INTENTS[last].required_slots:
                state.trace["routing"] = "order follow-up"
                return self.activate(state, last, text)
        frame = state.active or state.last_completed
        context = (
            f"{INTENTS[frame.intent].label}; "
            + ", ".join(
                f"{k}={v}" for k, v in frame.slots.items() if k in {"product", "order_id", "query"}
            )
            if frame
            else ""
        )
        # Candidate scorers can over-weight an old task even when instructed otherwise.
        # Only send the previous task when the utterance explicitly refers to it.
        reference_context = context[:180] if any(ref in text for ref in REFER) else ""
        result = await self.classifier.classify(text, reference_context)
        state.trace = {**result.trace, "candidates": result.candidates}
        if result.intent is None:
            state.clarification_text = text
            state.choices = [candidate["id"] for candidate in result.candidates]
            return self.reply(
                state,
                (
                    "어떤 문의인지 한 번만 더 알려주세요. 아래 업무를 "
                    "선택하거나 다르게 말씀해주세요."
                ),
                suggestions=[INTENTS[key].label for key in state.choices]
                or ["배송 도착 예정", "상품 검색", "상담사 연결"],
            )
        return self.activate(state, result.intent, text)

    @staticmethod
    def invalid_slot(text: str, expected: str) -> bool:
        if TASK_CUE.search(text):
            return False
        return expected in {
            "order_id",
            "quantity",
            "date",
            "email",
            "phone",
            "coupon",
            "channel",
            "option",
            "device",
            "product",
            "event",
        }

    def activate(self, state: Conversation, intent_id: str, text: str) -> dict:
        prior = state.active or state.last_completed
        if state.active and state.active.intent != intent_id:
            state.suspended.append(state.active)
            state.suspended = state.suspended[-8:]
            state.active = None
        if not state.active:
            state.active = Frame(intent_id)
            if prior and any(ref in text for ref in REFER):
                state.active.slots.update(
                    {
                        key: value
                        for key, value in prior.slots.items()
                        if key in {"order_id", "product"}
                        and key in INTENTS[intent_id].required_slots
                    }
                )
        values = extract(text)
        allowed = INTENTS[intent_id].required_slots
        state.active.slots.update({key: value for key, value in values.items() if key in allowed})
        state.active.phase = "collecting"
        state.choices = []
        state.clarification_text = ""
        return self.advance(state)

    def advance(self, state: Conversation, prefix: str = "") -> dict:
        frame = state.active
        assert frame is not None
        intent = INTENTS[frame.intent]
        if "order_id" in frame.slots and frame.slots["order_id"] not in state.store.orders:
            del frame.slots["order_id"]
            return self.reply(
                state,
                "그 주문은 없어요. ORD-1001, ORD-1002, ORD-1003 중 선택해주세요.",
                suggestions=["ORD-1001", "ORD-1002", "ORD-1003"],
            )
        if frame.missing:
            slot = frame.missing[0]
            return self.reply(
                state,
                prefix + f"{intent.label}을 도와드릴게요. " + SLOTS[slot][1],
                suggestions=[SLOTS[slot][2]],
            )
        if intent.confirmation:
            frame.phase = "confirming"
            summary = " / ".join(f"{SLOTS[key][0]}: {value}" for key, value in frame.slots.items())
            return self.reply(
                state,
                prefix + f"{intent.label}\n{summary}\n이 내용으로 데모 접수할까요? "
                "실제 주문 변경이나 상담 연결은 하지 않아요.",
                suggestions=["네", "취소"],
            )
        message, cards = execute(intent, frame.slots, state.store)
        if intent.handler == "search" and len(cards) == 1:
            frame.slots["product"] = cards[0]["name"]
        return self.finish(state, prefix + message, cards)

    def finish(self, state: Conversation, text: str, cards: list[dict]) -> dict:
        assert state.active is not None
        state.active.phase = "completed"
        state.last_completed = state.active
        state.active = None
        suggestions = ["이어서"] if state.suspended else []
        return self.reply(state, text, cards, suggestions)

    @staticmethod
    def reply(
        state: Conversation,
        text: str,
        cards: list[dict] | None = None,
        suggestions: list[str] | None = None,
    ) -> dict:
        entry = {
            "role": "assistant",
            "text": text,
            "cards": cards or [],
            "suggestions": suggestions or [],
        }
        state.messages.append(entry)
        state.messages = state.messages[-30:]
        return {
            "reply": entry,
            "messages": state.messages,
            "state": state.snapshot(),
            "trace": state.trace,
        }
