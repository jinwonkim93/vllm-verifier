"Conservative entity extraction for the finite demo inventory and typed slots."

import datetime as dt
import json
import re
from pathlib import Path

from .catalog import INTENTS

FIXTURES = json.loads(Path(__file__).with_name("fixtures.json").read_text())
PREFIXES = {
    "reason": "사유",
    "address": "주소",
    "instructions": "요청",
    "name": "이름",
    "message": "문구",
    "query": "검색어",
}
TASK_CUE = re.compile(
    r"연결|조회|검색|추천|언제|어디|재고|이벤트|보여|알려|바꾸|변경|취소|환불|반품|교환|로그인|싶어|싶어요|해줘"
)


def normalize(text: str) -> str:
    return re.sub(r"[\s.!?。]+", "", text).lower()


def is_request(text: str) -> bool:
    """Keep questions and named tasks out of implicit free-text slot filling."""
    command = normalize(text)
    return bool(
        TASK_CUE.search(text)
        or re.search(
            r"[?？]|(?:뭐|무엇|어떻게|얼마|몇시|몇 시|혹시)"
            r"|(?:인가요|나요|까요|있어|없어|돼요|되나요|가능해|궁금|하고 싶|해주세요)",
            text,
        )
        or command in {normalize(intent.label) for intent in INTENTS.values()}
    )


def invalid_slot_attempt(text: str, expected: str) -> bool:
    """Reprompt only recognizable malformed values, never arbitrary new utterances."""
    if is_request(text):
        return False
    patterns = {
        "order_id": r"(?:ORD[- ]?)?[0-9-]+",
        "quantity": r"-?\d+\s*(?:개)?",
        "date": r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",
        "email": r"\S*@\S*",
        "phone": r"[+\d][\d -]{5,}",
    }
    pattern = patterns.get(expected)
    return bool(pattern and re.fullmatch(pattern, text.strip(), re.I))


def extract(text: str, expected: str | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    match = re.search(r"\bORD[- ]?(\d{4})\b", text, re.I)
    if not match and expected == "order_id":
        match = re.fullmatch(r"\s*(\d{4})\s*", text)
    if match:
        result["order_id"] = "ORD-" + match[1]
    products = []
    for product in FIXTURES["products"]:
        # Short aliases such as '티' must not match arbitrary Korean words.
        names = [product["name"], product["id"], *[a for a in product["aliases"] if len(a) > 1]]
        matches = [
            text.lower().find(name.lower()) for name in names if name.lower() in text.lower()
        ]
        if matches:
            products.append((min(matches), product["name"]))
    products.sort()
    if products:
        result["product"] = products[0][1]
        if len(products) > 1:
            result["other_product"] = products[1][1]
        elif expected == "other_product":
            result["other_product"] = result.pop("product")
    for event in FIXTURES["events"]:
        if event["name"] in text:
            result["event"] = event["name"]
            break
    match = re.search(r"\b(?:WELCOME10|FALL20)\b", text, re.I)
    if match:
        result["coupon"] = match[0].upper()
    for region in ("서울", "부산", "제주", "대전", "대구", "인천", "해외"):
        if region in text:
            result["region"] = region
            break
    for slot, names in {
        "device": ("아이폰", "갤럭시", "노트북", "아이패드", "안드로이드"),
        "payment_method": ("카드", "계좌이체", "간편결제", "무통장"),
        "channel": ("채팅", "전화"),
        "option": ("화이트", "블랙", "베이지", "그린"),
    }.items():
        for name in names:
            if name in text:
                result[slot] = name
                break
    if expected == "option" and text.strip().upper() in {"S", "M", "L"}:
        result["option"] = text.strip().upper()
    match = re.search(r"(?<!\d)([1-9]\d?)\s*개", text)
    if not match and expected == "quantity":
        match = re.fullmatch(r"\s*([1-9]\d?)\s*", text)
    if match:
        result["quantity"] = match[1]
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if match:
        try:
            dt.date.fromisoformat(match[0])
            result["date"] = match[0]
        except ValueError:
            pass
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if match:
        result["email"] = match[0]
    match = re.search(r"\b010-\d{4}-\d{4}\b", text)
    if match:
        result["phone"] = match[0]
    if "끄" in text or "해제" in text:
        result["notification"] = "끄기"
    elif "켜" in text or "켜기" in text:
        result["notification"] = "켜기"
    for slot, prefix in PREFIXES.items():
        match = re.search(prefix + r"\s*[:：]\s*(.+)", text)
        if match:
            result[slot] = match[1].strip()[:200]
    match = re.search(r"(.+?)\s*(?:찾아|검색해|추천해)", text)
    if match:
        query = re.sub(r"^(?:혹시|그럼|저는|나는)\s*", "", match[1]).strip()
        if query not in {"상품", "제품", "다른 상품", "다른 제품"}:
            result["query"] = query[:100]
    if expected in PREFIXES and expected not in result and not is_request(text):
        if 1 <= len(text.strip()) <= 100 and normalize(text) not in {
            "네",
            "아니",
            "아니요",
            "몰라",
            "모르겠어",
        }:
            result[expected] = text.strip()
    return result


def is_slot_reply(text: str, expected: str, values: dict[str, str]) -> bool:
    if expected not in values:
        return False
    # Explicitly prefixed fields and bare entities are answers, not new tasks.
    if expected in PREFIXES and re.match(PREFIXES[expected] + r"\s*[:：]", text):
        return True
    return not is_request(text)
