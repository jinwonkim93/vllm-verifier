"""Declarative intent and slot contracts for the customer-service example."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Intent:
    id: str
    category: str
    label: str
    description: str
    required_slots: tuple[str, ...]
    handler: str
    confirmation: bool
    example: str


DATA = json.loads(Path(__file__).with_name("catalog.json").read_text())
CATEGORIES = {item["id"]: item for item in DATA["categories"]}
INTENTS = {
    item["id"]: Intent(**{**item, "required_slots": tuple(item["required_slots"])})
    for item in DATA["intents"]
}

# Prompts and examples are also displayed by the UI; never request credentials.
SLOTS = {
    "order_id": ("주문번호", "주문번호를 알려주세요. 예: ORD-1001", "ORD-1001"),
    "product": ("상품", "어떤 상품인가요? 에어버즈, 트래블백, 텀블러 등이 있어요.", "에어버즈"),
    "other_product": ("비교 상품", "어떤 상품과 비교할까요?", "미니 스피커"),
    "query": ("검색 조건", "어떤 상품을 찾으세요? 예: 무선 이어폰, 여행용 가방", "무선 이어폰"),
    "reason": ("사유", "사유를 알려주세요. 예: 사유: 단순 변심", "사유: 단순 변심"),
    "quantity": ("수량", "몇 개로 변경할까요? 1~99개를 입력해주세요.", "2개"),
    "option": ("옵션", "원하는 옵션을 알려주세요. 예: 블랙, 화이트, M", "블랙"),
    "address": (
        "주소",
        "데모용 주소를 입력해주세요. 예: 주소: 서울시 예시로 12",
        "주소: 서울시 예시로 12",
    ),
    "instructions": (
        "배송 요청",
        "전달할 내용을 적어주세요. 예: 요청: 문 앞에 놓아주세요",
        "요청: 문 앞",
    ),
    "region": ("지역", "지역을 알려주세요. 예: 서울, 제주", "서울"),
    "date": ("날짜", "날짜를 YYYY-MM-DD 형식으로 알려주세요. 예: 2026-10-01", "2026-10-01"),
    "device": ("사용 기기", "연결할 기기를 알려주세요. 예: 아이폰, 갤럭시, 노트북", "아이폰"),
    "event": ("이벤트", "어떤 이벤트인가요? 가을맞이, 신규회원, 주말 특가가 있어요.", "가을맞이"),
    "coupon": ("쿠폰 코드", "쿠폰 코드를 알려주세요. 예: WELCOME10", "WELCOME10"),
    "payment_method": ("결제 수단", "결제 수단을 알려주세요. 예: 카드, 계좌이체, 간편결제", "카드"),
    "email": (
        "이메일",
        "데모용 이메일을 입력해주세요. 실제 개인정보는 입력하지 마세요.",
        "demo@example.com",
    ),
    "phone": ("전화번호", "데모용 번호를 입력해주세요. 예: 010-0000-0000", "010-0000-0000"),
    "name": ("표시 이름", "변경할 이름을 입력해주세요. 예: 이름: 데모 고객", "이름: 데모 고객"),
    "notification": ("알림 설정", "마케팅 알림을 켤까요, 끌까요?", "끄기"),
    "channel": ("상담 채널", "채팅과 전화 중 어떤 상담을 원하세요?", "채팅"),
    "message": (
        "카드 문구",
        "선물 카드 문구를 입력해주세요. 예: 문구: 생일 축하해",
        "문구: 생일 축하해",
    ),
}

assert len(INTENTS) == 100 and len(CATEGORIES) == 10
assert all(set(i.required_slots) <= SLOTS.keys() for i in INTENTS.values())
