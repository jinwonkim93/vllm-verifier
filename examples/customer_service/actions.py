"""Local fixtures only; no external customer-service actions."""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import Intent
from .slots import FIXTURES

FAQ = json.loads(Path(__file__).with_name("faqs.json").read_text())


@dataclass
class DemoStore:
    orders: dict = field(default_factory=lambda: copy.deepcopy(FIXTURES["orders"]))
    tickets: list = field(default_factory=list)
    profile: dict = field(default_factory=lambda: copy.deepcopy(FIXTURES["customer"]))


def product_named(name: str) -> dict | None:
    return next((p for p in FIXTURES["products"] if p["name"] == name), None)


def execute(
    intent: Intent, slots: dict[str, str], store: DemoStore
) -> tuple[str, list[dict[str, Any]]]:
    key, handler = intent.id, intent.handler
    order = store.orders.get(slots.get("order_id", ""))
    if "order_id" in intent.required_slots and order is None:
        return ("그 주문은 데모에 없어요. ORD-1001, ORD-1002, ORD-1003을 조회할 수 있어요."), []
    if handler == "faq":
        return FAQ[key], []
    if handler == "orders":
        return "데모 고객의 주문 목록이에요.", [dict(id=k, **v) for k, v in store.orders.items()]
    if handler in {"events", "event"}:
        events = FIXTURES["events"]
        if handler == "event":
            events = [e for e in events if e["name"] == slots["event"]]
        return "2026-09-23 기준 데모 이벤트예요.", events
    if handler in {"coupons", "coupon"}:
        coupons = [
            dict(code=k, **v)
            for k, v in FIXTURES["coupons"].items()
            if handler == "coupons" or k == slots["coupon"]
        ]
        return "쿠폰의 사용 조건과 유효기간을 확인해주세요.", coupons
    if handler == "search":
        query = slots["query"].replace(" ", "").lower()
        found = [
            p
            for p in FIXTURES["products"]
            if any(
                word.replace(" ", "").lower() in query or query in word.replace(" ", "").lower()
                for word in [p["name"], *p["tags"], *p["aliases"]]
                if len(word) > 1
            )
        ]
        return (
            f"“{slots['query']}” 검색 결과 {len(found)}개예요. "
            "상품명을 말하면 상세 정보를 볼 수 있어요."
            if found
            else ("일치하는 상품이 없어요. 이어폰, 가방, 텀블러, 티셔츠, 스피커로 검색해보세요.")
        ), found
    if handler in {"product", "compare"}:
        product = product_named(slots["product"])
        if product is None:
            return "데모 상품 목록에서 상품을 선택해주세요.", []
        if handler == "compare":
            other = product_named(slots["other_product"])
            return "두 상품의 가격과 사양을 비교해보세요.", [product, other] if other else [product]
        message = {
            "product_stock": f"{product['name']} 재고는 {product['stock']}개예요.",
            "product_restock": f"{product['name']} 재입고: {product['restock']}.",
            "product_size": product["sizes"],
            "product_compatibility": (
                "블루투스를 지원하는 기기와 연결할 수 있어요. 전용 앱의 "
                "일부 기능은 기기별로 다를 수 있어요."
            )
            if product["id"] in {"P001", "P005"}
            else "전자기기 연결 기능이 없는 상품이에요.",
            "product_authenticity": "데모 상품은 공식 판매 상품으로 가정해요. "
            "실제 정품 판별 결과는 아니에요.",
            "product_reviews": f"데모 구매자 평점: {product['rating']}/5.",
            "preorder": "이 상품은 현재 예약 판매 일정이 없어요.",
            "warranty_info": product["warranty"],
            "installation_help": product["manual"],
            "product_manual": product["manual"],
            "gift_wrap": "데모 선물 포장은 2,000원이에요. 실제 결제나 포장 주문은 하지 않아요.",
        }.get(key, product["description"])
        return message, [product]
    if handler == "order":
        messages = {
            "delivery_eta": f"도착 예정: {order['eta']}.",
            "delivery_tracking": f"현재 위치: {order['location']}. 운송장: {order['tracking']}.",
            "delivery_delay": (
                "데모 물류 현황을 확인했어요. 표시된 예정 시간은 고정 "
                "예시이며 실시간 택배 정보가 아니에요."
            ),
            "delivery_missing": (
                "문 앞·경비실·택배 보관함을 확인해주세요. 찾지 못하면 "
                "상담사 연결로 배송 조사를 요청할 수 있어요."
            ),
            "delivery_wrong": (
                "받은 상품과 주문 내역을 비교해주세요. 오배송은 상담사를 "
                "통해 무료 교환을 요청할 수 있어요."
            ),
            "delivery_missed": "택배기사의 부재 안내를 확인하고 배송 요청사항을 지정해주세요.",
            "refund_status": (
                "접수된 실제 환불은 없어요. 이 데모에서는 반품 접수 흐름만 체험할 수 있어요."
            ),
            "payment_duplicate": (
                "실제 결제 원장을 조회하지 않는 데모예요. 주문 금액을 "
                "확인하고 실제 중복 승인은 판매처에 문의해주세요."
            ),
            "payment_deadline": "데모 미입금 주문의 입금 기한은 주문 다음 날 23:59예요.",
            "virtual_account": (
                "데모이므로 실제 입금 계좌를 제공하지 않아요. 결제 화면에서 "
                "계좌를 확인하는 흐름이에요."
            ),
            "deposit_confirmation": "표시된 주문 상태만 확인할 수 있어요. "
            "실제 은행 입금은 조회하지 않아요.",
            "cash_receipt": "데모 현금영수증 발급 요청 화면이에요. 실제 국세청 발급은 하지 않아요.",
            "tax_invoice": (
                "데모 세금계산서 안내예요. 실제 발급은 판매처에 사업자 정보를 제출해야 해요."
            ),
            "order_confirmation": "주문 확인서에 사용할 데모 주문 내역이에요.",
            "order_receipt": "데모 구매 내역이에요. 법적 증빙 영수증이 아니에요.",
            "order_split": "출고 전이라면 분리 배송을 문의할 수 있어요. "
            "배송 중 주문은 나눌 수 없어요.",
            "product_damage": (
                "사용을 멈추고 포장과 제품 사진을 보관해주세요. 반품·교환 "
                "또는 상담사 연결로 이어갈 수 있어요."
            ),
            "missing_parts": "포장 내부를 다시 확인해주세요. "
            "빠진 구성품은 상담사에게 문의할 수 있어요.",
        }
        return messages.get(key, f"주문 상태: {order['status']}."), [
            dict(id=slots["order_id"], **order)
        ]
    if handler in {"mutation", "handoff"}:
        if (
            key
            in {
                "order_cancel",
                "order_quantity",
                "order_option",
                "delivery_address",
                "coupon_apply",
            }
            and order["status"] != "결제 완료"
        ):
            return (
                (
                    "이 주문은 이미 출고되어 해당 변경을 할 수 없어요. "
                    "ORD-1002는 출고 전 주문이에요."
                ),
                [],
            )
        if key in {"order_option", "exchange_request"}:
            product = product_named(order["product"])
            if slots["option"] not in product["options"]:
                return (
                    f"선택 가능한 옵션은 {', '.join(product['options'])}예요. "
                    "옵션을 수정해 다시 요청해주세요.",
                    [],
                )
        if key == "order_cancel":
            order["status"] = "취소 완료"
        elif key == "order_quantity":
            order["quantity"] = int(slots["quantity"])
        elif key == "order_option":
            order["option"] = slots["option"]
        elif key == "delivery_address":
            order["address"] = slots["address"]
        elif key == "delivery_instructions":
            order["instructions"] = slots["instructions"]
        elif key in {"email_change", "phone_change", "profile_change", "notification_settings"}:
            store.profile.update(slots)
        ticket = dict(id=f"DEMO-{len(store.tickets) + 1:04d}", intent=key, slots=dict(slots))
        store.tickets.append(ticket)
        return (
            f"{intent.label} 데모 접수를 완료했어요. 접수번호 {ticket['id']}. "
            "실제 주문·계정 변경, 결제 또는 상담 연결은 발생하지 않아요."
        ), [ticket]
    raise ValueError(f"Unknown handler: {handler}")
