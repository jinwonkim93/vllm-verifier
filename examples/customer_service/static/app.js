"use strict";
const $ = (id) => document.getElementById(id);
let catalog,
  current,
  busy = false;
const labels = {
  order_id: "주문번호",
  product: "상품",
  other_product: "비교 상품",
  query: "검색 조건",
  reason: "사유",
  quantity: "수량",
  option: "옵션",
  address: "주소",
  instructions: "배송 요청",
  region: "지역",
  date: "날짜",
  device: "기기",
  event: "이벤트",
  coupon: "쿠폰",
  payment_method: "결제 수단",
  email: "이메일",
  phone: "전화번호",
  name: "이름",
  notification: "알림",
  channel: "상담 채널",
  message: "문구",
};
function node(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
async function api(path, body) {
  const res = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await res.json();
  if (!res.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "입력을 확인하고 다시 시도해주세요.",
    );
  return data;
}
function loading(value) {
  busy = value;
  document.querySelectorAll("button").forEach((b) => (b.disabled = value));
  $("message").disabled = value;
  $("activity").textContent = value
    ? "문의와 대화 상태를 확인하고 있어요…"
    : "대화할 준비가 됐어요";
}
function renderCatalog() {
  const query = $("search").value.trim();
  $("catalog").replaceChildren();
  for (const category of catalog.categories) {
    const intents = catalog.intents.filter(
      (i) =>
        i.category === category.id &&
        (!query ||
          (i.label + " " + i.description + " " + i.id).includes(query)),
    );
    if (!intents.length) continue;
    const group = node("details");
    group.open = !!query;
    const summary = node("summary", category.label);
    summary.append(node("span", String(intents.length)));
    group.append(summary);
    for (const intent of intents) {
      const b = node("button", intent.label, "intent-item");
      b.title = intent.description;
      b.disabled = busy;
      b.onclick = () => send(intent.label, intent.id);
      group.append(b);
    }
    $("catalog").append(group);
  }
}
function cardView(card) {
  const el = node("div", undefined, "card");
  el.append(node("strong", card.name || card.id || card.code || "안내"));
  if (card.description) el.append(node("p", card.description));
  if (card.price)
    el.append(node("strong", Number(card.price).toLocaleString() + "원"));
  const titles = {
    status: "상태",
    eta: "도착 예정",
    tracking: "운송장",
    location: "현재 위치",
    amount: "결제 금액",
    address: "배송지",
    stock: "재고",
    sizes: "크기",
    options: "옵션",
    rating: "평점",
    restock: "재입고",
    warranty: "보증",
    period: "기간",
    eligibility: "대상",
    expiry: "유효기간",
    condition: "사용 조건",
    intent: "접수 업무",
    slots: "접수 내용",
    product: "상품",
    quantity: "수량",
  };
  const dl = node("dl");
  for (const [key, title] of Object.entries(titles)) {
    if (card[key] === undefined) continue;
    dl.append(node("dt", title));
    const value = card[key];
    dl.append(
      node(
        "dd",
        Array.isArray(value)
          ? value.join(", ")
          : typeof value === "object"
            ? Object.entries(value)
                .map(([k, v]) => (labels[k] || k) + ": " + v)
                .join("\n")
            : String(value),
      ),
    );
  }
  el.append(dl);
  if (card.id && card.id.startsWith("P") && card.name) {
    const b = node("button", "이 상품 보기");
    b.onclick = () => send(card.name);
    el.append(b);
  }
  return el;
}
function render(data) {
  current = data;
  const messages = $("messages");
  if (data.messages.length) {
    messages.replaceChildren();
    for (const message of data.messages) {
      const row = node("div", undefined, "bubble-row " + message.role);
      row.append(
        node(
          "span",
          message.role === "user" ? "YOU" : "모아 · 고객센터",
          "speaker",
        ),
        node("div", message.text, "bubble"),
      );
      if (message.cards?.length) {
        const cards = node("div", undefined, "cards");
        message.cards.forEach((c) => cards.append(cardView(c)));
        row.append(cards);
      }
      messages.append(row);
    }
    messages.scrollTop = messages.scrollHeight;
  }
  const state = data.state,
    frame = state.active || state.last_completed;
  $("turn").textContent = "TURN " + state.turn;
  $("intent").textContent = frame?.label || "아직 선택되지 않았어요";
  $("phase").textContent =
    { collecting: "정보 수집 중", confirming: "확인 대기", completed: "완료" }[
      frame?.phase
    ] || "대기 중";
  $("slots").replaceChildren();
  $("slots").className =
    frame && Object.keys(frame.slots).length ? "" : "empty";
  if (frame && Object.keys(frame.slots).length) {
    for (const [key, value] of Object.entries(frame.slots)) {
      const row = node("div", undefined, "slot");
      row.append(node("span", labels[key] || key), node("strong", value));
      $("slots").append(row);
    }
  } else $("slots").textContent = "대화를 시작하면 여기에 표시돼요.";
  $("missing").textContent =
    state.active?.missing_slots.map((k) => labels[k] || k).join(" → ") ||
    "없음";
  $("suspended").textContent =
    state.suspended.map((f) => f.label).join(" · ") || "없음";
  $("raw").textContent = JSON.stringify(state, null, 2);
  const trace = data.trace || {};
  $("route").textContent =
    (trace.routing || "대화 상태 전이") +
    (trace.seconds ? " · " + (trace.seconds * 1000).toFixed(0) + "ms" : "");
  if (trace.model) $("engine").textContent = trace.model.split("/").pop();
  $("candidates").replaceChildren();
  for (const c of trace.candidates || []) {
    const row = node("div", undefined, "candidate");
    row.append(
      node("span", c.label),
      node("span", (c.probability * 100).toFixed(1) + "%"),
    );
    $("candidates").append(row);
  }
  $("suggestions").replaceChildren();
  const last = data.messages.at(-1);
  for (const text of last?.suggestions || []) {
    const b = node("button", text);
    b.onclick = () => send(text);
    $("suggestions").append(b);
  }
}
async function send(text, intent) {
  text = text.trim();
  if (!text || busy) return;
  $("error").hidden = true;
  loading(true);
  try {
    const data = await api("/api/chat", {
      text,
      ...(intent ? { intent } : {}),
    });
    render(data);
    $("message").value = "";
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
  } finally {
    loading(false);
    $("message").focus();
  }
}
$("composer").onsubmit = (e) => {
  e.preventDefault();
  send($("message").value);
};
$("search").oninput = () => renderCatalog();
document
  .querySelectorAll("[data-text]")
  .forEach((b) => (b.onclick = () => send(b.dataset.text)));
$("reset").onclick = async () => {
  if (busy) return;
  loading(true);
  try {
    await api("/api/reset", {});
    location.reload();
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
    loading(false);
  }
};
Promise.all([api("/api/catalog"), api("/api/session")])
  .then(([data, session]) => {
    catalog = data;
    renderCatalog();
    render(session);
  })
  .catch((err) => {
    $("error").textContent = err.message;
    $("error").hidden = false;
  });
