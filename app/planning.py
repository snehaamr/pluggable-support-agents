import json
import re

from sqlalchemy import select

from app.context import RequestContext
from app.db import Order
from app.turns import ModelTurn, ToolCall


def plan_supervisor(messages: list[dict], ctx: RequestContext) -> ModelTurn:
    user_text = latest_user(messages)
    called = tool_names(messages)
    preference = preference_fact(user_text)
    route = route_message(user_text)

    if preference and "remember" not in called:
        return ModelTurn(tool_calls=[ToolCall("remember", {"fact": preference})])

    if route is None:
        if preference:
            return ModelTurn(content="Got it. I'll remember that.")
        return ModelTurn(
            content="I can look up an order or start a refund. Share an order id, or ask about your orders."
        )

    if "search_agents" not in called:
        query = "refund eligibility policy" if route == "refund" else "order status lookup"
        return ModelTurn(tool_calls=[ToolCall("search_agents", {"query": query})])

    if "delegate" not in called:
        found = last_payload(messages)
        agents = found.get("agents") if isinstance(found, dict) else None
        if not agents:
            return ModelTurn(content="I couldn't find a specialist for that.")
        return ModelTurn(
            tool_calls=[
                ToolCall(
                    "delegate",
                    {"agent_name": agents[0]["name"], "task": user_text},
                )
            ]
        )

    payload = last_payload(messages)
    reply = payload.get("reply", "") if isinstance(payload, dict) else ""
    if preference:
        reply = f"I'll remember that. {reply}".strip()
    return ModelTurn(content=reply or "I couldn't complete that request.")


def plan_order(messages: list[dict], ctx: RequestContext) -> ModelTurn:
    if "check_order_details" not in tool_names(messages):
        order_id = referenced_order_id(messages, ctx)
        if order_id:
            arguments = {"order_id": order_id}
        else:
            arguments = {"customer_id": ctx.customer_id}
        return ModelTurn(tool_calls=[ToolCall("check_order_details", arguments)])

    payload = last_payload(messages) or {}
    return ModelTurn(content=format_orders(payload))


def plan_refund(messages: list[dict], ctx: RequestContext) -> ModelTurn:
    user_text = latest_user(messages)
    if ctx.role == "reviewer":
        return _plan_reviewer(messages)
    if re.search(r"\b(approve|reject|deny)\b", user_text, re.I) and re.search(r"\breview\b", user_text, re.I):
        return ModelTurn(content="A reviewer has to decide that refund.")
    called = tool_names(messages)
    order_id = referenced_order_id(messages, ctx)
    reason = "damaged on arrival" if re.search(r"damage|defect", user_text, re.I) else "customer request"
    damage_note = "customer reported damage" if "damage note" in user_text.lower() else ""

    if _is_eligibility_list(user_text):
        if "list_eligible" not in called:
            return ModelTurn(tool_calls=[ToolCall("list_eligible", {"reason": reason})])
        return ModelTurn(content=format_eligibility(last_payload(messages) or {}))

    if _wants_review_status(user_text):
        if "review_status" not in called:
            arguments = {"order_id": order_id} if order_id else {}
            return ModelTurn(tool_calls=[ToolCall("review_status", arguments)])
        return ModelTurn(content=_review_reply(last_payload(messages) or {}))

    if _wants_refund_action(user_text, order_id):
        if not order_id:
            return ModelTurn(content="Which order should I refund? Send the order id, for example ORD-10001.")
        if "check_eligible" not in called:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        "check_eligible",
                        {"order_id": order_id, "reason": reason, "damage_note": damage_note},
                    )
                ]
            )
        decision = last_payload(messages) or {}
        if decision.get("status") == "eligible" and "process_refund" not in called:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        "process_refund",
                        {"order_id": order_id, "reason": reason, "damage_note": damage_note},
                    )
                ]
            )
        if decision.get("status") == "needs_review" and "open_review" not in called:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        "open_review",
                        {"order_id": order_id, "reason": reason, "damage_note": damage_note},
                    )
                ]
            )
        return ModelTurn(content=_refund_reply(decision))

    if re.search(r"\b(policy|window|how long|percentage)\b", user_text, re.I):
        if "refund_policy" not in called:
            return ModelTurn(tool_calls=[ToolCall("refund_policy", {"question": user_text})])
        policy = (last_payload(messages) or {}).get("policy", "")
        return ModelTurn(content=policy or "I couldn't find the return policy.")

    return ModelTurn(content="I can check a refund, explain the return policy, or review which orders are eligible.")


def route_message(text: str) -> str | None:
    lower = text.lower()
    preference_only = bool(re.search(r"\b(i prefer|i like|my preferred)\b", lower)) and not re.search(
        r"\b(what|check|status|list|show|which|refund order|process|eligible)\b",
        lower,
    )
    if preference_only:
        return None
    if re.search(r"\b(refund|eligible|return policy|return window|review)\b", lower):
        return "refund"
    if re.search(r"\b(order|orders|status|tracking)\b", lower):
        return "order"
    return None


def preference_fact(text: str) -> str | None:
    if re.search(r"\b(i prefer|i like|my preferred)\b", text, re.I):
        return text.strip()
    return None


def referenced_order_id(messages: list[dict], ctx: RequestContext) -> str:
    """Resolve an order from this message, or from an earlier turn in the session."""
    latest = latest_user(messages)
    direct = re.search(r"ORD-\d+", latest)
    if direct:
        return direct.group(0)

    orders = ctx.db.scalars(
        select(Order).where(Order.customer_id == ctx.customer_id).order_by(Order.order_id)
    ).all()
    for order in orders:
        if re.search(rf"\b{re.escape(order.product_name)}\b", latest, re.IGNORECASE):
            return order.order_id

    if re.search(r"\b(it|that|this)\b", latest, re.IGNORECASE):
        earlier = "\n".join(
            message.get("content") or ""
            for message in messages
            if message.get("role") in {"user", "assistant"} and (message.get("content") or "") != latest
        )
        found = re.findall(r"ORD-\d+", earlier)
        if found:
            return found[-1]
    return ""


def latest_user(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def tool_names(messages: list[dict]) -> list[str]:
    return [message["name"] for message in messages if message.get("role") == "tool"]


def last_payload(messages: list[dict]):
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    return None


def format_orders(payload: dict) -> str:
    orders = payload.get("orders") or []
    if not orders:
        return "I couldn't find an order for that."
    lines = []
    for order in orders:
        delivered = order.get("delivered_at") or "no delivery date"
        lines.append(
            f"{order['order_id']} is {order['status']}: {order['product_name']} "
            f"({order['category']}) for ${order['total_amount']}, delivered {delivered}."
        )
    if len(lines) == 1:
        return lines[0]
    return "Here are the orders. " + " ".join(lines)


def format_eligibility(payload: dict) -> str:
    rows = payload.get("orders") or []
    if not rows:
        return "You have no orders to review."
    lines = [f"{row['order_id']} ({row['product_name']}): {row['message']}" for row in rows]
    return "Here's the eligibility of your orders. " + " ".join(lines)


def _is_eligibility_list(text: str) -> bool:
    lower = text.lower()
    return "eligible" in lower and bool(re.search(r"\b(which|all)\b", lower)) and not re.search(r"ORD-\d+", text)


def _plan_reviewer(messages: list[dict]) -> ModelTurn:
    user_text = latest_user(messages)
    called = tool_names(messages)
    order_match = re.search(r"ORD-\d+", user_text)
    order_id = order_match.group(0) if order_match else ""
    if re.search(r"\b(approve|accept)\b", user_text, re.I):
        decision = "approve"
    elif re.search(r"\b(reject|deny|decline)\b", user_text, re.I):
        decision = "reject"
    else:
        decision = ""

    if decision:
        if not order_id:
            return ModelTurn(content="Which order should I decide? Send the order id, for example ORD-30001.")
        if "decide_review" not in called:
            arguments = {"order_id": order_id, "decision": decision}
            amount = _review_amount(user_text)
            if amount:
                arguments["amount"] = amount
            return ModelTurn(tool_calls=[ToolCall("decide_review", arguments)])
        payload = last_payload(messages) or {}
        return ModelTurn(content=payload.get("message") or "I couldn't decide that review.")

    if re.search(r"\b(pending|waiting|which|list)\b", user_text, re.I):
        if "list_reviews" not in called:
            return ModelTurn(tool_calls=[ToolCall("list_reviews", {})])
        return ModelTurn(content=_review_reply(last_payload(messages) or {}))

    return ModelTurn(content="I can list pending reviews, or approve or reject one by order id.")


def _review_amount(text: str) -> str:
    found = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", text)
    if found:
        return found.group(1)
    found = re.search(r"\bfor\s+(\d+\.\d{2})\b", text, re.I)
    return found.group(1) if found else ""


def _wants_review_status(text: str) -> bool:
    return bool(re.search(r"\breview\b", text, re.I)) and not re.search(
        r"\b(refund order|process a refund|want a refund|need a refund)\b",
        text,
        re.I,
    )


def _review_reply(payload: dict) -> str:
    reviews = payload.get("reviews") or []
    if reviews:
        return " ".join(review.get("message") or "" for review in reviews).strip()
    return payload.get("message") or "You have no refund reviews."


def _wants_refund_action(text: str, order_id: str) -> bool:
    if re.search(r"\b(refund order|process a refund|want a refund|need a refund)\b", text, re.I):
        return True
    return bool(order_id) and bool(re.search(r"\brefund\b", text, re.I))


def _refund_reply(decision: dict) -> str:
    message = decision.get("message") or "I couldn't complete that refund."
    if decision.get("status") in {"approved", "pending"}:
        return message
    order_id = decision.get("order_id")
    if order_id and not message.startswith(order_id):
        return f"{order_id}: {message}"
    return message
