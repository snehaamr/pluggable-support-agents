import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select

from app.context import RequestContext
from app.db import Order, Refund
from app.policy import evaluate, search_policy


def check_order_details(arguments: dict, ctx: RequestContext) -> dict:
    order_id = (arguments.get("order_id") or "").strip()
    customer_id = (arguments.get("customer_id") or ctx.customer_id).strip()
    if not order_id and not customer_id:
        return {"orders": [], "message": "Provide an order id or a customer id."}

    query = select(Order).where(Order.customer_id == ctx.customer_id)
    if order_id:
        query = query.where(Order.order_id == order_id)
    elif customer_id != ctx.customer_id:
        return {"orders": [], "message": "Order not found."}

    orders = ctx.db.scalars(query.order_by(Order.order_id)).all()
    return {"orders": [_order_payload(order) for order in orders]}


def refund_policy(arguments: dict, ctx: RequestContext) -> dict:
    question = (arguments.get("question") or "").strip()
    return {"question": question, "tier": ctx.tier, "policy": search_policy(question, ctx.tier)}


def check_eligible(arguments: dict, ctx: RequestContext) -> dict:
    order = _owned_order(ctx, arguments.get("order_id", ""))
    if order is None:
        return {"order_id": arguments.get("order_id", ""), "status": "not_found", "message": "Order not found.", "amount": None}
    decision = _decide(order, ctx, arguments)
    decision["order_id"] = order.order_id
    decision["product_name"] = order.product_name
    return decision


def list_eligible(arguments: dict, ctx: RequestContext) -> dict:
    orders = ctx.db.scalars(
        select(Order).where(Order.customer_id == ctx.customer_id).order_by(Order.order_id)
    ).all()
    rows = []
    for order in orders:
        decision = _decide(order, ctx, {"reason": arguments.get("reason", "")})
        rows.append(
            {
                "order_id": order.order_id,
                "product_name": order.product_name,
                "status": decision["status"],
                "message": decision["message"],
                "amount": decision["amount"],
            }
        )
    return {"orders": rows}


def process_refund(arguments: dict, ctx: RequestContext) -> dict:
    order = _owned_order(ctx, arguments.get("order_id", ""))
    if order is None:
        return {"order_id": arguments.get("order_id", ""), "status": "not_found", "message": "Order not found."}

    existing = ctx.db.scalar(select(Refund).where(Refund.order_id == order.order_id))
    if existing is not None:
        return {
            "status": "already_refunded",
            "order_id": order.order_id,
            "refund_id": existing.refund_id,
            "amount": f"{existing.amount:.2f}",
            "eta": existing.eta.isoformat(),
            "message": f"Refund {existing.refund_id} was already issued for {order.order_id}.",
        }

    decision = _decide(order, ctx, arguments)
    if decision["status"] != "eligible":
        decision["order_id"] = order.order_id
        return decision

    eta = date.today() + timedelta(days=5)
    refund = Refund(
        refund_id=f"REF-{order.order_id.removeprefix('ORD-')}-{uuid.uuid4().hex[:6].upper()}",
        order_id=order.order_id,
        amount=Decimal(decision["amount"]),
        reason=(arguments.get("reason") or "customer request").strip(),
        eta=eta,
        created_at=datetime.now(timezone.utc),
    )
    ctx.db.add(refund)
    ctx.db.flush()
    return {
        "status": "approved",
        "order_id": order.order_id,
        "refund_id": refund.refund_id,
        "amount": f"{refund.amount:.2f}",
        "eta": eta.isoformat(),
        "message": (
            f"Refund {refund.refund_id} approved for ${refund.amount:.2f}. "
            f"It should appear by {eta.isoformat()}."
        ),
    }


def _owned_order(ctx: RequestContext, order_id: str) -> Order | None:
    order_id = (order_id or "").strip()
    if not order_id:
        return None
    order = ctx.db.get(Order, order_id)
    if order is None or order.customer_id != ctx.customer_id:
        return None
    return order


def _decide(order: Order, ctx: RequestContext, arguments: dict) -> dict:
    return evaluate(
        tier=ctx.tier,
        status=order.status,
        delivered_on=order.delivered_at,
        today=date.today(),
        amount=Decimal(order.total_amount),
        reason=arguments.get("reason") or "",
        damage_note=arguments.get("damage_note") or "",
    )


def _order_payload(order: Order) -> dict:
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "status": order.status,
        "product_name": order.product_name,
        "category": order.category,
        "total_amount": f"{Decimal(order.total_amount):.2f}",
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "contact_email": order.contact_email,
        "shipping_address": order.shipping_address,
    }
