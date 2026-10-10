import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select

from app.context import RequestContext
from app.db import DamageClaim, Order, Refund, RefundReview
from app.policy import evaluate, note_describes_damage, search_policy


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
    if decision["status"] == "needs_review":
        return open_review(arguments, ctx)
    if decision["status"] == "pending_evidence":
        return record_damage(arguments, ctx)
    if decision["status"] != "eligible":
        decision["order_id"] = order.order_id
        return decision

    return _issue_refund(
        ctx,
        order,
        Decimal(decision["amount"]),
        (arguments.get("reason") or "customer request").strip(),
    )


def open_review(arguments: dict, ctx: RequestContext) -> dict:
    order = _owned_order(ctx, arguments.get("order_id", ""))
    if order is None:
        return {"order_id": arguments.get("order_id", ""), "status": "not_found", "message": "Order not found."}

    existing_refund = ctx.db.scalar(select(Refund).where(Refund.order_id == order.order_id))
    if existing_refund is not None:
        return {
            "status": "already_refunded",
            "order_id": order.order_id,
            "refund_id": existing_refund.refund_id,
            "message": f"Refund {existing_refund.refund_id} was already issued for {order.order_id}.",
        }

    existing = ctx.db.scalar(select(RefundReview).where(RefundReview.order_id == order.order_id))
    if existing is not None:
        return _review_payload(existing)

    decision = _decide(order, ctx, arguments)
    if decision["status"] != "needs_review":
        decision["order_id"] = order.order_id
        return decision

    review = RefundReview(
        review_id=f"REV-{order.order_id.removeprefix('ORD-')}-{uuid.uuid4().hex[:6].upper()}",
        order_id=order.order_id,
        customer_id=ctx.customer_id,
        reason=(arguments.get("reason") or "customer request").strip(),
        status="pending",
        detail=decision["message"],
        created_at=datetime.now(timezone.utc),
    )
    ctx.db.add(review)
    ctx.db.flush()
    return _review_payload(review)


def record_damage(arguments: dict, ctx: RequestContext) -> dict:
    order = _owned_order(ctx, arguments.get("order_id", ""))
    if order is None:
        return {"order_id": arguments.get("order_id", ""), "status": "not_found", "message": "Order not found."}

    existing_refund = ctx.db.scalar(select(Refund).where(Refund.order_id == order.order_id))
    if existing_refund is not None:
        return {
            "status": "already_refunded",
            "order_id": order.order_id,
            "refund_id": existing_refund.refund_id,
            "message": f"Refund {existing_refund.refund_id} was already issued for {order.order_id}.",
        }

    existing = ctx.db.scalar(select(DamageClaim).where(DamageClaim.order_id == order.order_id))
    if existing is not None:
        return _damage_payload(existing)

    decision = _decide(order, ctx, {**arguments, "damage_note": ""})
    if decision["status"] != "pending_evidence":
        decision["order_id"] = order.order_id
        return decision

    claim = DamageClaim(
        claim_id=f"DMG-{order.order_id.removeprefix('ORD-')}-{uuid.uuid4().hex[:6].upper()}",
        order_id=order.order_id,
        customer_id=ctx.customer_id,
        reason=(arguments.get("reason") or "damaged on arrival").strip(),
        note="",
        status="pending_evidence",
        created_at=datetime.now(timezone.utc),
    )
    ctx.db.add(claim)
    ctx.db.flush()
    return _damage_payload(claim)


def add_damage_note(arguments: dict, ctx: RequestContext) -> dict:
    order = _owned_order(ctx, arguments.get("order_id", ""))
    if order is None:
        return {"order_id": arguments.get("order_id", ""), "status": "not_found", "message": "Order not found."}
    note = (arguments.get("note") or "").strip()
    if not note_describes_damage(note):
        return record_damage(arguments, ctx)

    claim = ctx.db.scalar(select(DamageClaim).where(DamageClaim.order_id == order.order_id))
    if claim is None:
        claim = DamageClaim(
            claim_id=f"DMG-{order.order_id.removeprefix('ORD-')}-{uuid.uuid4().hex[:6].upper()}",
            order_id=order.order_id,
            customer_id=ctx.customer_id,
            reason=(arguments.get("reason") or "damaged on arrival").strip(),
            note=note,
            status="noted",
            created_at=datetime.now(timezone.utc),
        )
        ctx.db.add(claim)
    else:
        claim.note = note
        claim.status = "noted"
    ctx.db.flush()

    decision = _decide(
        order,
        ctx,
        {"reason": arguments.get("reason") or "damaged on arrival", "damage_note": note},
    )
    decision["order_id"] = order.order_id
    decision["claim_id"] = claim.claim_id
    decision["message"] = f"Damage note saved on {claim.claim_id}. {decision['message']}"
    return decision


def damage_status(arguments: dict, ctx: RequestContext) -> dict:
    order_id = (arguments.get("order_id") or "").strip()
    query = select(DamageClaim).where(DamageClaim.customer_id == ctx.customer_id)
    if order_id:
        query = query.where(DamageClaim.order_id == order_id)
    claims = ctx.db.scalars(query.order_by(DamageClaim.created_at)).all()
    if not claims:
        return {"claims": [], "message": "You have no damage claims."}
    return {"claims": [_damage_payload(claim) for claim in claims]}


def _damage_payload(claim: DamageClaim) -> dict:
    if claim.status == "pending_evidence":
        message = (
            f"Claim {claim.claim_id} for {claim.order_id} is waiting for a damage note. "
            "Describe what was damaged."
        )
    else:
        message = f"Claim {claim.claim_id} for {claim.order_id} has a damage note: {claim.note}"
    return {
        "status": claim.status,
        "order_id": claim.order_id,
        "claim_id": claim.claim_id,
        "message": message,
    }


def decide_review(arguments: dict, ctx: RequestContext) -> dict:
    if ctx.role != "reviewer":
        return {"status": "forbidden", "message": "Only a reviewer can decide a refund review."}

    order_id = (arguments.get("order_id") or "").strip()
    review = ctx.db.scalar(select(RefundReview).where(RefundReview.order_id == order_id))
    if review is None:
        return {"status": "not_found", "order_id": order_id, "message": f"There is no refund review for {order_id}."}
    if review.status != "pending":
        return _review_payload(review)

    decision = (arguments.get("decision") or "").strip().lower()
    if decision == "reject":
        review.status = "rejected"
        review.detail = "A reviewer rejected this refund."
        ctx.db.flush()
        return _review_payload(review)
    if decision != "approve":
        return {
            "status": "pending",
            "order_id": order_id,
            "message": f"Review {review.review_id} is still pending. Say approve or reject.",
        }

    amount_text = (arguments.get("amount") or "").strip()
    if not amount_text:
        return {
            "status": "pending",
            "order_id": order_id,
            "review_id": review.review_id,
            "message": f"Review {review.review_id} is still pending. Include the refund amount.",
        }
    order = ctx.db.get(Order, order_id)
    try:
        amount = Decimal(amount_text).quantize(Decimal("0.01"))
    except Exception:
        return {"status": "pending", "order_id": order_id, "message": "The refund amount needs to be a number, for example 10.00."}
    if order is None or amount <= 0 or amount > Decimal(order.total_amount):
        limit = f"{Decimal(order.total_amount):.2f}" if order is not None else "0.00"
        return {
            "status": "pending",
            "order_id": order_id,
            "message": f"The amount must be greater than 0 and no more than ${limit}.",
        }

    existing_refund = ctx.db.scalar(select(Refund).where(Refund.order_id == order.order_id))
    if existing_refund is not None:
        review.status = "approved"
        review.detail = f"Refund {existing_refund.refund_id} was already issued for {order.order_id}."
        ctx.db.flush()
        return _review_payload(review)

    issued = _issue_refund(ctx, order, amount, review.reason or "customer request")
    review.status = "approved"
    review.detail = issued["message"]
    ctx.db.flush()
    payload = _review_payload(review)
    payload["refund_id"] = issued["refund_id"]
    payload["amount"] = issued["amount"]
    return payload


def list_reviews(arguments: dict, ctx: RequestContext) -> dict:
    del arguments
    if ctx.role != "reviewer":
        return {"reviews": [], "message": "Only a reviewer can list refund reviews."}
    reviews = ctx.db.scalars(
        select(RefundReview).where(RefundReview.status == "pending").order_by(RefundReview.created_at)
    ).all()
    if not reviews:
        return {"reviews": [], "message": "There are no pending refund reviews."}
    rows = []
    for review in reviews:
        order = ctx.db.get(Order, review.order_id)
        product = order.product_name if order is not None else "item"
        rows.append(
            {
                "review_id": review.review_id,
                "order_id": review.order_id,
                "customer_id": review.customer_id,
                "product_name": product,
                "message": f"{review.order_id} ({product}) for {review.customer_id} is pending as {review.review_id}.",
            }
        )
    return {"reviews": rows}


def refund_status(arguments: dict, ctx: RequestContext) -> dict:
    order_id = (arguments.get("order_id") or "").strip()
    if order_id:
        order = _owned_order(ctx, order_id)
        if order is None:
            return {"refunds": [], "order_id": order_id, "status": "not_found", "message": "Order not found."}
        refund = ctx.db.scalar(select(Refund).where(Refund.order_id == order.order_id))
        if refund is None:
            return {
                "refunds": [],
                "order_id": order.order_id,
                "message": f"There is no refund for {order.order_id}.",
            }
        return {"refunds": [_refund_status_payload(refund)]}

    refunds = ctx.db.scalars(
        select(Refund)
        .join(Order, Refund.order_id == Order.order_id)
        .where(Order.customer_id == ctx.customer_id)
        .order_by(Refund.created_at)
    ).all()
    if not refunds:
        return {"refunds": [], "message": "You have no refunds."}
    return {"refunds": [_refund_status_payload(refund) for refund in refunds]}


def _refund_status_payload(refund: Refund) -> dict:
    amount = f"{Decimal(refund.amount):.2f}"
    eta = refund.eta.isoformat()
    return {
        "status": "issued",
        "order_id": refund.order_id,
        "refund_id": refund.refund_id,
        "amount": amount,
        "eta": eta,
        "message": (
            f"Refund {refund.refund_id} for {refund.order_id} is ${amount}. "
            f"It should appear by {eta}."
        ),
    }


def review_status(arguments: dict, ctx: RequestContext) -> dict:
    order_id = (arguments.get("order_id") or "").strip()
    query = select(RefundReview).where(RefundReview.customer_id == ctx.customer_id)
    if order_id:
        query = query.where(RefundReview.order_id == order_id)
    reviews = ctx.db.scalars(query.order_by(RefundReview.created_at)).all()
    if not reviews:
        return {"reviews": [], "message": "You have no refund reviews."}
    return {"reviews": [_review_payload(review) for review in reviews]}


def _review_payload(review: RefundReview) -> dict:
    return {
        "status": review.status,
        "order_id": review.order_id,
        "review_id": review.review_id,
        "message": (
            f"Review {review.review_id} for {review.order_id} is {review.status}. {review.detail}"
        ),
    }


def _issue_refund(ctx: RequestContext, order: Order, amount: Decimal, reason: str) -> dict:
    eta = date.today() + timedelta(days=5)
    refund = Refund(
        refund_id=f"REF-{order.order_id.removeprefix('ORD-')}-{uuid.uuid4().hex[:6].upper()}",
        order_id=order.order_id,
        amount=amount,
        reason=reason,
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
