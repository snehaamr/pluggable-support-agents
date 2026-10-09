import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.auth import find_customer
from app.context import RequestContext
from app.db import AuthSession, ChatMessage, Customer, MemoryFact, TraceSpan
from app.guard import inspect_text, mask_for_storage


router = APIRouter()
COOKIE = "support_session"
INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class LoginIn(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


@router.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "model_provider": settings.model_provider,
        "model_name": settings.model_name if settings.model_provider != "deterministic" else "deterministic",
        "order_agent_url": settings.order_agent_url,
        "refund_agent_url": settings.refund_agent_url,
    }


@router.get("/agents")
def agents(request: Request) -> dict:
    supervisor = request.app.state.supervisor
    registry = request.app.state.registry
    return {
        "agents": [
            {
                "name": supervisor.name,
                "role": "supervisor",
                "description": supervisor.description,
            },
            *[
                {
                    "name": item["name"],
                    "role": "specialist",
                    "description": item["description"],
                    "url": item["url"],
                }
                for item in registry.all()
            ],
        ]
    }


@router.post("/login")
def login(body: LoginIn, request: Request):
    db = request.app.state.session_factory()
    try:
        customer = find_customer(db, body.username.strip(), body.password)
        if customer is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token = secrets.token_urlsafe(32)
        db.add(
            AuthSession(
                token=token,
                customer_id=customer.id,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        response = JSONResponse(_customer_payload(customer))
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 12,
            path="/",
        )
        return response
    except HTTPException:
        db.rollback()
        raise
    finally:
        db.close()


@router.post("/logout")
def logout(request: Request):
    db = request.app.state.session_factory()
    try:
        token = request.cookies.get(COOKIE)
        if token:
            row = db.get(AuthSession, token)
            if row is not None:
                db.delete(row)
                db.commit()
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/")
        return response
    finally:
        db.close()


@router.get("/me")
def me(request: Request):
    db = request.app.state.session_factory()
    try:
        customer = _customer_from_cookie(request, db)
        return _customer_payload(customer)
    finally:
        db.close()


@router.post("/chat")
def chat(body: ChatIn, request: Request):
    db = request.app.state.session_factory()
    try:
        customer = _customer_from_cookie(request, db)
        session_id = body.session_id or uuid.uuid4().hex
        _claim_session(db, session_id, customer.id)

        prior_rows = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        ).all()
        facts = db.scalars(
            select(MemoryFact)
            .where(MemoryFact.customer_id == customer.id)
            .order_by(MemoryFact.id)
        ).all()

        now = datetime.now(timezone.utc)
        refusal = inspect_text(body.message)
        db.add(
            ChatMessage(
                session_id=session_id,
                customer_id=customer.id,
                role="user",
                content=mask_for_storage(body.message),
                created_at=now,
            )
        )
        db.commit()

        trace_id = uuid.uuid4().hex
        if refusal:
            steps = [{"agent": "Guard", "tool": "input", "status": "blocked", "duration_ms": 0}]
            _finish_turn(db, session_id, customer.id, trace_id, refusal, steps)
            return _chat_response(session_id, trace_id, refusal, ["Guard"], steps)

        ctx = RequestContext(
            trace_id=trace_id,
            customer_id=customer.id,
            tier=customer.tier,
            role=customer.role or "customer",
            session_id=session_id,
            db=db,
            prior_messages=[
                {"role": row.role, "content": row.content} for row in prior_rows[-20:]
            ],
            memory_facts=[row.fact for row in facts],
        )
        reply = request.app.state.supervisor.run(body.message, ctx)
        _finish_turn(db, session_id, customer.id, trace_id, reply, ctx.steps)
        return _chat_response(session_id, trace_id, reply, ctx.agents_used, ctx.steps)
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@router.get("/traces/{trace_id}")
def trace(trace_id: str, request: Request):
    db = request.app.state.session_factory()
    try:
        customer = _customer_from_cookie(request, db)
        spans = db.scalars(
            select(TraceSpan)
            .where(TraceSpan.trace_id == trace_id, TraceSpan.customer_id == customer.id)
            .order_by(TraceSpan.sequence)
        ).all()
        if not spans:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {
            "trace_id": trace_id,
            "session_id": spans[0].session_id,
            "spans": [
                {
                    "agent": span.agent,
                    "tool": span.tool,
                    "status": span.status,
                    "duration_ms": span.duration_ms,
                }
                for span in spans
            ],
        }
    finally:
        db.close()


def _finish_turn(db, session_id: str, customer_id: str, trace_id: str, reply: str, steps: list[dict]) -> None:
    db.add(
        ChatMessage(
            session_id=session_id,
            customer_id=customer_id,
            role="assistant",
            content=reply,
            created_at=datetime.now(timezone.utc),
        )
    )
    for index, step in enumerate(steps):
        db.add(
            TraceSpan(
                trace_id=trace_id,
                session_id=session_id,
                customer_id=customer_id,
                sequence=index,
                agent=step.get("agent", ""),
                tool=step.get("tool", ""),
                status=step.get("status", "ok"),
                duration_ms=float(step.get("duration_ms") or 0),
            )
        )
    db.commit()


def _chat_response(session_id: str, trace_id: str, reply: str, agents: list[str], steps: list[dict]):
    return JSONResponse(
        content={
            "session_id": session_id,
            "trace_id": trace_id,
            "reply": reply,
            "agents": agents,
            "steps": steps,
        },
        headers={"X-Trace-Id": trace_id},
    )


def _customer_from_cookie(request: Request, db) -> Customer:
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Login required")
    row = db.get(AuthSession, token)
    if row is None:
        raise HTTPException(status_code=401, detail="Login required")
    customer = db.get(Customer, row.customer_id)
    if customer is None:
        raise HTTPException(status_code=401, detail="Login required")
    return customer


def _claim_session(db, session_id: str, customer_id: str) -> None:
    owner = db.scalar(
        select(ChatMessage.customer_id).where(ChatMessage.session_id == session_id).limit(1)
    )
    if owner is not None and owner != customer_id:
        raise HTTPException(status_code=404, detail="Session not found")


def _customer_payload(customer: Customer) -> dict:
    return {
        "id": customer.id,
        "username": customer.username,
        "name": customer.name,
        "tier": customer.tier,
    }
