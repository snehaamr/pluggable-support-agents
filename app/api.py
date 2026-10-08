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
from app.db import AuthSession, ChatMessage, Customer, MemoryFact


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
                {"name": item["name"], "role": "specialist", "description": item["description"]}
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
        db.add(
            ChatMessage(
                session_id=session_id,
                customer_id=customer.id,
                role="user",
                content=body.message,
                created_at=now,
            )
        )
        db.flush()

        trace_id = uuid.uuid4().hex
        ctx = RequestContext(
            trace_id=trace_id,
            customer_id=customer.id,
            tier=customer.tier,
            session_id=session_id,
            db=db,
            prior_messages=[
                {"role": row.role, "content": row.content} for row in prior_rows[-20:]
            ],
            memory_facts=[row.fact for row in facts],
        )
        reply = request.app.state.supervisor.run(body.message, ctx)
        db.add(
            ChatMessage(
                session_id=session_id,
                customer_id=customer.id,
                role="assistant",
                content=reply,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        return JSONResponse(
            content={
                "session_id": session_id,
                "trace_id": trace_id,
                "reply": reply,
                "agents": ctx.agents_used,
                "steps": ctx.steps,
            },
            headers={"X-Trace-Id": trace_id},
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


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
