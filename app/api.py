import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.context import RequestContext
from app.db import ChatMessage, Customer


router = APIRouter()


class ChatIn(BaseModel):
    customer_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    session_id: str | None = None


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


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


@router.post("/chat")
def chat(body: ChatIn, request: Request):
    db = request.app.state.session_factory()
    try:
        customer = db.get(Customer, body.customer_id)
        if customer is None:
            raise HTTPException(status_code=404, detail="Customer not found")

        session_id = body.session_id or uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
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

        history = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        ).all()
        task = "\n".join(message.content for message in history if message.role == "user")
        ctx = RequestContext(
            trace_id=trace_id,
            customer_id=customer.id,
            tier=customer.tier,
            session_id=session_id,
            db=db,
        )
        reply = request.app.state.supervisor.run(task, ctx)
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
        payload = {
            "session_id": session_id,
            "trace_id": trace_id,
            "reply": reply,
            "agents": ctx.agents_used,
            "steps": ctx.steps,
        }
        return JSONResponse(content=payload, headers={"X-Trace-Id": trace_id})
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
