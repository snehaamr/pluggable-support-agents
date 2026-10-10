import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError

from app.agents import Agent
from app.catalog import Tool, ToolCatalog
from app.config import Settings
from app.context import RequestContext
from app.db import Base, make_engine, make_session_factory
from app.llm import build_model
from app.tools import (
    add_damage_note,
    check_eligible,
    check_order_details,
    damage_status,
    decide_review,
    list_eligible,
    list_reviews,
    open_review,
    process_refund,
    record_damage,
    refund_policy,
    refund_status,
    review_status,
)


ORDER_DESCRIPTION = "Order lookup by id and listing a customer's orders."
REFUND_DESCRIPTION = "Refund eligibility, return policy, damage claims, refund reviews, issued refunds, and issuing refunds."

_OBJECT = {"type": "object", "additionalProperties": False}


class RunIn(BaseModel):
    task: str
    customer_id: str
    tier: str
    role: str = "customer"
    trace_id: str
    session_id: str
    prior_messages: list[dict] = Field(default_factory=list)
    memory_facts: list[str] = Field(default_factory=list)


class AgentCaller:
    def call(self, url: str, payload: dict) -> dict:
        raise NotImplementedError


class HttpAgentCaller(AgentCaller):
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def call(self, url: str, payload: dict) -> dict:
        try:
            response = httpx.post(f"{url.rstrip('/')}/run", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            return {"reply": f"The specialist could not be reached at {url}.", "steps": [], "agents": [], "error": str(exc)}


class AsgiAgentCaller(AgentCaller):
    """Calls the specialist apps over HTTP without opening a port."""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        self.apps = apps

    def call(self, url: str, payload: dict) -> dict:
        app = self.apps[url]

        async def _post() -> dict:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=url) as client:
                response = await client.post("/run", json=payload)
            response.raise_for_status()
            return response.json()

        return asyncio.run(_post())


def create_order_app(settings: Settings) -> FastAPI:
    catalog = ToolCatalog()
    _add_order_tools(catalog)
    agent = _specialist(
        settings,
        catalog,
        name="Order agent",
        description=ORDER_DESCRIPTION,
        system_prompt=(
            "You are the Order agent. Use check_order_details for every order question. "
            "Return only the facts the tool gives you. Do not invent orders."
        ),
        tool_names=["check_order_details"],
    )
    return _service_app(settings, agent)


def create_refund_app(settings: Settings) -> FastAPI:
    catalog = ToolCatalog()
    _add_refund_tools(catalog)
    agent = _specialist(
        settings,
        catalog,
        name="Refund agent",
        description=REFUND_DESCRIPTION,
        system_prompt=(
            "You are the Refund agent. "
            "If the customer asks where a refund is, when it will appear, or for the status of a refund, "
            "call refund_status first. That is not a policy question and not a new refund. "
            "Do not call refund_policy, check_eligible, or process_refund for it. "
            "A return window or policy question must call refund_policy, "
            "and the reply must use only the text that tool returns. "
            "Use refund_policy before deciding eligibility. "
            "Call check_eligible before process_refund or open_review. Never invent a refund. "
            "If check_eligible says needs_review, call open_review and do not call process_refund. "
            "If check_eligible says pending_evidence, call record_damage and do not call process_refund. "
            "Arrived damaged, or defective, is not a description of what broke. Do not call add_damage_note for that. "
            "Call add_damage_note only when the customer describes the damage, such as a cracked screen, "
            "and pass their words as the note. Then call process_refund when eligible "
            "or open_review when the tier needs a review. "
            "If the customer asks about an existing damage claim, call damage_status. "
            "If the customer asks about an existing review, call review_status. "
            "A reviewer uses list_reviews and decide_review. Customers cannot decide a review."
        ),
        tool_names=[
            "refund_policy",
            "check_eligible",
            "list_eligible",
            "process_refund",
            "open_review",
            "record_damage",
            "add_damage_note",
            "damage_status",
            "refund_status",
            "review_status",
            "decide_review",
            "list_reviews",
        ],
    )
    return _service_app(settings, agent)


def _specialist(settings: Settings, catalog: ToolCatalog, name: str, description: str, system_prompt: str, tool_names: list[str]) -> Agent:
    model = build_model(
        settings.model_provider,
        settings.model_name,
        settings.model_base_url,
        settings.model_api_key,
        settings.model_timeout_seconds,
    )
    return Agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tool_names=tool_names,
        model=model,
        catalog=catalog,
    )


def _create_tables(engine) -> None:
    """Retry because several processes can create the same tables at startup."""
    last_error: Exception | None = None
    for attempt in range(15):
        try:
            Base.metadata.create_all(engine)
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(0.2 if attempt else 0.05)
    raise RuntimeError("Database tables were not created") from last_error


def _service_app(settings: Settings, agent: Agent) -> FastAPI:
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _create_tables(engine)
        yield

    app = FastAPI(title=agent.name, lifespan=lifespan)
    app.state.agent = agent
    app.state.session_factory = session_factory

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "agent": agent.name}

    @app.post("/run")
    def run(body: RunIn) -> dict:
        db = session_factory()
        try:
            ctx = RequestContext(
                trace_id=body.trace_id,
                customer_id=body.customer_id,
                tier=body.tier,
                role=body.role,
                session_id=body.session_id,
                db=db,
                prior_messages=body.prior_messages,
                memory_facts=body.memory_facts,
            )
            reply = agent.run(body.task, ctx)
            db.commit()
            return {"reply": reply, "steps": ctx.steps, "agents": ctx.agents_used}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    return app


def _add_order_tools(catalog: ToolCatalog) -> None:
    catalog.add(
        Tool(
            name="check_order_details",
            description="Look up one order, or every order for the signed-in customer.",
            parameters={
                **_OBJECT,
                "properties": {"order_id": {"type": "string"}, "customer_id": {"type": "string"}},
            },
            fn=check_order_details,
        )
    )


def _add_refund_tools(catalog: ToolCatalog) -> None:
    catalog.add(
        Tool(
            name="refund_policy",
            description="Read the matching section of the return policy.",
            parameters={
                **_OBJECT,
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
            fn=refund_policy,
        )
    )
    catalog.add(
        Tool(
            name="check_eligible",
            description="Decide whether one order can be refunded.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "damage_note": {"type": "string"},
                },
                "required": ["order_id"],
            },
            fn=check_eligible,
        )
    )
    catalog.add(
        Tool(
            name="list_eligible",
            description="Check refund eligibility for every order belonging to the customer.",
            parameters={**_OBJECT, "properties": {"reason": {"type": "string"}}},
            fn=list_eligible,
        )
    )
    catalog.add(
        Tool(
            name="process_refund",
            description="Issue a refund after eligibility has been confirmed.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "damage_note": {"type": "string"},
                },
                "required": ["order_id"],
            },
            fn=process_refund,
        )
    )
    catalog.add(
        Tool(
            name="open_review",
            description="Record a pending refund review when the tier cannot be refunded automatically.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "damage_note": {"type": "string"},
                },
                "required": ["order_id"],
            },
            fn=open_review,
        )
    )
    catalog.add(
        Tool(
            name="record_damage",
            description="Save a damage claim that is waiting for a note describing what broke.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
            fn=record_damage,
        )
    )
    catalog.add(
        Tool(
            name="add_damage_note",
            description="Save the customer's description of the damage and continue the refund.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "note": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "note"],
            },
            fn=add_damage_note,
        )
    )
    catalog.add(
        Tool(
            name="damage_status",
            description="Look up damage claims for the signed-in customer.",
            parameters={
                **_OBJECT,
                "properties": {"order_id": {"type": "string"}},
            },
            fn=damage_status,
        )
    )
    catalog.add(
        Tool(
            name="refund_status",
            description="Look up an issued refund, including its id, amount, and the date it should appear.",
            parameters={
                **_OBJECT,
                "properties": {"order_id": {"type": "string"}},
            },
            fn=refund_status,
        )
    )
    catalog.add(
        Tool(
            name="review_status",
            description="Look up pending refund reviews for the signed-in customer.",
            parameters={
                **_OBJECT,
                "properties": {"order_id": {"type": "string"}},
            },
            fn=review_status,
        )
    )
    catalog.add(
        Tool(
            name="decide_review",
            description="Approve or reject a pending refund review. Approving requires an amount and issues the refund.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "decision": {"type": "string"},
                    "amount": {"type": "string"},
                },
                "required": ["order_id", "decision"],
            },
            fn=decide_review,
        )
    )
    catalog.add(
        Tool(
            name="list_reviews",
            description="List pending refund reviews for a reviewer.",
            parameters={**_OBJECT, "properties": {}},
            fn=list_reviews,
        )
    )
