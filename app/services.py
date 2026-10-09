import asyncio

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agents import Agent
from app.catalog import Tool, ToolCatalog
from app.config import Settings
from app.context import RequestContext
from app.db import make_engine, make_session_factory
from app.llm import build_model
from app.tools import check_eligible, check_order_details, list_eligible, process_refund, refund_policy


ORDER_DESCRIPTION = "Order lookup by id and listing a customer's orders."
REFUND_DESCRIPTION = "Refund eligibility, return policy, and issuing refunds."

_OBJECT = {"type": "object", "additionalProperties": False}


class RunIn(BaseModel):
    task: str
    customer_id: str
    tier: str
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
            "You are the Refund agent. Use refund_policy before deciding eligibility. "
            "Call check_eligible before process_refund. Never invent a refund. "
            "If the tool says the refund needs evidence or review, stop and say so."
        ),
        tool_names=["refund_policy", "check_eligible", "list_eligible", "process_refund"],
    )
    return _service_app(settings, agent)


def _specialist(settings: Settings, catalog: ToolCatalog, name: str, description: str, system_prompt: str, tool_names: list[str]) -> Agent:
    model = build_model(
        settings.model_provider,
        settings.model_name,
        settings.model_base_url,
        settings.model_api_key,
    )
    return Agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tool_names=tool_names,
        model=model,
        catalog=catalog,
    )


def _service_app(settings: Settings, agent: Agent) -> FastAPI:
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    app = FastAPI(title=agent.name)
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
