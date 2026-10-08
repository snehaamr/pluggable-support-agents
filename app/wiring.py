from datetime import datetime, timezone

from app.agents import Agent
from app.catalog import Tool, ToolCatalog
from app.config import Settings
from app.context import RequestContext
from app.llm import build_model
from app.registry import Registry
from app.tools import check_eligible, check_order_details, list_eligible, process_refund, refund_policy
from app.db import MemoryFact


_OBJECT = {
    "type": "object",
    "additionalProperties": False,
}


def build_runtime(settings: Settings):
    catalog = ToolCatalog()
    _add_data_tools(catalog)
    model = build_model(
        settings.model_provider,
        settings.model_name,
        settings.model_base_url,
        settings.model_api_key,
    )
    order_agent = Agent(
        name="Order agent",
        description="Order lookup by id and listing a customer's orders.",
        system_prompt=(
            "You are the Order agent. Use check_order_details for every order question. "
            "Return only the facts the tool gives you. Do not invent orders."
        ),
        tool_names=["check_order_details"],
        model=model,
        catalog=catalog,
    )
    refund_agent = Agent(
        name="Refund agent",
        description="Refund eligibility, return policy, and issuing refunds.",
        system_prompt=(
            "You are the Refund agent. Read the return policy before deciding eligibility. "
            "Call check_eligible before process_refund. Never invent a refund. "
            "If the tool says the refund needs evidence or review, stop and say so."
        ),
        tool_names=["refund_policy", "check_eligible", "list_eligible", "process_refund"],
        model=model,
        catalog=catalog,
    )
    registry = Registry()
    registry.register(order_agent)
    registry.register(refund_agent)
    _add_supervisor_tools(catalog, registry)
    supervisor = Agent(
        name="Supervisor",
        description="Routes customer messages to specialist agents and remembers preferences.",
        system_prompt=(
            "You are the support supervisor. A preference statement is not a request: "
            "acknowledge it with the remember tool and do not start a workflow. "
            "For an order or refund request, search for a specialist, then delegate. "
            "Do not answer order or refund facts yourself."
        ),
        tool_names=["remember", "search_agents", "delegate"],
        model=model,
        catalog=catalog,
    )
    return supervisor, registry, catalog


def _add_data_tools(catalog: ToolCatalog) -> None:
    catalog.add(
        Tool(
            name="check_order_details",
            description="Look up one order, or every order for the signed-in customer.",
            parameters={
                **_OBJECT,
                "properties": {
                    "order_id": {"type": "string"},
                    "customer_id": {"type": "string"},
                },
            },
            fn=check_order_details,
        )
    )
    catalog.add(
        Tool(
            name="refund_policy",
            description="Read the return policy.",
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


def _add_supervisor_tools(catalog: ToolCatalog, registry: Registry) -> None:
    def search_agents(arguments: dict, ctx: RequestContext) -> dict:
        del ctx
        return {"agents": registry.search(arguments.get("query", ""))}

    def delegate(arguments: dict, ctx: RequestContext) -> dict:
        agent = registry.get(arguments.get("agent_name", ""))
        if agent is None:
            return {"reply": "No specialist is registered under that name."}
        return {"reply": agent.run(arguments.get("task", ""), ctx)}

    def remember(arguments: dict, ctx: RequestContext) -> dict:
        fact = (arguments.get("fact") or "").strip()
        if not fact:
            return {"saved": False, "message": "Nothing to remember."}
        ctx.db.add(
            MemoryFact(
                customer_id=ctx.customer_id,
                fact=fact,
                created_at=datetime.now(timezone.utc),
            )
        )
        ctx.db.flush()
        return {"saved": True}

    catalog.add(
        Tool(
            name="search_agents",
            description="Find a specialist agent for a customer request.",
            parameters={
                **_OBJECT,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=search_agents,
        )
    )
    catalog.add(
        Tool(
            name="delegate",
            description="Hand the customer request to a specialist agent.",
            parameters={
                **_OBJECT,
                "properties": {
                    "agent_name": {"type": "string"},
                    "task": {"type": "string"},
                },
                "required": ["agent_name", "task"],
            },
            fn=delegate,
        )
    )
    catalog.add(
        Tool(
            name="remember",
            description="Save a customer preference.",
            parameters={
                **_OBJECT,
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
            fn=remember,
        )
    )
