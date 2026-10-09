from datetime import datetime, timezone

from app.agents import Agent
from app.catalog import Tool, ToolCatalog
from app.config import Settings
from app.context import RequestContext
from app.db import MemoryFact
from app.llm import build_model
from app.registry import RegisteredAgent, Registry
from app.services import (
    ORDER_DESCRIPTION,
    REFUND_DESCRIPTION,
    AgentCaller,
)


_OBJECT = {"type": "object", "additionalProperties": False}


def build_runtime(settings: Settings, caller: AgentCaller):
    catalog = ToolCatalog()
    model = build_model(
        settings.model_provider,
        settings.model_name,
        settings.model_base_url,
        settings.model_api_key,
        settings.model_timeout_seconds,
    )
    registry = Registry()
    registry.register(RegisteredAgent("Order agent", ORDER_DESCRIPTION, settings.order_agent_url))
    registry.register(RegisteredAgent("Refund agent", REFUND_DESCRIPTION, settings.refund_agent_url))
    _add_supervisor_tools(catalog, registry, caller)
    supervisor = Agent(
        name="Supervisor",
        description="Routes customer messages to specialist agents and remembers preferences.",
        system_prompt=(
            "You are the support supervisor. A preference statement is not a request: "
            "acknowledge it with the remember tool and do not start a workflow. "
            "Do not call remember for a preference already listed in the prompt. "
            "For an order or refund request, call search_agents, then delegate. "
            "When you delegate, include the concrete order id if the conversation identifies one. "
            "Do not answer order or refund facts yourself."
        ),
        tool_names=["remember", "search_agents", "delegate"],
        model=model,
        catalog=catalog,
    )
    return supervisor, registry, catalog


def _add_supervisor_tools(catalog: ToolCatalog, registry: Registry, caller: AgentCaller) -> None:
    def search_agents(arguments: dict, ctx: RequestContext) -> dict:
        del ctx
        return {"agents": registry.search(arguments.get("query", ""))}

    def delegate(arguments: dict, ctx: RequestContext) -> dict:
        agent = registry.get(arguments.get("agent_name", ""))
        if agent is None:
            return {"reply": "No specialist is registered under that name."}
        ctx.db.commit()
        result = caller.call(
            agent.url,
            {
                "task": arguments.get("task", ""),
                "customer_id": ctx.customer_id,
                "tier": ctx.tier,
                "role": ctx.role,
                "trace_id": ctx.trace_id,
                "session_id": ctx.session_id,
                "prior_messages": ctx.prior_messages,
                "memory_facts": ctx.memory_facts,
            },
        )
        for step in result.get("steps") or []:
            ctx.steps.append(step)
        for name in result.get("agents") or []:
            if name not in ctx.agents_used:
                ctx.agents_used.append(name)
        return {"reply": result.get("reply", "")}

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
            description="Hand the customer request to a specialist agent at its registered URL.",
            parameters={
                **_OBJECT,
                "properties": {"agent_name": {"type": "string"}, "task": {"type": "string"}},
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
