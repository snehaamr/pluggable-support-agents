import json
import re
import uuid

from app.catalog import ToolCatalog
from app.context import RequestContext
from app.llm import LanguageModel
from app.planning import _refund_status_reply, _wants_refund_status, format_eligibility
from app.redact import redact_text


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tool_names: list[str],
        model: LanguageModel,
        catalog: ToolCatalog,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tool_names = tool_names
        self.model = model
        self.catalog = catalog

    def run(self, task: str, ctx: RequestContext) -> str:
        previous = ctx.current_agent
        ctx.current_agent = self.name
        if self.name not in ctx.agents_used:
            ctx.agents_used.append(self.name)
        messages = [dict(message) for message in ctx.prior_messages]
        messages.append({"role": "user", "content": task})
        try:
            if self.name == "Refund agent" and _wants_refund_status(task):
                found = re.search(r"ORD-\d+", task)
                arguments = {"order_id": found.group(0)} if found else {}
                result = self.catalog.call("refund_status", arguments, ctx, allowed=self.tool_names)
                return redact_text(_refund_status_reply(result))
            used_tool = False
            for _ in range(8):
                turn = self.model.complete(
                    agent_name=self.name,
                    system=self._system(ctx),
                    messages=messages,
                    tools=self.catalog.schemas(self.tool_names),
                    context=ctx,
                    force_tool=not used_tool,
                )
                if not turn.tool_calls:
                    content = _grounded_reply(messages, turn.content or "I don't have a response for that.")
                    return redact_text(content)
                assistant_calls = []
                tool_results = []
                for call in turn.tool_calls:
                    call_id = uuid.uuid4().hex
                    arguments = dict(call.arguments)
                    if call.name == "delegate" and self.name == "Supervisor" and _wants_refund_status(task):
                        arguments["task"] = task
                    result = self.catalog.call(call.name, arguments, ctx, allowed=self.tool_names)
                    assistant_calls.append(
                        {"id": call_id, "name": call.name, "arguments": call.arguments}
                    )
                    tool_results.append((call_id, call.name, result))
                    used_tool = True
                messages.append(
                    {"role": "assistant", "content": turn.content, "tool_calls": assistant_calls}
                )
                for call_id, name, result in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )
            return "I couldn't finish that request."
        finally:
            ctx.current_agent = previous

    def _system(self, ctx: RequestContext) -> str:
        lines = [self.system_prompt]
        if ctx.role == "reviewer":
            lines.append(
                "The signed-in user is a reviewer. "
                "For a refund review, call search_agents and delegate if you are the supervisor. "
                "If you are the Refund agent, call list_reviews or decide_review. "
                "Do not issue a refund except through decide_review."
            )
        if ctx.memory_facts:
            lines.append("Customer preferences already saved:")
            lines.extend(f"- {fact}" for fact in ctx.memory_facts)
        lines.append(
            "Earlier messages in this conversation are included. "
            "Resolve references such as 'that laptop' or 'it' from those messages before calling a tool."
        )
        return "\n".join(lines)


def _grounded_reply(messages: list[dict], content: str) -> str:
    """Use the tool result when a later summary can drop or swap an order."""
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        payload = json.loads(message["content"])
        if message.get("name") == "list_eligible":
            return format_eligibility(payload)
        if message.get("name") == "delegate" and payload.get("reply"):
            reply = payload["reply"]
            if _remembered(messages):
                return f"I'll remember that. {reply}".strip()
            return reply
    return content


def _remembered(messages: list[dict]) -> bool:
    for message in messages:
        if message.get("role") == "tool" and message.get("name") == "remember":
            if json.loads(message["content"]).get("saved"):
                return True
    return False
