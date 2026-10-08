import json
import uuid

from app.catalog import ToolCatalog
from app.context import RequestContext
from app.llm import LanguageModel
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
        messages = [{"role": "user", "content": task}]
        try:
            for _ in range(6):
                turn = self.model.complete(
                    agent_name=self.name,
                    system=self.system_prompt,
                    messages=messages,
                    tools=self.catalog.schemas(self.tool_names),
                    context=ctx,
                )
                if not turn.tool_calls:
                    return redact_text(turn.content or "I don't have a response for that.")
                assistant_calls = []
                tool_results = []
                for call in turn.tool_calls:
                    call_id = uuid.uuid4().hex
                    result = self.catalog.call(call.name, call.arguments, ctx, allowed=self.tool_names)
                    assistant_calls.append(
                        {"id": call_id, "name": call.name, "arguments": call.arguments}
                    )
                    tool_results.append((call_id, call.name, result))
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
