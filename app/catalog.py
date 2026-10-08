import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.context import RequestContext
from app.redact import redact


logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict, RequestContext], dict]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolCatalog:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self, names: list[str]) -> list[dict]:
        return [self._tools[name].schema() for name in names]

    def call(self, name: str, arguments: dict, ctx: RequestContext, allowed: list[str]) -> dict:
        if name not in allowed or name not in self._tools:
            return {"status": "error", "message": f"Tool '{name}' is not available to this agent."}

        ctx.steps.append({"agent": ctx.current_agent, "tool": name})
        logger.info("trace_id=%s agent=%s tool=%s", ctx.trace_id, ctx.current_agent, name)
        try:
            result = self._tools[name].fn(arguments or {}, ctx)
        except Exception as exc:
            logger.exception("tool %s failed", name)
            result = {"status": "error", "message": str(exc)}
        return redact(result)
