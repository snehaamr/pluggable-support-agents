import json
from typing import Protocol

import httpx

from app.context import RequestContext
from app.planning import plan_order, plan_refund, plan_supervisor
from app.turns import ModelTurn, ToolCall


class LanguageModel(Protocol):
    def complete(
        self,
        *,
        agent_name: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        context: RequestContext,
    ) -> ModelTurn: ...


class DeterministicModel:
    """Runs the agent loop without an external model.

    The same tools and message shape are used when a live model is configured.
    """

    def complete(
        self,
        *,
        agent_name: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        context: RequestContext,
    ) -> ModelTurn:
        del system, tools
        if agent_name == "Supervisor":
            return plan_supervisor(messages, context)
        if agent_name == "Order agent":
            return plan_order(messages, context)
        if agent_name == "Refund agent":
            return plan_refund(messages, context)
        return ModelTurn(content="I can't handle that.")


class OpenAICompatibleModel:
    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: float) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        *,
        agent_name: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        context: RequestContext,
    ) -> ModelTurn:
        del agent_name, context
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, *_to_openai(messages)],
            "tools": [_tool_schema(tool) for tool in tools],
            "tool_choice": "auto",
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        tool_calls = []
        for call in message.get("tool_calls") or []:
            raw = call["function"].get("arguments") or "{}"
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            tool_calls.append(ToolCall(call["function"]["name"], arguments))
        return ModelTurn(content=message.get("content"), tool_calls=tool_calls)


def build_model(provider: str, model_name: str, base_url: str, api_key: str, timeout: float = 60.0) -> LanguageModel:
    if provider == "deterministic":
        return DeterministicModel()
    if provider == "openai":
        if not api_key:
            raise RuntimeError("MODEL_API_KEY is required when MODEL_PROVIDER=openai")
        return OpenAICompatibleModel(model_name, base_url, api_key, timeout)
    raise RuntimeError(f"Unknown MODEL_PROVIDER '{provider}'")


def _to_openai(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        if message["role"] == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )
        elif message["role"] == "assistant" and message.get("tool_calls"):
            converted.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                }
            )
        else:
            converted.append({"role": message["role"], "content": message.get("content") or ""})
    return converted


def _tool_schema(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }
