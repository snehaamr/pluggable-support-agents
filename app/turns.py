from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ModelTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
