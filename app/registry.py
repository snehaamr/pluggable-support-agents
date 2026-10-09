import re
from dataclasses import dataclass


@dataclass
class RegisteredAgent:
    name: str
    description: str
    url: str


class Registry:
    def __init__(self) -> None:
        self._agents: dict[str, RegisteredAgent] = {}

    def register(self, agent: RegisteredAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> RegisteredAgent | None:
        return self._agents.get(name)

    def all(self) -> list[dict]:
        return [_public(agent) for agent in self._agents.values()]

    def search(self, query: str) -> list[dict]:
        words = [word for word in re.findall(r"[a-z]+", query.lower()) if len(word) > 3]
        scored: list[tuple[int, str, RegisteredAgent]] = []
        for agent in self._agents.values():
            haystack = f"{agent.name} {agent.description}".lower()
            score = sum(1 for word in words if word in haystack)
            scored.append((score, agent.name, agent))
        scored.sort(key=lambda item: (-item[0], item[1]))
        matched = [agent for score, _, agent in scored if score > 0]
        chosen = matched or [agent for _, _, agent in scored]
        return [_public(agent) for agent in chosen]


def _public(agent: RegisteredAgent) -> dict:
    return {"name": agent.name, "description": agent.description, "url": agent.url}
