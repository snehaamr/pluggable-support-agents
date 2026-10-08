import re

from app.agents import Agent


class Registry:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def all(self) -> list[dict]:
        return [
            {"name": agent.name, "description": agent.description}
            for agent in self._agents.values()
        ]

    def search(self, query: str) -> list[dict]:
        words = [word for word in re.findall(r"[a-z]+", query.lower()) if len(word) > 3]
        scored: list[tuple[int, str, Agent]] = []
        for agent in self._agents.values():
            haystack = f"{agent.name} {agent.description}".lower()
            score = sum(1 for word in words if word in haystack)
            scored.append((score, agent.name, agent))
        scored.sort(key=lambda item: (-item[0], item[1]))
        matched = [agent for score, _, agent in scored if score > 0]
        chosen = matched or [agent for _, _, agent in scored]
        return [{"name": agent.name, "description": agent.description} for agent in chosen]
