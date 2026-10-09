import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _message(name=None, arguments=None, content=None):
    message = {"role": "assistant", "content": content}
    if name:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {})},
            }
        ]
    request = httpx.Request("POST", "http://models.test/v1/chat/completions")
    return httpx.Response(200, json={"choices": [{"message": message}]}, request=request)


def _called(payload: dict) -> list[str]:
    names = []
    for message in payload["messages"]:
        for call in message.get("tool_calls") or []:
            names.append(call["function"]["name"])
    return names


def _last_user(payload: dict) -> str:
    users = [
        message.get("content") or ""
        for message in payload["messages"]
        if message.get("role") == "user"
    ]
    return users[-1] if users else ""


@pytest.fixture
def live_client(tmp_path, monkeypatch):
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(json)
        system = json["messages"][0]["content"]
        called = _called(json)
        latest = _last_user(json)

        if "prefer email" in latest.lower():
            if "remember" not in called:
                return _message("remember", {"fact": latest})
            return _message(content="Got it. I'll remember that.")

        if "Order agent" in system:
            if "check_order_details" not in called:
                return _message("check_order_details", {"order_id": "ORD-10001"})
            return _message(content="ORD-10001 is delivered: Laptop.")

        if "Refund agent" in system:
            if "check_eligible" not in called:
                return _message("check_eligible", {"order_id": "ORD-10001", "reason": "customer request"})
            if "process_refund" not in called:
                return _message("process_refund", {"order_id": "ORD-10001", "reason": "customer request"})
            return _message(content="Refund approved for $1999.99.")

        if "Refund that laptop" in latest:
            assert "I prefer email" in system
            assert any("ORD-10001" in (message.get("content") or "") for message in json["messages"])
            if "search_agents" not in called:
                return _message("search_agents", {"query": "refund eligibility policy"})
            if "delegate" not in called:
                return _message("delegate", {"agent_name": "Refund agent", "task": "Refund that laptop"})
            return _message(content="Refund approved for $1999.99.")

        if "search_agents" not in called:
            return _message("search_agents", {"query": "order status lookup"})
        if "delegate" not in called:
            return _message(
                "delegate",
                {"agent_name": "Order agent", "task": "What is the status of ORD-10001?"},
            )
        return _message(content="ORD-10001 is delivered: Laptop.")

    monkeypatch.setattr("app.llm.httpx.post", fake_post)
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/support.db",
        model_provider="openai",
        model_api_key="test-key",
        model_base_url="http://models.test/v1",
        start_agent_services=False,
    )
    application = create_app(settings)
    with TestClient(application) as test_client:
        test_client.seen = seen
        yield test_client


def test_live_model_keeps_memory_and_resolves_a_follow_up(live_client):
    login = live_client.post("/login", json={"username": "avery", "password": "gold-pass"})
    assert login.status_code == 200

    remembered = live_client.post(
        "/chat",
        json={"message": "I prefer email for refund updates"},
    )
    assert remembered.status_code == 200
    assert remembered.json()["steps"][0]["tool"] == "remember"

    status = live_client.post(
        "/chat",
        json={
            "message": "What is the status of ORD-10001?",
            "session_id": remembered.json()["session_id"],
        },
    )
    assert status.status_code == 200
    assert "ORD-10001" in status.json()["reply"]

    refund = live_client.post(
        "/chat",
        json={
            "message": "Refund that laptop",
            "session_id": status.json()["session_id"],
        },
    )
    assert refund.status_code == 200
    body = refund.json()
    assert "1999.99" in body["reply"]
    assert [step["tool"] for step in body["steps"]] == [
        "search_agents",
        "delegate",
        "check_eligible",
        "process_refund",
    ]
    assert any("I prefer email" in payload["messages"][0]["content"] for payload in live_client.seen)
