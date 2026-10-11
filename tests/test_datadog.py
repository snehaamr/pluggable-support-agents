import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.datadog import DatadogExporter, build_payload, intake_url
from app.main import create_app
from app.usage import TokenUsage


def test_intake_url_uses_the_datadog_site():
    assert intake_url("datadoghq.com") == "https://api.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
    assert intake_url("https://us5.datadoghq.com") == "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
    assert intake_url("app.datadoghq.eu") == "https://api.datadoghq.eu/api/intake/llm-obs/v1/trace/spans"


def test_payload_nests_tool_spans_and_records_cost():
    usage = TokenUsage()
    usage.add("gpt-4o-mini", 1000, 20)
    payload = build_payload(
        trace_id="a" * 32,
        session_id="session",
        customer_id="CUST-789",
        tier="gold",
        message="What is the gold return window?",
        reply="Gold is 30 days.",
        steps=[{"agent": "Refund agent", "tool": "refund_policy", "status": "ok", "duration_ms": 12}],
        usage=usage,
        model_name="gpt-4o-mini",
        model_provider="openai",
        now_ns=2_000_000_000,
    )
    spans = payload["data"]["attributes"]["spans"]
    root, tool, model = spans
    assert payload["data"]["attributes"]["ml_app"] == "pluggable-support-agents"
    assert root["parent_id"] == "undefined"
    assert root["meta"]["kind"] == "agent"
    assert root["meta"]["input"]["value"] == "What is the gold return window?"
    assert tool["parent_id"] == root["span_id"]
    assert tool["name"] == "refund_policy"
    assert tool["meta"]["kind"] == "tool"
    assert tool["duration"] == 12_000_000
    assert model["meta"]["kind"] == "llm"
    assert model["metrics"]["input_tokens"] == 1000
    assert model["metrics"]["output_tokens"] == 20
    assert model["metrics"]["total_cost"] == pytest.approx(0.000162)


def test_blocked_step_is_an_error_span():
    payload = build_payload(
        trace_id="b" * 32,
        session_id="session",
        customer_id="CUST-789",
        tier="gold",
        message="blocked",
        reply="refused",
        steps=[{"agent": "Guard", "tool": "input", "status": "blocked", "duration_ms": 0}],
        usage=TokenUsage(),
        model_name="gpt-4o-mini",
        model_provider="deterministic",
        now_ns=1_000,
    )
    spans = payload["data"]["attributes"]["spans"]
    assert [span["meta"]["kind"] for span in spans] == ["agent", "tool"]
    assert spans[0]["status"] == "error"
    assert spans[1]["status"] == "error"


def test_missing_api_key_does_not_send(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("Datadog was called without an API key")

    monkeypatch.setattr("app.datadog.httpx.post", fail_post)
    DatadogExporter("", "datadoghq.com").export(
        trace_id="c" * 32,
        session_id="session",
        customer_id="CUST-789",
        tier="gold",
        message="hello",
        reply="hi",
        steps=[],
        usage=TokenUsage(),
        model_name="gpt-4o-mini",
        model_provider="deterministic",
    )


def test_chat_sends_a_trace_and_survives_a_datadog_failure(tmp_path, monkeypatch):
    posts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        posts.append({"url": url, "header_present": "DD-API-KEY" in (headers or {}), "json": json})
        request = httpx.Request("POST", url)
        return httpx.Response(202, request=request)

    monkeypatch.setattr("app.datadog.httpx.post", fake_post)
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/support.db",
        model_provider="deterministic",
        agent_transport="asgi",
        datadog_api_key="test-key",
        datadog_site="us5.datadoghq.com",
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["datadog"] == "enabled"
        login = client.post("/login", json={"username": "avery", "password": "gold-pass"})
        assert login.status_code == 200
        response = client.post("/chat", json={"message": "What is the status of ORD-10001?"})
        assert response.status_code == 200
        body = response.json()
        assert "ORD-10001" in body["reply"]

        def fail_post(*args, **kwargs):
            request = httpx.Request("POST", "https://api.us5.datadoghq.com")
            response = httpx.Response(403, request=request)
            raise httpx.HTTPStatusError("forbidden", request=request, response=response)

        monkeypatch.setattr("app.datadog.httpx.post", fail_post)
        second = client.post("/chat", json={"message": "List all my orders", "session_id": body["session_id"]})
        assert second.status_code == 200

    assert posts[0]["url"] == "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
    assert posts[0]["header_present"] is True
    assert posts[0]["json"]["data"]["attributes"]["ml_app"] == "pluggable-support-agents"
    tool_names = [span["name"] for span in posts[0]["json"]["data"]["attributes"]["spans"]]
    assert "check_order_details" in tool_names
    assert "test-key" not in str(posts[0]["json"])
