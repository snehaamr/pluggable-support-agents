import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings
from app.context import RequestContext
from app.db import Customer, MemoryFact, Refund
from app.main import create_app
from app.planning import route_message


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/support.db",
        model_provider="deterministic",
    )
    application = create_app(settings)
    with TestClient(application) as test_client:
        yield test_client


def test_health_and_agent_registry(client):
    assert client.get("/health").json() == {"status": "ok"}
    agents = {item["name"]: item["role"] for item in client.get("/agents").json()["agents"]}
    assert agents == {
        "Supervisor": "supervisor",
        "Order agent": "specialist",
        "Refund agent": "specialist",
    }


def test_order_lookup_uses_the_order_agent_and_hides_contact_details(client):
    response = client.post(
        "/chat",
        json={"customer_id": "CUST-789", "message": "What is the status of ORD-10001?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]
    assert response.headers["X-Trace-Id"] == body["trace_id"]
    assert body["agents"] == ["Supervisor", "Order agent"]
    assert [step["tool"] for step in body["steps"]] == [
        "search_agents",
        "delegate",
        "check_order_details",
    ]
    assert "ORD-10001" in body["reply"]
    assert "delivered" in body["reply"]
    assert "Laptop" in body["reply"]
    assert "example.com" not in response.text
    assert "123 Main St" not in response.text


def test_tool_payload_redacts_email_and_address(client):
    application = client.app
    db = application.state.session_factory()
    try:
        customer = db.get(Customer, "CUST-789")
        ctx = RequestContext(
            trace_id="trace",
            customer_id=customer.id,
            tier=customer.tier,
            session_id="session",
            db=db,
            current_agent="Order agent",
        )
        result = application.state.catalog.call(
            "check_order_details",
            {"order_id": "ORD-10001"},
            ctx,
            allowed=["check_order_details"],
        )
    finally:
        db.close()

    order = result["orders"][0]
    assert order["product_name"] == "Laptop"
    assert order["contact_email"] == "[REDACTED]"
    assert order["shipping_address"] == "[REDACTED]"


def test_preference_is_remembered_without_calling_a_specialist(client):
    response = client.post(
        "/chat",
        json={"customer_id": "CUST-789", "message": "I prefer email for refund updates"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == [{"agent": "Supervisor", "tool": "remember"}]
    assert "remember" in body["reply"].lower()
    db = client.app.state.session_factory()
    try:
        facts = db.scalars(select(MemoryFact).where(MemoryFact.customer_id == "CUST-789")).all()
    finally:
        db.close()
    assert any("email" in fact.fact.lower() for fact in facts)


def test_preference_and_order_request_do_both(client):
    response = client.post(
        "/chat",
        json={
            "customer_id": "CUST-789",
            "message": "I prefer email and what is the status of ORD-10001?",
        },
    )
    body = response.json()
    assert [step["tool"] for step in body["steps"]] == [
        "remember",
        "search_agents",
        "delegate",
        "check_order_details",
    ]
    assert "ORD-10001" in body["reply"]
    assert "remember" in body["reply"].lower()


def test_gold_refund_is_issued_through_the_refund_agent(client):
    response = client.post(
        "/chat",
        json={"customer_id": "CUST-789", "message": "Refund order ORD-10001"},
    )
    body = response.json()
    assert [step["tool"] for step in body["steps"]] == [
        "search_agents",
        "delegate",
        "check_eligible",
        "process_refund",
    ]
    assert "Refund agent" in body["agents"]
    assert "approved" in body["reply"].lower()
    assert "1999.99" in body["reply"]
    db = client.app.state.session_factory()
    try:
        refund = db.scalar(select(Refund).where(Refund.order_id == "ORD-10001"))
    finally:
        db.close()
    assert refund is not None
    assert f"{refund.amount:.2f}" == "1999.99"


def test_old_order_is_not_refunded(client):
    response = client.post(
        "/chat",
        json={"customer_id": "CUST-789", "message": "Refund order ORD-10002"},
    )
    body = response.json()
    assert "process_refund" not in [step["tool"] for step in body["steps"]]
    assert "not eligible" in body["reply"].lower() or "outside" in body["reply"].lower()
    db = client.app.state.session_factory()
    try:
        refund = db.scalar(select(Refund).where(Refund.order_id == "ORD-10002"))
    finally:
        db.close()
    assert refund is None


def test_customer_cannot_refund_another_customers_order(client):
    response = client.post(
        "/chat",
        json={"customer_id": "CUST-456", "message": "Refund order ORD-10001"},
    )
    body = response.json()
    assert "not found" in body["reply"].lower()
    assert "process_refund" not in [step["tool"] for step in body["steps"]]


def test_silver_refund_uses_the_tier_percentage(client):
    response = client.post(
        "/chat",
        json={"customer_id": "CUST-456", "message": "Refund order ORD-20001"},
    )
    assert "112.49" in response.json()["reply"]


def test_eligibility_list_covers_each_order(client):
    response = client.post(
        "/chat",
        json={
            "customer_id": "CUST-789",
            "message": "Which of my orders are eligible for a refund?",
        },
    )
    reply = response.json()["reply"]
    assert "ORD-10001" in reply
    assert "ORD-10002" in reply
    assert "ORD-10004" in reply
    assert "list_eligible" in [step["tool"] for step in response.json()["steps"]]


def test_unknown_customer_is_rejected(client):
    response = client.post("/chat", json={"customer_id": "NOPE", "message": "hello"})
    assert response.status_code == 404


def test_process_refund_refuses_an_ineligible_order(client):
    application = client.app
    db = application.state.session_factory()
    try:
        customer = db.get(Customer, "CUST-789")
        ctx = RequestContext(
            trace_id="trace",
            customer_id=customer.id,
            tier=customer.tier,
            session_id="session",
            db=db,
            current_agent="Refund agent",
        )
        result = application.state.catalog.call(
            "process_refund",
            {"order_id": "ORD-10002", "reason": "customer request"},
            ctx,
            allowed=["process_refund"],
        )
        db.rollback()
    finally:
        db.close()
    assert result["status"] == "ineligible"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What is the status of ORD-10001?", "order"),
        ("List all my orders", "order"),
        ("Refund order ORD-10001", "refund"),
        ("Which of my orders are eligible for a refund?", "refund"),
        ("What is the gold return window?", "refund"),
        ("I prefer email for refund updates", None),
        ("I prefer email and what is the status of ORD-10001?", "order"),
    ],
)
def test_routing(message, expected):
    assert route_message(message) == expected
