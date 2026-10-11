from datetime import date
from decimal import Decimal

import httpx

from app.context import RequestContext
from app.embeddings import OpenAIEmbedder
from app.policy import evaluate, search_policy
from app.tools import refund_policy
from app.usage import TokenUsage, cost_usd


TODAY = date(2026, 10, 7)


def test_gold_inside_window_is_a_full_refund():
    decision = evaluate(
        tier="gold",
        status="delivered",
        delivered_on=date(2026, 9, 27),
        today=TODAY,
        amount=Decimal("1999.99"),
        reason="customer request",
        damage_note="",
    )
    assert decision["status"] == "eligible"
    assert decision["amount"] == "1999.99"


def test_gold_outside_window_is_refused():
    decision = evaluate(
        tier="gold",
        status="delivered",
        delivered_on=date(2026, 8, 1),
        today=TODAY,
        amount=Decimal("1199.99"),
        reason="customer request",
        damage_note="",
    )
    assert decision["status"] == "ineligible"
    assert "30 days" in decision["message"]


def test_silver_refund_is_seventy_five_percent():
    decision = evaluate(
        tier="silver",
        status="delivered",
        delivered_on=date(2026, 9, 29),
        today=TODAY,
        amount=Decimal("149.99"),
        reason="customer request",
        damage_note="",
    )
    assert decision["status"] == "eligible"
    assert decision["amount"] == "112.49"


def test_bronze_stays_in_review():
    decision = evaluate(
        tier="bronze",
        status="delivered",
        delivered_on=date(2026, 10, 4),
        today=TODAY,
        amount=Decimal("24.99"),
        reason="customer request",
        damage_note="",
    )
    assert decision["status"] == "needs_review"
    assert decision["amount"] is None


def test_a_generic_damage_reason_is_not_a_note():
    decision = evaluate(
        tier="gold",
        status="delivered",
        delivered_on=date(2026, 10, 1),
        today=TODAY,
        amount=Decimal("649.99"),
        reason="damaged on arrival",
        damage_note="it arrived damaged",
    )
    assert decision["status"] == "pending_evidence"


def test_a_cracked_screen_note_allows_the_refund():
    decision = evaluate(
        tier="gold",
        status="delivered",
        delivered_on=date(2026, 10, 1),
        today=TODAY,
        amount=Decimal("649.99"),
        reason="damaged on arrival",
        damage_note="The screen is cracked",
    )
    assert decision["status"] == "eligible"
    assert decision["amount"] == "649.99"


def test_damage_without_a_note_waits_for_evidence():
    decision = evaluate(
        tier="gold",
        status="delivered",
        delivered_on=date(2026, 10, 1),
        today=TODAY,
        amount=Decimal("10.00"),
        reason="damaged on arrival",
        damage_note="",
    )
    assert decision["status"] == "pending_evidence"


def test_cancelled_orders_are_not_eligible():
    decision = evaluate(
        tier="gold",
        status="cancelled",
        delivered_on=None,
        today=TODAY,
        amount=Decimal("29.99"),
        reason="customer request",
        damage_note="",
    )
    assert decision["status"] == "ineligible"
    assert "cancelled" in decision["message"]


class _Vectors:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.pending = TokenUsage()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embeddings unavailable")
        self.pending.add("text-embedding-3-small", 500000)
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "refund timing" in lowered or "money put back" in lowered:
                vectors.append([0.0, 0.0, 0.0, 1.0])
            elif "membership tiers" in lowered or "gold return" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "eligible order" in lowered:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "damage and defects" in lowered:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 0.0])
        return vectors

    def take_usage(self) -> TokenUsage:
        taken = self.pending
        self.pending = TokenUsage()
        return taken


def test_semantic_search_picks_the_closest_section_when_keywords_do_not():
    question = "How soon is the money put back on the card?"
    keyword = search_policy(question, "gold")
    semantic = search_policy(question, "gold", embedder=_Vectors())
    assert "Membership tiers" in keyword
    assert "within 5 days" in semantic
    assert "Membership tiers" not in semantic


def test_semantic_search_keeps_only_the_callers_tier():
    text = search_policy("What is the gold return window?", "gold", embedder=_Vectors())
    assert "30-day" in text
    assert "Gold" in text
    assert "15-day" not in text
    assert "Silver" not in text


def test_embedding_failure_uses_keyword_search():
    question = "How soon is the money put back on the card?"
    text = search_policy(question, "gold", embedder=_Vectors(fail=True))
    assert text == search_policy(question, "gold")


def test_weak_embeddings_use_keyword_search():
    text = search_policy("zzzz", "gold", embedder=_Vectors())
    assert "Membership tiers" in text


def test_policy_tool_adds_embedding_cost():
    embedder = _Vectors()
    ctx = RequestContext(
        trace_id="trace",
        customer_id="CUST-789",
        tier="gold",
        session_id="session",
        db=None,
    )
    result = refund_policy(
        {"question": "How soon is the money put back on the card?"},
        ctx,
        embedder=embedder,
    )
    assert "within 5 days" in result["policy"]
    assert ctx.usage.prompt_tokens == 500000
    assert cost_usd(ctx.usage) == Decimal("0.01")


def test_embedder_reuses_vectors_and_records_tokens(monkeypatch):
    posts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        posts.append(list(json["input"]))
        data = [{"index": index, "embedding": [float(index + 1), 0.0]} for index in range(len(json["input"]))]
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"data": data, "usage": {"prompt_tokens": 12, "total_tokens": 12}},
            request=request,
        )

    monkeypatch.setattr("app.embeddings.httpx.post", fake_post)
    embedder = OpenAIEmbedder("text-embedding-3-small", "http://models.test/v1", "test-key", 5)
    first = embedder.embed(["alpha", "beta"])
    second = embedder.embed(["beta", "gamma"])
    assert posts == [["alpha", "beta"], ["gamma"]]
    assert first[1] == second[0]
    assert embedder.take_usage().prompt_tokens == 24
    assert embedder.take_usage().charges == []
