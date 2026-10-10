from datetime import date
from decimal import Decimal

from app.policy import evaluate


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
