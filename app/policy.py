import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class TierRule:
    days: int
    percent: Decimal | None
    auto_refund: bool


TIER_RULES = {
    "gold": TierRule(days=30, percent=Decimal("1.00"), auto_refund=True),
    "silver": TierRule(days=20, percent=Decimal("0.75"), auto_refund=True),
    "bronze": TierRule(days=15, percent=None, auto_refund=False),
}

ELIGIBLE_STATUSES = {"delivered", "shipped", "return_requested"}


def load_policy_text() -> str:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "return_policy.txt",
        Path.cwd() / "data" / "return_policy.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("return_policy.txt was not found")


def evaluate(
    *,
    tier: str,
    status: str,
    delivered_on: date | None,
    today: date,
    amount: Decimal,
    reason: str,
    damage_note: str,
) -> dict:
    rule = TIER_RULES.get(tier)
    if rule is None:
        return _decision("ineligible", f"Tier '{tier}' has no return rule.")
    if status not in ELIGIBLE_STATUSES:
        return _decision("ineligible", f"Status '{status}' is not eligible for a refund.")
    if delivered_on is None:
        return _decision("ineligible", "This order has no delivery date.")

    age_days = (today - delivered_on).days
    if age_days > rule.days:
        return _decision(
            "ineligible",
            f"Outside the {tier} return window of {rule.days} days.",
        )
    if re.search(r"damage|defect", reason or "", re.IGNORECASE) and not damage_note:
        return _decision(
            "pending_evidence",
            "A damage note is required before this refund can be issued.",
        )
    if not rule.auto_refund:
        return _decision(
            "needs_review",
            (
                f"Inside the {tier} window of {rule.days} days. "
                "The refund amount needs a condition review, so it was not issued automatically."
            ),
        )

    refund_amount = (amount * rule.percent).quantize(Decimal("0.01"))
    return _decision(
        "eligible",
        f"Inside the {tier} return window of {rule.days} days.",
        amount=f"{refund_amount:.2f}",
    )


def _decision(status: str, message: str, amount: str | None = None) -> dict:
    return {"status": status, "message": message, "amount": amount}
