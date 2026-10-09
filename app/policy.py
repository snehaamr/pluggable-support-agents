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


_SECTION_TERMS = {
    "Membership tiers": (
        "tier", "gold", "silver", "bronze", "window", "day", "percent", "percentage", "refund", "return",
    ),
    "Eligible order statuses": ("status", "delivered", "shipped", "return_requested", "eligible"),
    "Damage and defects": ("damage", "damaged", "defect", "defective", "note", "cracked", "broken"),
    "Refund timing": ("when", "timing", "credit", "payment", "business", "appear", "how long"),
}


def load_policy_text() -> str:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "return_policy.txt",
        Path.cwd() / "data" / "return_policy.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("return_policy.txt was not found")


def search_policy(question: str, tier: str = "") -> str:
    """Return only the policy sections that match the question and tier."""
    sections = _parse_sections(load_policy_text())
    scored = [(_score_section(section, question), section) for section in sections]
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [section for score, section in scored if score > 0][:2]
    if not chosen and sections:
        chosen = [sections[0]]
    parts = []
    for section in chosen:
        body = section["body"]
        if section["title"] == "Membership tiers" and tier:
            body = _keep_tier(body, tier)
        parts.append(f"{section['title']}\n{body}".strip())
    return "\n\n".join(parts)


def _parse_sections(text: str) -> list[dict]:
    sections = []
    title = ""
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "RETURN POLICY":
            continue
        if not stripped.startswith("-"):
            if title:
                sections.append({"title": title, "body": "\n".join(body).strip()})
            title = stripped
            body = []
        else:
            body.append(stripped)
    if title:
        sections.append({"title": title, "body": "\n".join(body).strip()})
    return sections


def _score_section(section: dict, question: str) -> int:
    haystack = question.lower()
    score = 0
    for term in _SECTION_TERMS.get(section["title"], ()):
        if term in haystack:
            score += 2
    for word in re.findall(r"[a-z]+", section["title"].lower()):
        if len(word) > 3 and word in haystack:
            score += 3
    return score


def _keep_tier(body: str, tier: str) -> str:
    kept = []
    for line in body.splitlines():
        if line.startswith("-") and not line.lower().startswith(f"- {tier.lower()}"):
            continue
        kept.append(line)
    return "\n".join(kept)


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
