import re

from app.redact import redact_text


_CARD = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")
_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) instructions"
    r"|reveal (your |the )?(system )?prompt"
    r"|disregard (the |your )?(rules|instructions)"
    r"|you are now",
    re.IGNORECASE,
)


def inspect_text(text: str) -> str | None:
    """Return a refusal when the text must not reach a model or a tool."""
    if _CARD.search(text or ""):
        return "I can't process a card number. Remove it and ask again."
    if _INJECTION.search(text or ""):
        return "I can't follow an instruction to ignore the support rules."
    return None


def inspect_payload(value) -> str | None:
    return inspect_text(_flatten(value))


def mask_for_storage(text: str) -> str:
    return redact_text(text)


def _flatten(value) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return str(value or "")
