import re


SENSITIVE_KEYS = {
    "contact_email",
    "email",
    "shipping_address",
    "address",
    "ssn",
    "credit_card",
}

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def redact(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_KEYS:
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = redact(item)
        return cleaned
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _EMAIL.sub("[REDACTED]", value)
    return value


def redact_text(text: str) -> str:
    return _EMAIL.sub("[REDACTED]", text)
