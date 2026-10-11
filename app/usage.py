from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000,000 tokens. Input, cached input, and output."""

    input: Decimal
    cached: Decimal
    output: Decimal


# Published OpenAI standard rates. A dated model id uses the rate of its family.
_PRICES = {
    "gpt-4o-mini": ModelPrice(Decimal("0.15"), Decimal("0.075"), Decimal("0.60")),
    "gpt-4o": ModelPrice(Decimal("2.50"), Decimal("1.25"), Decimal("10")),
    "text-embedding-3-small": ModelPrice(Decimal("0.02"), Decimal("0.02"), Decimal("0")),
    "text-embedding-3-large": ModelPrice(Decimal("0.13"), Decimal("0.13"), Decimal("0")),
}


@dataclass
class ModelCharge:
    model: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelCharge":
        return cls(
            model=str(payload.get("model") or ""),
            prompt_tokens=int(payload.get("prompt_tokens") or 0),
            cached_tokens=int(payload.get("cached_tokens") or 0),
            completion_tokens=int(payload.get("completion_tokens") or 0),
        )


@dataclass
class TokenUsage:
    charges: list[ModelCharge] = field(default_factory=list)

    def add(self, model: str, prompt_tokens: int, completion_tokens: int = 0, cached_tokens: int = 0) -> None:
        if prompt_tokens or completion_tokens:
            self.charges.append(
                ModelCharge(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                    completion_tokens=completion_tokens,
                )
            )

    def merge(self, other: "TokenUsage") -> None:
        self.charges.extend(other.charges)

    @property
    def prompt_tokens(self) -> int:
        return sum(charge.prompt_tokens for charge in self.charges)

    @property
    def completion_tokens(self) -> int:
        return sum(charge.completion_tokens for charge in self.charges)


def price_for(model: str) -> ModelPrice | None:
    for name in sorted(_PRICES, key=len, reverse=True):
        if model == name or model.startswith(f"{name}-"):
            return _PRICES[name]
    return None


def cost_usd(usage: TokenUsage) -> Decimal | None:
    """Return the USD cost, or None when a model has no published rate."""
    if not usage.charges:
        return Decimal("0")
    total = Decimal("0")
    for charge in usage.charges:
        price = price_for(charge.model)
        if price is None:
            return None
        cached = min(max(charge.cached_tokens, 0), charge.prompt_tokens)
        billable = charge.prompt_tokens - cached
        total += (
            Decimal(billable) * price.input
            + Decimal(cached) * price.cached
            + Decimal(charge.completion_tokens) * price.output
        ) / Decimal("1000000")
    return total.quantize(Decimal("0.000001"))


def format_cost(amount: Decimal) -> str:
    quantized = amount.quantize(Decimal("0.000001"))
    if quantized == 0:
        return "0.00"
    text = f"{quantized:.6f}".rstrip("0")
    fraction = text.split(".", 1)[1]
    if len(fraction) < 2:
        return f"{quantized:.2f}"
    return text


def usage_payload(usage: TokenUsage) -> dict:
    amount = cost_usd(usage)
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": None if amount is None else format_cost(amount),
    }


def usage_from_charges(payload: dict | None) -> TokenUsage:
    usage = TokenUsage()
    for item in (payload or {}).get("charges") or []:
        charge = ModelCharge.from_dict(item)
        if charge.prompt_tokens or charge.completion_tokens:
            usage.charges.append(charge)
    return usage
