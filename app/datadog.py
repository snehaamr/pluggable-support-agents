import logging
import time
import uuid

import httpx

from app.usage import TokenUsage, cost_usd


logger = logging.getLogger(__name__)

ML_APP = "pluggable-support-agents"


class DatadogExporter:
    """Sends one chat trace to Datadog LLM Observability.

    With no API key, export does nothing. A failed send does not change the chat.
    """

    def __init__(self, api_key: str, site: str, timeout: float = 5.0) -> None:
        self.api_key = api_key
        self.site = site
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def export(
        self,
        *,
        trace_id: str,
        session_id: str,
        customer_id: str,
        tier: str,
        message: str,
        reply: str,
        steps: list[dict],
        usage: TokenUsage,
        model_name: str,
        model_provider: str,
    ) -> None:
        if not self.enabled:
            return
        payload = build_payload(
            trace_id=trace_id,
            session_id=session_id,
            customer_id=customer_id,
            tier=tier,
            message=message,
            reply=reply,
            steps=steps,
            usage=usage,
            model_name=model_name,
            model_provider=model_provider,
        )
        response = httpx.post(
            intake_url(self.site),
            headers={"DD-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()


def intake_url(site: str) -> str:
    host = (site or "").strip()
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    host = host.strip("/")
    for prefix in ("app.", "api."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    if not host:
        host = "datadoghq.com"
    return f"https://api.{host}/api/intake/llm-obs/v1/trace/spans"


def build_payload(
    *,
    trace_id: str,
    session_id: str,
    customer_id: str,
    tier: str,
    message: str,
    reply: str,
    steps: list[dict],
    usage: TokenUsage,
    model_name: str,
    model_provider: str,
    now_ns: int | None = None,
) -> dict:
    now = now_ns if now_ns is not None else time.time_ns()
    durations = [int(float(step.get("duration_ms") or 0) * 1_000_000) for step in steps]
    total = sum(durations)
    start = now - total
    root_id = _span_id()
    shared_trace = _trace_id(trace_id)
    failed = any((step.get("status") or "ok") in {"error", "blocked"} for step in steps)
    spans = [
        {
            "name": "support-chat",
            "span_id": root_id,
            "trace_id": shared_trace,
            "parent_id": "undefined",
            "start_ns": start,
            "duration": float(total),
            "status": "error" if failed else "ok",
            "session_id": session_id,
            "meta": {
                "kind": "agent",
                "input": {"value": message},
                "output": {"value": reply},
                "metadata": {"customer_id": customer_id, "tier": tier},
            },
        }
    ]
    cursor = start
    for step, duration in zip(steps, durations):
        status = step.get("status") or "ok"
        spans.append(
            {
                "name": step.get("tool") or "tool",
                "span_id": _span_id(),
                "trace_id": shared_trace,
                "parent_id": root_id,
                "start_ns": cursor,
                "duration": float(duration),
                "status": "error" if status in {"error", "blocked"} else "ok",
                "meta": {
                    "kind": "tool",
                    "metadata": {"agent": step.get("agent") or "", "status": status},
                },
            }
        )
        cursor += duration
    if usage.prompt_tokens or usage.completion_tokens:
        metrics = {
            "input_tokens": float(usage.prompt_tokens),
            "output_tokens": float(usage.completion_tokens),
            "total_tokens": float(usage.prompt_tokens + usage.completion_tokens),
        }
        amount = cost_usd(usage)
        if amount is not None:
            metrics["total_cost"] = float(amount)
        spans.append(
            {
                "name": model_name,
                "span_id": _span_id(),
                "trace_id": shared_trace,
                "parent_id": root_id,
                "start_ns": start,
                "duration": float(total),
                "status": "ok",
                "meta": {
                    "kind": "llm",
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
                "metrics": metrics,
            }
        )
    return {
        "data": {
            "type": "span",
            "attributes": {
                "ml_app": ML_APP,
                "session_id": session_id,
                "tags": [f"tier:{tier}", f"customer_id:{customer_id}"],
                "spans": spans,
            },
        }
    }


def _span_id() -> str:
    return str(uuid.uuid4().int & ((1 << 63) - 1) or 1)


def _trace_id(trace_id: str) -> str:
    try:
        return str(int(trace_id[:16], 16) or 1)
    except ValueError:
        return _span_id()
