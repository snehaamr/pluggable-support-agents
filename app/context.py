from dataclasses import dataclass, field

from sqlalchemy.orm import Session


@dataclass
class RequestContext:
    trace_id: str
    customer_id: str
    tier: str
    session_id: str
    db: Session
    current_agent: str = ""
    agents_used: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    prior_messages: list[dict] = field(default_factory=list)
    memory_facts: list[str] = field(default_factory=list)
