"""Deterministic fake support agent so the harness runs end-to-end with
zero LLM calls. Replace `task_fn` in the config with your real agent's
invoke function; the contract is task_fn(case_input) -> Trace."""

from __future__ import annotations

import uuid
from typing import Any

from agent_evals.core.schemas import ToolCall, Trace

_CATEGORY_RULES = [
    ("billing", ["refund", "charge", "billing", "invoice", "payment"]),
    ("account", ["password", "login", "locked", "2fa", "sign in"]),
    ("technical", ["crash", "error", "bug", "broken", "not working"]),
]
_HIGH_PRIORITY = ["urgent", "asap", "immediately", "right away"]


def invoke(ticket: dict[str, Any]) -> Trace:
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()

    category = "general"
    for name, keywords in _CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            category = name
            break
    priority = "high" if any(kw in text for kw in _HIGH_PRIORITY) else "normal"

    customer = ticket.get("customer_email", "unknown")
    tool_calls = [
        ToolCall(
            name="lookup_customer",
            arguments={"email": customer},
            result={"customer_id": "cus_123", "plan": "pro"},
            duration_ms=42.0,
        ),
        ToolCall(
            name="update_ticket",
            arguments={"category": category, "priority": priority},
            result={"status": "updated"},
            duration_ms=31.0,
        ),
    ]

    return Trace(
        trace_id=f"demo-{uuid.uuid4().hex[:12]}",
        agent="support-agent",
        input=ticket,
        output={
            "category": category,
            "priority": priority,
            "reply": f"Thanks for reaching out — I've routed this to our {category} team "
                     f"with {priority} priority and updated your ticket.",
        },
        tool_calls=tool_calls,
        steps=len(tool_calls) + 1,
        latency_ms=210.0,
        cost_usd=0.004,
        tokens_in=350,
        tokens_out=90,
    )
