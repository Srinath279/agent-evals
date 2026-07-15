from __future__ import annotations

import pytest

from agent_evals.core.schemas import Case, ToolCall, Trace


@pytest.fixture
def triage_case() -> Case:
    return Case(
        case_id="fixture-001",
        input={"subject": "Refund request", "body": "Please refund me urgently."},
        expected_output="Route to billing with high priority.",
        expected_labels={"category": "billing", "priority": "high"},
        expected_tools=["lookup_customer", "update_ticket"],
    )


@pytest.fixture
def good_trace(triage_case: Case) -> Trace:
    return Trace(
        trace_id="trace-good",
        agent="support-agent",
        input=triage_case.input,
        output={"category": "billing", "priority": "high", "reply": "Routed to billing."},
        tool_calls=[
            ToolCall(name="lookup_customer", arguments={"email": "a@b.com"}, result={"id": 1}),
            ToolCall(name="update_ticket", arguments={"category": "billing"}, result={"ok": True}),
        ],
        steps=3,
        latency_ms=200.0,
    )


@pytest.fixture
def bad_trace(triage_case: Case) -> Trace:
    return Trace(
        trace_id="trace-bad",
        agent="support-agent",
        input=triage_case.input,
        output={"category": "general", "priority": "normal", "reply": "Thanks!"},
        tool_calls=[ToolCall(name="lookup_customer", arguments={}, result=None),
                    ToolCall(name="send_marketing_email", arguments={}, result=None)],
        steps=3,
    )
