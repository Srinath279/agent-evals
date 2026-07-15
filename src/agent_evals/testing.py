"""Onboarding kit (note 06 §4): the adapter conformance check. A new
agent is onboardable when its adapter passes these invariants against a
real sampled trace — run it in the agent team's own test suite:

    from agent_evals.testing import assert_adapter_conformance
    trace = assert_adapter_conformance("my-adapter", raw_trace_dict)
    assert trace.tool_calls  # plus your agent-specific assertions
"""

from __future__ import annotations

from typing import Any

from agent_evals.core.adapters import get_adapter
from agent_evals.core.schemas import Trace


def assert_adapter_conformance(adapter_name: str, raw: dict[str, Any]) -> Trace:
    adapter = get_adapter(adapter_name)
    trace = adapter.to_trace(raw)

    assert isinstance(trace, Trace), "adapter must return the canonical Trace"
    assert trace.trace_id, "trace_id must be non-empty (score write-back needs it)"
    for tc in trace.tool_calls:
        assert tc.name, "every ToolCall needs a name (tool evaluators key on it)"
        assert isinstance(tc.arguments, dict), "ToolCall.arguments must be a dict"
    assert trace.steps >= len(trace.tool_calls), "steps must count at least the tool calls"
    if trace.latency_ms is not None:
        assert trace.latency_ms >= 0
    if trace.cost_usd is not None:
        assert trace.cost_usd >= 0
    # idempotent: adapters must be pure
    again = adapter.to_trace(raw)
    assert again.model_dump() == trace.model_dump(), "adapter must be deterministic"
    return trace
