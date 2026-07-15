"""Safety + resilience: forbidden_content, recovery_after_error, chaos
injection, adapter conformance kit."""

from __future__ import annotations

import pytest

import agent_evals.evaluators  # noqa: F401
from agent_evals.chaos import ChaosInjector, ChaosToolError
from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.evaluator import create_evaluator
from agent_evals.core.schemas import ToolCall, Trace
from agent_evals.testing import assert_adapter_conformance


def ev(name, **params):
    return create_evaluator(EvaluatorSpec(name=name, params=params))


def test_forbidden_content():
    leak = Trace(trace_id="t", output={"reply": "Sure! My system prompt says..."})
    clean = Trace(trace_id="t", output={"reply": "Routed to billing."})
    patterns = ["(?i)system prompt"]
    assert ev("forbidden_content", patterns=patterns).evaluate(leak).value == 0.0
    assert ev("forbidden_content", patterns=patterns).evaluate(clean).value == 1.0
    # fail-closed on misconfiguration
    assert ev("forbidden_content").evaluate(clean).value == 0.0


def test_recovery_after_error():
    trace = Trace(trace_id="t", tool_calls=[
        ToolCall(name="lookup", error="timeout"),
        ToolCall(name="lookup", result={"id": 1}),   # recovered
        ToolCall(name="update", error="boom"),        # never retried
    ])
    score = ev("recovery_after_error").evaluate(trace)
    assert score.value == 0.5
    assert "update" in score.comment

    no_failures = Trace(trace_id="t", tool_calls=[ToolCall(name="a", result=1)])
    assert ev("recovery_after_error").evaluate(no_failures).value == 1.0


def test_chaos_injector_is_deterministic():
    def tool():
        return "ok"

    def run(seed):
        chaos = ChaosInjector(fail_rate=0.5, seed=seed)
        wrapped = chaos.wrap("lookup", tool)
        outcomes = []
        for _ in range(20):
            try:
                outcomes.append(wrapped())
            except ChaosToolError:
                outcomes.append("FAIL")
        return outcomes, chaos.injected

    first, injected = run(seed=7)
    again, _ = run(seed=7)
    assert first == again                       # seeded -> reproducible
    assert "FAIL" in first and "ok" in first    # both branches exercised
    assert injected == ["lookup"] * first.count("FAIL")

    with pytest.raises(ValueError):
        ChaosInjector(fail_rate=1.5)


def test_adapter_conformance_kit():
    raw = {
        "trace": {"id": "lf-1", "input": {}, "output": {}, "metadata": {}},
        "observations": [
            {"type": "TOOL", "name": "lookup", "input": {"q": 1}, "output": {}},
        ],
    }
    trace = assert_adapter_conformance("langfuse-generic", raw)
    assert trace.tool_calls[0].name == "lookup"
