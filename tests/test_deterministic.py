from __future__ import annotations

import agent_evals.evaluators  # noqa: F401
from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.evaluator import create_evaluator
from agent_evals.core.schemas import Case, ToolCall, Trace


def ev(name, **params):
    return create_evaluator(EvaluatorSpec(name=name, params=params))


def test_exact_match_normalizes_strings():
    trace = Trace(trace_id="t", output="  Billing ")
    case = Case(case_id="c", input={}, expected_output="billing")
    assert ev("exact_match").evaluate(trace, case).value == 1.0
    assert ev("exact_match", normalize=False).evaluate(trace, case).value == 0.0


def test_exact_match_field():
    trace = Trace(trace_id="t", output={"category": "billing", "reply": "hi"})
    case = Case(case_id="c", input={}, expected_output={"category": "billing"})
    assert ev("exact_match", field="category").evaluate(trace, case).value == 1.0


def test_json_schema_valid():
    schema = {"type": "object", "required": ["category"],
              "properties": {"category": {"type": "string"}}}
    good = Trace(trace_id="t", output={"category": "billing"})
    bad = Trace(trace_id="t", output={"category": 3})
    missing = Trace(trace_id="t", output={})
    not_json = Trace(trace_id="t", output="{oops")

    assert ev("json_schema_valid", schema=schema).evaluate(good).value == 1.0
    assert ev("json_schema_valid", schema=schema).evaluate(bad).value == 0.0
    assert ev("json_schema_valid", schema=schema).evaluate(missing).value == 0.0
    assert ev("json_schema_valid", schema=schema).evaluate(not_json).value == 0.0
    # fail-closed on misconfiguration
    assert ev("json_schema_valid").evaluate(good).value == 0.0


def test_tool_called_requires_success():
    trace = Trace(trace_id="t", tool_calls=[
        ToolCall(name="lookup_customer", result={"ok": 1}),
        ToolCall(name="update_ticket", error="timeout"),
    ])
    case = Case(case_id="c", input={}, expected_tools=["lookup_customer", "update_ticket"])
    score = ev("tool_called").evaluate(trace, case)
    assert score.value == 0.5
    assert "update_ticket" in score.comment


def test_thresholds_fail_closed_without_measurement():
    trace = Trace(trace_id="t")  # no latency, no cost
    assert ev("latency_threshold", threshold_ms=1000).evaluate(trace).value == 0.0
    assert ev("cost_threshold", max_usd=0.1).evaluate(trace).value == 0.0


def test_thresholds_binary():
    fast_cheap = Trace(trace_id="t", latency_ms=500, cost_usd=0.01)
    slow_pricey = Trace(trace_id="t", latency_ms=5000, cost_usd=0.5)
    assert ev("latency_threshold", threshold_ms=1000).evaluate(fast_cheap).value == 1.0
    assert ev("latency_threshold", threshold_ms=1000).evaluate(slow_pricey).value == 0.0
    assert ev("cost_threshold", max_usd=0.1).evaluate(fast_cheap).value == 1.0
    assert ev("cost_threshold", max_usd=0.1).evaluate(slow_pricey).value == 0.0
