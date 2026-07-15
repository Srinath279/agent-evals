from __future__ import annotations

import agent_evals.evaluators  # noqa: F401
from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.evaluator import create_evaluator
from agent_evals.core.schemas import Case, ToolCall, Trace


def ev(name, **params):
    return create_evaluator(EvaluatorSpec(name=name, params=params))


def test_redundant_calls_detects_identical_invocations():
    trace = Trace(trace_id="t", tool_calls=[
        ToolCall(name="search_kb", arguments={"q": "refund"}),
        ToolCall(name="search_kb", arguments={"q": "refund"}),   # identical -> redundant
        ToolCall(name="search_kb", arguments={"q": "refund policy"}),  # different args ok
    ])
    score = ev("redundant_calls").evaluate(trace)
    assert score.value == 2 / 3
    assert "search_kb" in score.comment


def test_redundant_calls_clean_trace():
    trace = Trace(trace_id="t", tool_calls=[ToolCall(name="a"), ToolCall(name="b")])
    assert ev("redundant_calls").evaluate(trace).value == 1.0


def test_tool_success_rate():
    trace = Trace(trace_id="t", tool_calls=[
        ToolCall(name="a", result=1),
        ToolCall(name="b", error="boom"),
        ToolCall(name="c", result=1),
        ToolCall(name="d", result=1),
    ])
    score = ev("tool_success_rate").evaluate(trace)
    assert score.value == 0.75
    assert "b" in score.comment


def test_steps_efficiency():
    case = Case(case_id="c", input={}, metadata={"max_steps": 4})
    assert ev("steps_efficiency").evaluate(Trace(trace_id="t", steps=3), case).value == 1.0
    assert ev("steps_efficiency").evaluate(Trace(trace_id="t", steps=8), case).value == 0.5
    assert ev("steps_efficiency", max_steps=10).evaluate(Trace(trace_id="t", steps=8)).value == 1.0
    # unconfigured -> vacuous
    assert ev("steps_efficiency").evaluate(Trace(trace_id="t", steps=99)).value == 1.0
