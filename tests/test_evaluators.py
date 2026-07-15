from __future__ import annotations

import agent_evals.evaluators  # noqa: F401
from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.evaluator import available_evaluators, create_evaluator
from agent_evals.core.judge import MockJudge, Verdict


def test_builtins_registered():
    assert {"goal_success", "label_match", "tool_selection"} <= set(available_evaluators())


def test_label_match_all_fields(good_trace, triage_case):
    ev = create_evaluator(EvaluatorSpec(name="label_match", params={"fields": ["category", "priority"]}))
    score = ev.evaluate(good_trace, triage_case)
    assert score.value == 1.0
    assert score.comment == "all labels match"


def test_label_match_partial(bad_trace, triage_case):
    ev = create_evaluator(EvaluatorSpec(name="label_match", params={"fields": ["category", "priority"]}))
    score = ev.evaluate(bad_trace, triage_case)
    assert score.value == 0.0
    assert "category" in score.comment and "priority" in score.comment


def test_tool_selection_exact(good_trace, triage_case):
    ev = create_evaluator(EvaluatorSpec(name="tool_selection"))
    assert ev.evaluate(good_trace, triage_case).value == 1.0


def test_tool_selection_missing_and_unexpected(bad_trace, triage_case):
    ev = create_evaluator(EvaluatorSpec(name="tool_selection"))
    score = ev.evaluate(bad_trace, triage_case)
    # expected {lookup_customer, update_ticket}; actual {lookup_customer, send_marketing_email}
    assert score.value == 0.5
    assert "update_ticket" in score.comment and "send_marketing_email" in score.comment


def test_goal_success_stamps_judge_metadata(good_trace, triage_case):
    judge = MockJudge(lambda rubric, payload: Verdict(reasoning="goal met", score=0.9))
    ev = create_evaluator(EvaluatorSpec(name="goal_success"), judge=judge)
    score = ev.evaluate(good_trace, triage_case)
    assert score.value == 0.9
    assert score.comment == "goal met"
    assert score.metadata["judge_provider"] == "mock"
    assert score.metadata["rubric_version"] == "goal_success/v1"
    assert judge.calls == 1


def test_goal_success_payload_is_redacted(triage_case, good_trace):
    captured = {}

    def spy(rubric, payload):
        captured["payload"] = payload
        return Verdict(reasoning="ok", score=1.0)

    good_trace.input = {"body": "refund me, mail jane.doe@example.com or call 415-555-0142"}
    ev = create_evaluator(EvaluatorSpec(name="goal_success"), judge=MockJudge(spy))
    ev.evaluate(good_trace, triage_case)
    assert "jane.doe@example.com" not in captured["payload"]
    assert "<EMAIL>" in captured["payload"]
    assert "415-555-0142" not in captured["payload"]
