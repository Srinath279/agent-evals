"""The score cache is the idempotency layer: retries must never
double-spend judge calls (notes 05 §A3, 08 rule 3)."""

from __future__ import annotations

import agent_evals.evaluators  # noqa: F401
from agent_evals.core.cache import ScoreCache, score_cache_key
from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.evaluator import create_evaluator
from agent_evals.core.judge import MockJudge, Verdict
from agent_evals.core.schemas import Score
from agent_evals.runner import score_trace


def test_cache_roundtrip(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite3")
    key = score_cache_key("t1", "goal_success", "goal_success/v1", "mock", "mock-judge")
    assert cache.get(key) is None
    cache.put(key, Score(name="goal_success", value=0.7, trace_id="t1"))
    cached = cache.get(key)
    assert cached is not None and cached.value == 0.7


def test_key_changes_with_rubric_and_judge():
    base = score_cache_key("t1", "goal_success", "v1", "mock", "m1")
    assert base != score_cache_key("t1", "goal_success", "v2", "mock", "m1")
    assert base != score_cache_key("t1", "goal_success", "v1", "anthropic", "m1")
    assert base != score_cache_key("t2", "goal_success", "v1", "mock", "m1")


def test_rescoring_same_trace_never_repays_judge(tmp_path, good_trace, triage_case):
    judge = MockJudge(lambda r, p: Verdict(reasoning="ok", score=1.0))
    evaluators = [create_evaluator(EvaluatorSpec(name="goal_success"), judge=judge)]
    cache = ScoreCache(tmp_path / "scores.sqlite3")

    first = score_trace(good_trace, triage_case, evaluators, cache=cache)
    second = score_trace(good_trace, triage_case, evaluators, cache=cache)  # simulated retry

    assert judge.calls == 1
    assert first[0].value == second[0].value == 1.0
