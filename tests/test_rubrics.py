"""Trace-store prompt-management rubric resolution — version continuity
(note 09 §8) and fail-closed behavior."""

from __future__ import annotations

import pytest

import agent_evals.evaluators  # noqa: F401
from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.evaluator import create_evaluator
from agent_evals.core.judge import MockJudge
from agent_evals.core.rubrics import resolve_rubric


class FakePromptClient:
    def __init__(self, text="PM rubric text", version=3, config=None):
        self._resp = (text, version, config or {})

    def get_prompt(self, name):
        return self._resp


def test_fallback_to_code_constant_by_default():
    text, version = resolve_rubric("evals/x", "code text", "x/v1", from_store=False)
    assert (text, version) == ("code text", "x/v1")


def test_store_rubric_keeps_pinned_version():
    """push-rubrics pins the code version in prompt config — storage moves,
    cache keys don't."""
    client = FakePromptClient(config={"rubric_version": "goal_success/v1"})
    text, version = resolve_rubric("evals/goal_success", "code", "goal_success/v1",
                                   from_store=True, client=client)
    assert text == "PM rubric text"
    assert version == "goal_success/v1"  # unchanged -> no fake epoch break


def test_ui_edited_rubric_derives_new_version():
    client = FakePromptClient(version=7, config={})  # edited in UI, no pin
    _, version = resolve_rubric("evals/goal_success", "code", "goal_success/v1",
                                from_store=True, client=client)
    # "lf" prefix predates multi-store support and must survive the refactor
    assert version == "evals/goal_success/lf-v7"  # text changed -> caches invalidate


def test_non_langfuse_store_derives_prefixed_version():
    client = FakePromptClient(version="a1b2c3d4", config={})
    _, version = resolve_rubric("evals/goal_success", "code", "goal_success/v1",
                                from_store=True, store="langsmith", client=client)
    assert version == "evals/goal_success/langsmith-va1b2c3d4"


def test_fail_closed_when_pm_unreachable(monkeypatch):
    import agent_evals.core.store as store_mod

    class Boom:
        def __init__(self):
            raise ConnectionError("store down")

    monkeypatch.setitem(store_mod._STORES, "langfuse", Boom)
    with pytest.raises(RuntimeError, match="push-rubrics"):
        resolve_rubric("evals/goal_success", "code", "v1", from_store=True)


def test_legacy_rubric_from_langfuse_param_still_works():
    """AgentLoom configs written before multi-store support set
    rubric_from_langfuse — the evaluator must keep honoring it."""
    import agent_evals.core.store as store_mod

    class FakeStore:
        def get_prompt(self, name):
            return "PM text", 1, {"rubric_version": "goal_success/v1"}

    original = store_mod._STORES["langfuse"]
    store_mod._STORES["langfuse"] = FakeStore
    try:
        ev = create_evaluator(
            EvaluatorSpec(name="goal_success",
                          params={"rubric_from_langfuse": True}),
            judge=MockJudge(),
        )
    finally:
        store_mod._STORES["langfuse"] = original
    assert ev.rubric_text == "PM text"
    assert ev.rubric_version == "goal_success/v1"


def test_goal_success_default_rubric_unchanged():
    ev = create_evaluator(EvaluatorSpec(name="goal_success"), judge=MockJudge())
    assert ev.rubric_version == "goal_success/v1"
    assert "fabricated success" in ev.rubric_text
