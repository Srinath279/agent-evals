from __future__ import annotations

import pytest

from agent_evals.core.config import JudgeConfig
from agent_evals.core.judge import (
    BudgetedJudge,
    BudgetExceededError,
    FallbackJudge,
    MockJudge,
    Verdict,
    make_judge,
)


def _failing_judge():
    def boom(rubric, payload):
        raise RuntimeError("primary down")

    judge = MockJudge(boom)
    judge.provider = "primary-mock"
    judge.model = "primary-model"
    return judge


def test_fallback_judge_fails_over_and_stamps_truthfully():
    fallback = MockJudge(lambda r, p: Verdict(reasoning="fallback verdict", score=0.8))
    fallback.provider = "fallback-mock"
    fallback.model = "fallback-model"

    fj = FallbackJudge(_failing_judge(), fallback)
    verdict = fj.verdict("rubric", "payload")
    assert verdict.score == 0.8
    assert fj.provider == "fallback-mock"  # stamp reflects who actually scored
    assert fj.model == "fallback-model"


def test_fallback_judge_prefers_healthy_primary():
    primary, fallback = MockJudge(), MockJudge()
    primary.provider = "primary-mock"
    fj = FallbackJudge(primary, fallback)
    fj.verdict("r", "p")
    assert fj.provider == "primary-mock"
    assert fallback.calls == 0


def test_budgeted_judge_enforces_daily_cap(tmp_path):
    db = str(tmp_path / "budget.sqlite3")
    bj = BudgetedJudge(MockJudge(), daily_budget_usd=0.03,
                       est_cost_per_call_usd=0.01, db_path=db)
    for _ in range(3):
        bj.verdict("r", "p")
    with pytest.raises(BudgetExceededError):
        bj.verdict("r", "p")

    # the cap persists across processes (sqlite, not memory)
    bj2 = BudgetedJudge(MockJudge(), daily_budget_usd=0.03,
                        est_cost_per_call_usd=0.01, db_path=db)
    with pytest.raises(BudgetExceededError):
        bj2.verdict("r", "p")


def test_make_judge_chain():
    assert isinstance(make_judge(JudgeConfig()), MockJudge)
    # mock is never budget-wrapped (free judge)
    assert isinstance(make_judge(JudgeConfig(daily_budget_usd=5)), MockJudge)
    with_fallback = make_judge(JudgeConfig(fallback=JudgeConfig()))
    assert isinstance(with_fallback, FallbackJudge)
