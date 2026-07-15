"""End-to-end: demo agent + golden set + mock judge through the full
runner — no network, no spend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evals.cli import main as cli_main
from agent_evals.core.config import load_config
from agent_evals.runner import load_cases, run_offline

CONFIG = Path(__file__).parent.parent / "configs" / "support_agent.yaml"


N_EVALUATORS = 9


def test_load_config_and_cases():
    cfg = load_config(CONFIG)
    assert cfg.agent == "support-agent"
    assert len(cfg.evaluators) == N_EVALUATORS
    assert cfg.evaluators[0].name == "goal_success"
    assert cfg.evaluators[1].params == {"fields": ["category", "priority"]}
    cases = load_cases(cfg)
    assert len(cases) == 8
    assert all(c.expected_labels for c in cases)


def test_run_offline_end_to_end(tmp_path):
    cfg = load_config(CONFIG)
    result = run_offline(cfg, k=2, out_dir=tmp_path)

    assert result.n_cases == 8 and result.k == 2
    assert len(result.case_results) == 16
    assert result.gate_passed, result.gate_failures
    assert result.pass_rate == 1.0 and result.pass_k_rate == 1.0
    assert result.metric_means["label_match"] == 1.0
    assert result.metric_means["tool_selection"] == 1.0

    run_dir = Path(result.out_dir)
    assert (run_dir / "report.md").exists()
    assert len((run_dir / "scores.jsonl").read_text().splitlines()) == 16 * N_EVALUATORS

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["judge_provider"] == "mock"
    assert manifest["rubric_versions"] == {"goal_success": "goal_success/v1"}
    assert manifest["k"] == 2


def test_gate_fails_on_regression(tmp_path):
    cfg = load_config(CONFIG)
    # simulate a quality bar the demo agent can't meet
    cfg.score_thresholds["label_match"] = 1.1
    result = run_offline(cfg, k=1, out_dir=tmp_path)
    assert not result.gate_passed
    assert any("label_match" in failure for failure in result.gate_failures)
    assert result.pass_rate == 0.0


def test_cli_exit_codes(tmp_path):
    assert cli_main(["run", "--config", str(CONFIG), "--k", "1", "--out", str(tmp_path)]) == 0


def test_cli_list_evaluators(capsys):
    assert cli_main(["list-evaluators"]) == 0
    out = capsys.readouterr().out
    assert "goal_success" in out and "label_match" in out


def test_missing_thresholded_metric_fails_gate(tmp_path):
    cfg = load_config(CONFIG)
    cfg.score_thresholds["nonexistent_metric"] = 0.5
    result = run_offline(cfg, k=1, out_dir=tmp_path)
    assert not result.gate_passed
    assert any("never scored" in failure for failure in result.gate_failures)


def test_retried_execution_is_idempotent(tmp_path):
    """A retried (run, case, repeat) must hit the score cache even though the
    demo agent mints a random trace ID per invocation (note 09 §1)."""
    from agent_evals.core.cache import ScoreCache
    from agent_evals.core.judge import MockJudge
    from agent_evals.runner import build_evaluators, resolve_task_fn, run_single_case

    cfg = load_config(CONFIG)
    judge = MockJudge()
    evaluators, _ = build_evaluators(cfg, judge=judge)
    task_fn = resolve_task_fn(cfg.task_fn)
    case = load_cases(cfg)[0]
    cache = ScoreCache(tmp_path / "scores.sqlite3")

    first = run_single_case(case, task_fn, evaluators, cfg, cache, repeat_index=0, run_id="run-x")
    retry = run_single_case(case, task_fn, evaluators, cfg, cache, repeat_index=0, run_id="run-x")

    assert judge.calls == 1  # second execution served from cache
    assert first.trace_id == retry.trace_id == "run-x/ticket-001/r0"
    assert first.passed and retry.passed


def test_unknown_evaluator_raises():
    cfg = load_config(CONFIG)
    cfg.evaluators[0].name = "does_not_exist"
    with pytest.raises(KeyError, match="does_not_exist"):
        run_offline(cfg, k=1)
