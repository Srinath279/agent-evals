"""Baseline registry + bootstrap regression gating (master plan §8)."""

from __future__ import annotations

from pathlib import Path

from agent_evals.baselines import load_baseline, promote_baseline
from agent_evals.core.config import load_config
from agent_evals.core.judge import MockJudge, Verdict
from agent_evals.runner import run_offline

CONFIG = Path(__file__).parent.parent / "configs" / "support_agent.yaml"


def test_promote_and_regression_detection(tmp_path):
    cfg = load_config(CONFIG)
    baselines = tmp_path / "baselines"

    good = run_offline(cfg, k=2, out_dir=tmp_path / "runs")
    promote_baseline(baselines, cfg.agent, good.out_dir)
    assert load_baseline(baselines, cfg.agent)["metrics"]["goal_success"] == [1.0] * 16

    # equal quality -> comparison present, no regression
    same = run_offline(cfg, k=2, out_dir=tmp_path / "runs", baselines_dir=baselines)
    assert same.baseline_comparison
    assert not any(c["regression"] for c in same.baseline_comparison.values())
    assert same.gate_passed

    # degraded judge metric -> significant regression flagged in the gate
    worse_judge = MockJudge(lambda r, p: Verdict(reasoning="degraded", score=0.5))
    bad = run_offline(cfg, k=2, out_dir=tmp_path / "runs",
                      judge=worse_judge, baselines_dir=baselines)
    assert bad.baseline_comparison["goal_success"]["regression"] is True
    assert not bad.gate_passed
    assert any("regression vs baseline" in f for f in bad.gate_failures)
