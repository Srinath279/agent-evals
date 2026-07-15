"""Trace-replay mode: re-score stored trajectories without re-invoking the
agent (master plan §9)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_evals.core.config import load_config
from agent_evals.runner import run_offline

CONFIG = Path(__file__).parent.parent / "configs" / "support_agent.yaml"


def _raw_trace(trace_id: str, case_id: str, category: str, priority: str) -> dict:
    return {
        "trace": {
            "id": trace_id,
            "input": {"subject": "s", "body": "b"},
            "output": {"category": category, "priority": priority, "reply": "done"},
            "latency": 300.0,
            "metadata": {"agent": "support-agent", "case_id": case_id},
        },
        "observations": [
            {"type": "GENERATION", "usage": {"input": 100, "output": 30},
             "calculatedTotalCost": 0.002},
            {"type": "TOOL", "name": "lookup_customer", "input": {}, "output": {"id": 1}},
            {"type": "TOOL", "name": "update_ticket", "input": {}, "output": {"ok": True}},
        ],
    }


def test_replay_scores_stored_traces(tmp_path):
    traces = tmp_path / "prod_traces.jsonl"
    with open(traces, "w") as f:
        # ticket-001 expects billing/high; prod-2 got the priority wrong
        f.write(json.dumps(_raw_trace("prod-1", "ticket-001", "billing", "high")) + "\n")
        f.write(json.dumps(_raw_trace("prod-2", "ticket-002", "account", "high")) + "\n")

    cfg = load_config(CONFIG)
    result = run_offline(cfg, out_dir=tmp_path / "runs", mode="replay", traces_path=traces)

    assert result.k == 1 and result.n_cases == 2
    assert result.pass_rate == 0.5
    assert result.metric_means["label_match"] == 0.75  # 1.0 + 0.5(priority wrong)
    assert not result.gate_passed  # label_match mean below 0.90 threshold
    assert "label_match" in result.failure_clusters

    by_case = {cr.case_id: cr for cr in result.case_results}
    assert by_case["ticket-001"].passed
    assert not by_case["ticket-002"].passed
