"""The Temporal engine is opt-in via config (pipeline.engine). These tests run
WITHOUT the 'temporal' extra installed — they assert the local default, clean
config parsing, and that the temporal path degrades to a friendly error rather
than an ImportError. Nothing here may import temporalio at module load."""

from __future__ import annotations

import pytest

from agent_evals.core.config import AgentConfig, load_config
from agent_evals import pipelines


def _cfg(**kw) -> AgentConfig:
    base = {"agent": "t", "local_dataset": "x.jsonl"}
    base.update(kw)
    return AgentConfig(**base)


def test_engine_defaults_to_local():
    cfg = _cfg()
    assert cfg.pipeline.engine == "local"
    assert cfg.pipeline.temporal.task_queue == "agent-evals"
    assert cfg.pipeline.temporal.address == "localhost:7233"


def test_temporal_block_parses_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "agent: t\nlocal_dataset: x.jsonl\n"
        "pipeline:\n"
        "  engine: temporal\n"
        "  temporal:\n"
        "    address: temporal.prod:7233\n"
        "    task_queue: evals-prod\n"
        "    max_concurrent_activities: 16\n"
    )
    cfg = load_config(p)
    assert cfg.pipeline.engine == "temporal"
    assert cfg.pipeline.temporal.address == "temporal.prod:7233"
    assert cfg.pipeline.temporal.task_queue == "evals-prod"
    assert cfg.pipeline.temporal.max_concurrent_activities == 16


def test_unknown_engine_rejected():
    with pytest.raises(ValueError):
        _cfg(pipeline={"engine": "airflow"})


def test_is_available_matches_import(monkeypatch):
    # is_available() must never raise and must not require temporalio.
    assert isinstance(pipelines.is_available(), bool)


def test_require_temporal_raises_friendly_when_missing():
    if pipelines.is_available():
        pytest.skip("temporal extra installed in this env")
    with pytest.raises(RuntimeError, match="temporal"):
        pipelines.require_temporal()


def test_temporal_aggregation_matches_local(tmp_path):
    """The temporal path returns serialized case results; the client rebuilds
    them and calls the SAME aggregate_run() as run_offline. Prove the two
    produce identical gating/metrics without needing a cluster."""
    from agent_evals.core.config import load_config
    from agent_evals.pipelines.client import _to_case_result
    from agent_evals.runner import aggregate_run, run_offline

    cfg = load_config("configs/support_agent.yaml")

    # 1. local engine, real run
    local = run_offline(cfg, k=2, out_dir=str(tmp_path / "local"))

    # 2. simulate what the worker's score_case activity returns over the wire:
    #    full serialized scores (list of Score dicts) + trace_id, per (case, k)
    wire = [
        {
            "case_id": cr.case_id,
            "repeat_index": cr.repeat_index,
            "trace_id": cr.trace_id,
            "passed": cr.passed,
            "failures": cr.failures,
            "scores": [s.model_dump() for s in cr.scores],
        }
        for cr in local.case_results
    ]

    # 3. client-side aggregation (what submit_eval_run does after the workflow)
    rebuilt = [_to_case_result(d) for d in wire]
    remote = aggregate_run(
        cfg, rebuilt,
        run_id="wf-parity", run_dir=tmp_path / "remote",
        k=2, n_cases=local.n_cases, mode="experiment",
        judge_provider=cfg.judge.provider, judge_model=cfg.judge.model,
    )

    assert remote.metric_means == local.metric_means
    assert remote.gate_passed == local.gate_passed
    assert remote.pass_rate == local.pass_rate
    assert remote.pass_k_rate == local.pass_k_rate
    # artifacts written on the client side
    assert (tmp_path / "remote" / "report.md").exists()
    assert (tmp_path / "remote" / "scores.jsonl").exists()
    assert (tmp_path / "remote" / "manifest.json").exists()


def test_worker_refuses_local_engine_config(tmp_path):
    if not pipelines.is_available():
        pytest.skip("worker guard past require_temporal needs the extra")
    from agent_evals.pipelines.worker import run_worker

    p = tmp_path / "c.yaml"
    p.write_text("agent: t\nlocal_dataset: x.jsonl\npipeline:\n  engine: local\n")
    with pytest.raises(SystemExit):
        run_worker(str(p))
