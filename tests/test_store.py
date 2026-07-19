"""Trace-store abstraction — platform switching is a config edit."""

from __future__ import annotations

import pytest

from agent_evals.core.adapters import get_adapter
from agent_evals.core.config import AgentConfig
from agent_evals.core.store import _STORES, available_stores, get_store


def test_both_platforms_registered():
    assert {"langfuse", "langsmith"} <= set(available_stores())


def test_unknown_store_lists_available():
    with pytest.raises(KeyError, match="langfuse"):
        get_store("wandb")


def test_store_classes_conform():
    """Every registered store implements the full TraceStore contract —
    a new platform can't silently ship half the surface."""
    required = {
        "load_dataset", "post_score", "seed_dataset", "get_prompt",
        "push_prompt", "fetch_trace_raw", "enqueue_annotation",
    }
    for name, cls in _STORES.items():
        missing = {m for m in required if getattr(cls, m, None) is None
                   or getattr(getattr(cls, m), "__isabstractmethod__", False)}
        assert not missing, f"store '{name}' missing {missing}"


def test_config_defaults_to_langfuse():
    cfg = AgentConfig(agent="a", local_dataset="x.jsonl")
    assert cfg.trace_store == "langfuse"
    assert cfg.trace_adapter == "langfuse-generic"


def test_langfuse_dataset_aliases_to_dataset():
    """Configs written before multi-store support keep working."""
    cfg = AgentConfig(agent="a", langfuse_dataset="golden-v1")
    assert cfg.dataset == "golden-v1"


def test_explicit_dataset_wins_over_alias():
    cfg = AgentConfig(agent="a", dataset="new", langfuse_dataset="old")
    assert cfg.dataset == "new"


def test_langsmith_generic_adapter_maps_canonical_trace():
    raw = {
        "run": {
            "id": "run-1",
            "name": "support_agent",
            "inputs": {"ticket": "refund please"},
            "outputs": {"reply": "refund issued"},
            "extra": {"metadata": {"agent": "support", "case_id": "c1"}},
            "total_cost": 0.02,
        },
        "child_runs": [
            {"run_type": "llm", "prompt_tokens": 100, "completion_tokens": 50},
            {"run_type": "tool", "name": "issue_refund",
             "inputs": {"order_id": "o-1"}, "outputs": {"ok": True}},
            {"run_type": "tool", "name": "update_ticket",
             "inputs": {"id": "t-1"}, "error": "timeout"},
        ],
    }
    trace = get_adapter("langsmith-generic").to_trace(raw)
    assert trace.trace_id == "run-1"
    assert trace.agent == "support"
    assert trace.steps == 3
    assert trace.tokens_in == 100 and trace.tokens_out == 50
    assert trace.cost_usd == 0.02
    assert [t.name for t in trace.tool_calls] == ["issue_refund", "update_ticket"]
    assert not trace.tool_calls[0].failed
    assert trace.tool_calls[1].failed
    assert trace.metadata["case_id"] == "c1"


def test_load_cases_uses_configured_store(monkeypatch):
    from agent_evals.core.schemas import Case
    from agent_evals.runner import load_cases

    class FakeStore:
        def load_dataset(self, name):
            assert name == "golden-v1"
            return [Case(case_id="c1", input="hi")]

    monkeypatch.setitem(_STORES, "langsmith", FakeStore)
    cfg = AgentConfig(agent="a", trace_store="langsmith", dataset="golden-v1")
    cases = load_cases(cfg)
    assert [c.case_id for c in cases] == ["c1"]
