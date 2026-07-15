"""Online sampling policy + cheap-tier scoring (master plan §9)."""

from __future__ import annotations

from pathlib import Path

from agent_evals.core.config import load_config
from agent_evals.core.schemas import ToolCall, Trace
from agent_evals.online import score_online, should_score

CONFIG = Path(__file__).parent.parent / "configs" / "support_agent.yaml"


def _clean_trace() -> Trace:
    return Trace(
        trace_id="prod-t1",
        output={"category": "billing", "priority": "high", "reply": "done"},
        tool_calls=[ToolCall(name="lookup_customer", result={"id": 1}),
                    ToolCall(name="update_ticket", result={"ok": True})],
        latency_ms=400.0,
        cost_usd=0.004,
    )


def test_suspicious_traces_always_scored():
    cfg = load_config(CONFIG)
    cfg.online_sample_rate = 0.0  # sampling off — suspicious still scores

    errored = _clean_trace()
    errored.tool_calls.append(ToolCall(name="update_ticket", error="timeout"))
    assert should_score(errored, cfg, rand=0.99) == (True, "tool_error")

    angry = _clean_trace()
    angry.metadata["user_feedback"] = 0
    assert should_score(angry, cfg, rand=0.99) == (True, "negative_feedback")

    cfg.outlier_latency_ms = 1000
    slow = _clean_trace()
    slow.latency_ms = 5000
    assert should_score(slow, cfg, rand=0.99) == (True, "latency_outlier")


def test_random_sampling_rate():
    cfg = load_config(CONFIG)
    cfg.online_sample_rate = 0.10
    trace = _clean_trace()
    assert should_score(trace, cfg, rand=0.05) == (True, "sampled")
    assert should_score(trace, cfg, rand=0.50) == (False, "not_sampled")


def test_score_online_uses_caseless_cheap_tier():
    cfg = load_config(CONFIG)
    scores, failures = score_online(_clean_trace(), cfg)

    names = {s.name for s in scores}
    assert "goal_success" not in names      # judge tier excluded by default
    assert "label_match" not in names       # requires a golden case
    assert {"tool_success_rate", "redundant_calls", "latency_threshold"} <= names
    assert not failures

    broken = _clean_trace()
    broken.tool_calls.append(ToolCall(name="update_ticket", error="boom"))
    _, failures = score_online(broken, cfg)
    assert any("tool_success_rate" in f for f in failures)
