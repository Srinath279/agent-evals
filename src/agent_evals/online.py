"""Online pipeline building blocks (master plan §9) — pure library code;
the Temporal TraceScoreWorkflow drives these from activities.

Sampling policy: `online_sample_rate` random sampling PLUS 100% of
suspicious traces — tool errors, negative user feedback, latency/cost
outliers. The suspicious ones are exactly the traces you can't afford to
skip."""

from __future__ import annotations

import random
from typing import Optional

from agent_evals.core.cache import ScoreCache
from agent_evals.core.config import AgentConfig
from agent_evals.core.judge import BaseJudge
from agent_evals.core.schemas import Score, Trace
from agent_evals.runner import _check_thresholds, build_evaluators, score_trace


def should_score(trace: Trace, cfg: AgentConfig, rand: Optional[float] = None) -> tuple[bool, str]:
    """Returns (score_it, reason). Suspicious traces are always scored."""
    if any(t.failed for t in trace.tool_calls):
        return True, "tool_error"
    feedback = trace.metadata.get("user_feedback")
    if isinstance(feedback, (int, float)) and feedback <= 0:
        return True, "negative_feedback"
    if cfg.outlier_latency_ms and trace.latency_ms and trace.latency_ms > cfg.outlier_latency_ms:
        return True, "latency_outlier"
    if cfg.outlier_cost_usd and trace.cost_usd and trace.cost_usd > cfg.outlier_cost_usd:
        return True, "cost_outlier"
    roll = rand if rand is not None else random.random()
    if roll < cfg.online_sample_rate:
        return True, "sampled"
    return False, "not_sampled"


def score_online(
    trace: Trace,
    cfg: AgentConfig,
    judge: Optional[BaseJudge] = None,
    cache: Optional[ScoreCache] = None,
    post_scores: bool = False,
) -> tuple[list[Score], list[str]]:
    """Score one production trace with the cheap-tier evaluator subset.
    Returns (scores, threshold_failures); failures make the trace an
    annotation-queue / golden-set candidate — the flywheel's intake."""
    evaluators, _ = build_evaluators(cfg, judge=judge, online=True)
    scores = score_trace(trace, None, evaluators, cache=cache)
    failures = _check_thresholds(
        scores, {m: t for m, t in cfg.score_thresholds.items()
                 if m in {s.name for s in scores}}
    )
    if post_scores:
        from agent_evals.core.store import get_store

        store = get_store(cfg.trace_store)
        for score in scores:
            store.post_score(score)
        store.flush()
    return scores, failures
