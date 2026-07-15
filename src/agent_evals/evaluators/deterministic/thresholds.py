"""latency_threshold / cost_threshold — deterministic, binary, from trace
metadata (free — no judge). Fail-closed: a trace that doesn't record the
measurement fails the check rather than silently passing."""

from __future__ import annotations

from typing import Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class LatencyThresholdEvaluator(BaseEvaluator):
    name = "latency_threshold"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        threshold = self.params.get("threshold_ms")
        if threshold is None:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="misconfigured: 'threshold_ms' param is required")
        if trace.latency_ms is None:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="no latency recorded on trace (fail-closed)")
        ok = trace.latency_ms <= threshold
        return Score(name=self.name, value=1.0 if ok else 0.0, level=self.level,
                     comment=f"{trace.latency_ms:.0f}ms vs threshold {threshold}ms")


@register
class CostThresholdEvaluator(BaseEvaluator):
    name = "cost_threshold"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        max_usd = self.params.get("max_usd")
        if max_usd is None:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="misconfigured: 'max_usd' param is required")
        if trace.cost_usd is None:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="no cost recorded on trace (fail-closed)")
        ok = trace.cost_usd <= max_usd
        return Score(name=self.name, value=1.0 if ok else 0.0, level=self.level,
                     comment=f"${trace.cost_usd:.4f} vs max ${max_usd}")
